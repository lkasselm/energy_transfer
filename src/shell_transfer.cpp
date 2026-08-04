#include "energy_transfer/shell_transfer.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <tuple>

#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

#include "energy_transfer/convert.hpp"
#include "energy_transfer/spectral_kernels.hpp"

namespace energy_transfer {

namespace {

std::vector<Real> BuildShellEdges(const BinningSpec &binning, int Nx) {
  std::vector<Real> edges;
  switch (binning.type) {
  case BinningSpec::Type::Linear: {
    edges.push_back(0.5);
    for (int i = 1; i < binning.num_shells; i++) edges.push_back(0.5 + i);
    edges.push_back(Real(Nx) / 2.0 * std::sqrt(3.0));
    break;
  }
  case BinningSpec::Type::Log: {
    edges.push_back(0.0);
    const Real resolution_exp = std::log(Real(Nx) / 8.0) / std::log(2.0) * 4.0 + 1.0;
    const int n_log_bins = static_cast<int>(resolution_exp) + 1;
    for (int i = 0; i <= n_log_bins; i++) {
      edges.push_back(4.0 * std::pow(2.0, (Real(i) - 1.0) / 4.0));
    }
    break;
  }
  case BinningSpec::Type::Custom:
    PARTHENON_REQUIRE_THROWS(binning.custom_edges.size() >= 2,
                             "BinningSpec::Custom needs at least 2 edges");
    edges = binning.custom_edges;
    break;
  }
  return edges;
}

std::set<std::string> QuantityNamesForTerms(const std::vector<TermRequest> &terms) {
  const auto &term_table = BuiltinTerms();
  std::set<std::string> names;
  for (auto &tr : terms) {
    auto it = term_table.find(tr.name);
    PARTHENON_REQUIRE_THROWS(it != term_table.end(),
                             "energy_transfer: unknown term '" + tr.name + "'");
    names.insert(it->second.q_side_quantity);
    names.insert(it->second.k_side_quantity);
  }
  return names;
}

std::set<std::string> BaseFieldClosure(const std::set<std::string> &quantity_names) {
  const auto &qtable = BuiltinQuantities();
  std::set<std::string> tags;
  for (auto &qn : quantity_names) {
    auto it = qtable.find(qn);
    PARTHENON_REQUIRE_THROWS(it != qtable.end(),
                             "energy_transfer: unknown derived quantity '" + qn + "'");
    for (auto &t : it->second.required_base_fields) tags.insert(t);
  }
  return tags;
}

void CheckRuntimeConstraints(parthenon::Mesh *pmesh) {
  PARTHENON_REQUIRE_THROWS(pmesh->DefaultNumPartitions() == 1,
                           "energy_transfer: only pack_size=-1 (a single mesh partition) is "
                           "currently supported.");

  const auto &mesh_size = pmesh->mesh_size;
  const Real Lx = mesh_size.xmax(parthenon::X1DIR) - mesh_size.xmin(parthenon::X1DIR);
  const Real Ly = mesh_size.xmax(parthenon::X2DIR) - mesh_size.xmin(parthenon::X2DIR);
  const Real Lz = mesh_size.xmax(parthenon::X3DIR) - mesh_size.xmin(parthenon::X3DIR);
  PARTHENON_REQUIRE_THROWS(std::abs(Lx - Ly) < 1e-10 * Lx && std::abs(Ly - Lz) < 1e-10 * Ly,
                           "energy_transfer: the domain must be cubic (Lx == Ly == Lz).");

  for (auto bc : pmesh->mesh_bcs) {
    PARTHENON_REQUIRE_THROWS(bc == parthenon::BoundaryFlag::periodic,
                             "energy_transfer: all boundary conditions must be periodic.");
  }
}

} // namespace

FieldRequirements ComputeFieldRequirements(const ShellTransferConfig &cfg) {
  auto qnames = QuantityNamesForTerms(cfg.terms);
  auto tags = BaseFieldClosure(qnames);

  FieldRequirements req;
  req.mag = tags.count("B") > 0 || tags.count("b") > 0;
  req.pres_or_energy = tags.count("P") > 0;
  req.acc = tags.count("Acc") > 0;

  const auto &spec_table = BuiltinSpectra();
  for (auto &name : cfg.spectrum_names) {
    PARTHENON_REQUIRE_THROWS(spec_table.count(name) > 0,
                             "energy_transfer: unknown spectrum '" + name + "'");
  }
  if (std::find(cfg.spectrum_names.begin(), cfg.spectrum_names.end(), "spec_B") !=
      cfg.spectrum_names.end()) {
    req.mag = true;
  }
  return req;
}

TransferResult ComputeShellTransfer(parthenon::Mesh *pmesh, FlatFields &fields,
                                    const ShellTransferConfig &cfg) {
  PARTHENON_REQUIRE_THROWS(!fields.is_conserved,
                           "energy_transfer: FlatFields must be in primitive form -- call "
                           "ConvertConservedToPrimitive() first.");
  CheckRuntimeConstraints(pmesh);

  auto FFTMgr = pmesh->GetFFTManager();
  const auto fft_size_inbox = FFTMgr->size_real_space_box();
  const auto fft_size_outbox = FFTMgr->size_fourier_space_box();

  const auto &mesh_size = pmesh->mesh_size;
  const int Nx = mesh_size.nx(parthenon::X1DIR);
  const int Ny = mesh_size.nx(parthenon::X2DIR);
  const int Nz = mesh_size.nx(parthenon::X3DIR);
  const Real Lx = mesh_size.xmax(parthenon::X1DIR) - mesh_size.xmin(parthenon::X1DIR);
  const Real two_pi_over_L = 2.0 * M_PI / Lx;

  const auto shell_edges = BuildShellEdges(cfg.binning, Nx);
  const int n_shells = static_cast<int>(shell_edges.size()) - 1;
  const Real collapsed_k_low = shell_edges.front();
  const Real collapsed_k_high = shell_edges.back();

  const auto qnames = QuantityNamesForTerms(cfg.terms);
  const auto tags = BaseFieldClosure(qnames);
  const bool needs_U = tags.count("U") > 0;
  const bool needs_B = tags.count("B") > 0;
  const bool needs_P = tags.count("P") > 0;
  const bool needs_Acc = tags.count("Acc") > 0;
  const bool needs_b = tags.count("b") > 0;
  const bool needs_DivU = tags.count("DivU") > 0;
  const bool needs_Divb = tags.count("Divb") > 0;

  if (needs_B || needs_b) {
    PARTHENON_REQUIRE_THROWS(fields.mag.size() > 0,
                             "energy_transfer: requested terms need the magnetic field, but "
                             "FlatFields::mag is not populated.");
  }
  if (needs_P) {
    PARTHENON_REQUIRE_THROWS(fields.pres_or_energy.size() > 0,
                             "energy_transfer: requested terms need pressure, but "
                             "FlatFields::pres_or_energy is not populated.");
  }
  if (needs_Acc) {
    PARTHENON_REQUIRE_THROWS(fields.acc.size() > 0,
                             "energy_transfer: requested terms need the acceleration field, "
                             "but FlatFields::acc is not populated.");
  }

  // W = sqrt(rho) * velocity -- computed unconditionally since FT_W is always needed.
  auto rho = fields.rho;
  auto vel = fields.mom_or_vel;
  parthenon::ParArray1D<Real> W_flat("W_flat", 3 * fft_size_inbox);
  parthenon::par_for(
      "ComputeW", std::size_t(0), fft_size_inbox - 1,
      KOKKOS_LAMBDA(const std::size_t idx) {
        const Real sqrt_rho = Kokkos::sqrt(rho(idx));
        for (int n = 0; n < 3; n++) {
          W_flat(n * fft_size_inbox + idx) = sqrt_rho * vel(n * fft_size_inbox + idx);
        }
      });
  Kokkos::fence();

  FourierFields ft;
  ft.FT_W = parthenon::ParArray1D<Kokkos::complex<Real>>("FT_W", 3 * fft_size_outbox);
  for (int n = 0; n < 3; n++) {
    FFTMgr->Forward(W_flat.data() + n * fft_size_inbox, ft.FT_W.data() + n * fft_size_outbox);
  }
  if (needs_U) {
    ft.FT_U = parthenon::ParArray1D<Kokkos::complex<Real>>("FT_U", 3 * fft_size_outbox);
    for (int n = 0; n < 3; n++) {
      FFTMgr->Forward(vel.data() + n * fft_size_inbox, ft.FT_U.data() + n * fft_size_outbox);
    }
  }
  if (needs_B) {
    auto mag = fields.mag;
    ft.FT_B = parthenon::ParArray1D<Kokkos::complex<Real>>("FT_B", 3 * fft_size_outbox);
    for (int n = 0; n < 3; n++) {
      FFTMgr->Forward(mag.data() + n * fft_size_inbox, ft.FT_B.data() + n * fft_size_outbox);
    }
  }
  if (needs_P) {
    ft.FT_P = parthenon::ParArray1D<Kokkos::complex<Real>>("FT_P", fft_size_outbox);
    FFTMgr->Forward(fields.pres_or_energy.data(), ft.FT_P.data());
  }
  if (needs_Acc) {
    auto acc = fields.acc;
    ft.FT_Acc = parthenon::ParArray1D<Kokkos::complex<Real>>("FT_Acc", 3 * fft_size_outbox);
    for (int n = 0; n < 3; n++) {
      FFTMgr->Forward(acc.data() + n * fft_size_inbox, ft.FT_Acc.data() + n * fft_size_outbox);
    }
  }

  GlobalAux aux;
  if (needs_b) {
    auto mag = fields.mag;
    aux.b_flat = parthenon::ParArray1D<Real>("b_flat", 3 * fft_size_inbox);
    auto b_flat = aux.b_flat;
    parthenon::par_for(
        "ComputeSmallB", std::size_t(0), fft_size_inbox - 1,
        KOKKOS_LAMBDA(const std::size_t idx) {
          const Real inv_sqrt_rho = 1.0 / Kokkos::sqrt(rho(idx));
          for (int n = 0; n < 3; n++) {
            b_flat(n * fft_size_inbox + idx) = mag(n * fft_size_inbox + idx) * inv_sqrt_rho;
          }
        });
    Kokkos::fence();
  }
  if (needs_DivU) {
    PARTHENON_REQUIRE_THROWS(needs_U, "energy_transfer: internal error -- DivU requires FT_U");
    aux.DivU = parthenon::ParArray1D<Real>("DivU", fft_size_inbox);
    parthenon::ParArray1D<Kokkos::complex<Real>> scratch("DivU_scratch", fft_size_outbox);
    SpectralDivergence(FFTMgr, ft.FT_U, scratch, aux.DivU, two_pi_over_L);
  }
  if (needs_Divb) {
    PARTHENON_REQUIRE_THROWS(needs_b, "energy_transfer: internal error -- Divb requires b_flat");
    parthenon::ParArray1D<Kokkos::complex<Real>> FT_b("FT_b_tmp", 3 * fft_size_outbox);
    for (int n = 0; n < 3; n++) {
      FFTMgr->Forward(aux.b_flat.data() + n * fft_size_inbox, FT_b.data() + n * fft_size_outbox);
    }
    aux.Divb = parthenon::ParArray1D<Real>("Divb", fft_size_inbox);
    parthenon::ParArray1D<Kokkos::complex<Real>> scratch("Divb_scratch", fft_size_outbox);
    SpectralDivergence(FFTMgr, FT_b, scratch, aux.Divb, two_pi_over_L);
  }

  ShellWorkspace ws_template;
  ws_template.fft_mgr = FFTMgr;
  ws_template.fields = &fields;
  ws_template.ft = &ft;
  ws_template.aux = &aux;
  ws_template.fft_size_inbox = fft_size_inbox;
  ws_template.fft_size_outbox = fft_size_outbox;
  ws_template.Nx = Nx;
  ws_template.Ny = Ny;
  ws_template.Nz = Nz;
  ws_template.two_pi_over_L = two_pi_over_L;

  const auto &quantity_table = BuiltinQuantities();
  const auto &term_table = BuiltinTerms();

  // Memoizes derived-quantity evaluations by (name, k_low, k_high) so terms
  // that share a quantity (e.g. "W_filter" as the K-side of UU/BUT/BUPbb/
  // PU/FU) only pay for it once per shell within this call.
  using CacheKey = std::tuple<std::string, Real, Real>;
  std::map<CacheKey, parthenon::ParArray1D<Real>> cache;
  auto evaluate = [&](const std::string &qname, Real k_low, Real k_high) {
    CacheKey key{qname, k_low, k_high};
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    ShellWorkspace ws = ws_template;
    ws.k_low = k_low;
    ws.k_high = k_high;
    auto result = quantity_table.at(qname).fn(ws);
    cache.emplace(key, result);
    return result;
  };

  TransferResult result;
  result.n_shells = n_shells;
  result.binning = cfg.binning;
  result.shell_edges = shell_edges;

  for (auto &tr : cfg.terms) {
    const auto &term = term_table.at(tr.name);
    const bool q_resolved =
        tr.mode == DecompositionMode::Full || tr.mode == DecompositionMode::BySender;
    const bool k_resolved =
        tr.mode == DecompositionMode::Full || tr.mode == DecompositionMode::ByReceiver;
    const int n_q = q_resolved ? n_shells : 1;
    const int n_k = k_resolved ? n_shells : 1;

    parthenon::HostArray2D<TransferReal> matrix(tr.name, n_q, n_k);
    for (int qi = 0; qi < n_q; qi++) {
      const Real q_low = q_resolved ? shell_edges[qi] : collapsed_k_low;
      const Real q_high = q_resolved ? shell_edges[qi + 1] : collapsed_k_high;
      auto q_side = evaluate(term.q_side_quantity, q_low, q_high);
      for (int ki = 0; ki < n_k; ki++) {
        const Real k_low = k_resolved ? shell_edges[ki] : collapsed_k_low;
        const Real k_high = k_resolved ? shell_edges[ki + 1] : collapsed_k_high;
        auto k_side = evaluate(term.k_side_quantity, k_low, k_high);
        matrix(qi, ki) = term.prefactor * DotProductReduce(k_side, q_side, 3 * fft_size_inbox);
      }
    }
    result.matrices.emplace(tr.name, matrix);
  }

  const auto &spectrum_table = BuiltinSpectra();
  for (auto &name : cfg.spectrum_names) {
    result.spectra.emplace(name, spectrum_table.at(name).fn(pmesh, fields, W_flat));
  }

  return result;
}

TransferResult ComputeShellTransferLive(parthenon::Mesh *pmesh, parthenon::MeshData<Real> *md,
                                        const LiveFieldSpec &spec,
                                        const ShellTransferConfig &cfg) {
  const auto req = ComputeFieldRequirements(cfg);

  LiveFieldSpec effective = spec;
  if (!req.mag) {
    effective.magnetic_var.reset();
    effective.magnetic_components.reset();
  }
  if (!req.pres_or_energy) {
    effective.pressure_or_energy_var.reset();
    effective.pressure_or_energy_component.reset();
  }
  if (!req.acc) {
    effective.acceleration_var.reset();
    effective.acceleration_components.reset();
  }
  // Total energy includes the magnetic contribution, so converting it to
  // pressure always requires the magnetic field, even if no B-dependent
  // term was requested -- mirrors driver.cpp:310-311.
  if (spec.is_conserved && req.pres_or_energy && !effective.magnetic_var) {
    PARTHENON_REQUIRE_THROWS(
        spec.magnetic_var.has_value(),
        "ComputeShellTransferLive: converting conserved total energy to pressure requires "
        "the magnetic field -- populate LiveFieldSpec::magnetic_var even though no "
        "B-dependent term was requested.");
    effective.magnetic_var = spec.magnetic_var;
    effective.magnetic_components = spec.magnetic_components;
  }

  auto fields = GatherLiveFields(pmesh, md, effective);
  ConvertConservedToPrimitive(fields);
  return ComputeShellTransfer(pmesh, fields, cfg);
}

TransferResult ComputeShellTransferFromFile(parthenon::Mesh *pmesh,
                                            const std::string &input_file,
                                            const ADIOS2FieldNaming &naming,
                                            const ShellTransferConfig &cfg) {
  const auto req = ComputeFieldRequirements(cfg);
  PARTHENON_REQUIRE_THROWS(!req.mag || naming.mag.has_value(),
                           "ComputeShellTransferFromFile: requested terms need the magnetic "
                           "field, but naming.mag is not set.");
  PARTHENON_REQUIRE_THROWS(!req.pres_or_energy || naming.pres_or_energy.has_value(),
                           "ComputeShellTransferFromFile: requested terms need "
                           "pressure/energy, but naming.pres_or_energy is not set.");
  PARTHENON_REQUIRE_THROWS(!req.acc || naming.acc.has_value(),
                           "ComputeShellTransferFromFile: requested terms need the "
                           "acceleration field, but naming.acc is not set.");
  PARTHENON_REQUIRE_THROWS(!(naming.input_conserved && req.pres_or_energy) ||
                               naming.mag.has_value(),
                           "ComputeShellTransferFromFile: converting conserved total energy "
                           "to pressure requires naming.mag to be set.");

  auto fields = ReadADIOS2Fields(pmesh, input_file, naming);
  ConvertConservedToPrimitive(fields);
  return ComputeShellTransfer(pmesh, fields, cfg);
}

namespace {

DecompositionMode ParseDecompositionMode(const std::string &s) {
  if (s == "full") return DecompositionMode::Full;
  if (s == "by_sender") return DecompositionMode::BySender;
  if (s == "by_receiver") return DecompositionMode::ByReceiver;
  if (s == "total") return DecompositionMode::Total;
  PARTHENON_FAIL("energy_transfer: unknown decomposition mode '" + s + "'");
  return DecompositionMode::Full;
}

std::vector<std::string> SplitCommaList(const std::string &s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) out.push_back(token);
  }
  return out;
}

} // namespace

ShellTransferConfig ShellTransferConfig::FromInput(parthenon::ParameterInput *pin) {
  ShellTransferConfig cfg;

  const auto binning_str = pin->GetOrAddString("energy_transfer", "binning", "lin");
  const auto num_shells = pin->GetOrAddInteger("energy_transfer", "num_shells", 20);
  if (binning_str == "lin") {
    cfg.binning = BinningSpec::Linear(num_shells);
  } else if (binning_str == "log") {
    cfg.binning = BinningSpec::Log(num_shells);
  } else {
    PARTHENON_FAIL("energy_transfer/binning must be 'lin' or 'log' (use BinningSpec::Custom "
                   "directly in code for a custom edge list)");
  }

  const auto terms_str = pin->GetOrAddString("energy_transfer", "terms", "UUA,UUC");
  for (const auto &token : SplitCommaList(terms_str)) {
    const auto colon = token.find(':');
    if (colon == std::string::npos) {
      cfg.terms.emplace_back(token, DecompositionMode::Full);
    } else {
      cfg.terms.emplace_back(token.substr(0, colon),
                             ParseDecompositionMode(token.substr(colon + 1)));
    }
  }

  const auto spectra_str = pin->GetOrAddString("energy_transfer", "spectra", "spec_U");
  cfg.spectrum_names = SplitCommaList(spectra_str);

  return cfg;
}

} // namespace energy_transfer
