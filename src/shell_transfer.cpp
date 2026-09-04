#include "energy_transfer/shell_transfer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>

#include <globals.hpp>
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

std::set<std::string> QuantityNamesForTerms(const std::vector<std::string> &terms) {
  const auto &term_table = BuiltinTerms();
  std::set<std::string> names;
  for (auto &name : terms) {
    auto it = term_table.find(name);
    PARTHENON_REQUIRE_THROWS(it != term_table.end(),
                             "energy_transfer: unknown term '" + name + "'");
    names.insert(it->second.q_side_quantity);
    names.insert(it->second.k_side_quantity);
  }
  return names;
}

// Splits the same lookup by which side each term uses a quantity on, so the
// shared shell sweep in ComputeShellTransfer can build one small set of
// Q-side buffers and one of K-side buffers per (qi,mi)/(mi,ki) rather than a
// single undifferentiated set.
struct TermQuantityNames {
  std::set<std::string> q_side;
  std::set<std::string> k_side;
};

TermQuantityNames SplitQuantityNamesForTerms(const std::vector<std::string> &terms) {
  const auto &term_table = BuiltinTerms();
  TermQuantityNames names;
  for (auto &name : terms) {
    auto it = term_table.find(name);
    PARTHENON_REQUIRE_THROWS(it != term_table.end(),
                             "energy_transfer: unknown term '" + name + "'");
    names.q_side.insert(it->second.q_side_quantity);
    names.k_side.insert(it->second.k_side_quantity);
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
  const auto &bundle_table = BuiltinSpectrumBundles();
  for (auto &name : cfg.spectrum_names) {
    auto spec_it = spec_table.find(name);
    if (spec_it != spec_table.end()) {
      if (spec_it->second.needs_mag) req.mag = true;
      continue;
    }
    auto bundle_it = bundle_table.find(name);
    PARTHENON_REQUIRE_THROWS(bundle_it != bundle_table.end(),
                             "energy_transfer: unknown spectrum '" + name + "'");
    if (bundle_it->second.needs_mag) req.mag = true;
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

  const auto donor_edges = BuildShellEdges(cfg.donor_binning, Nx);
  const auto mediator_edges = BuildShellEdges(cfg.mediator_binning, Nx);
  const auto receiver_edges = BuildShellEdges(cfg.receiver_binning, Nx);
  const int n_donor_shells = static_cast<int>(donor_edges.size()) - 1;
  const int n_mediator_shells = static_cast<int>(mediator_edges.size()) - 1;
  const int n_receiver_shells = static_cast<int>(receiver_edges.size()) - 1;
  const Real collapsed_donor_low = donor_edges.front();
  const Real collapsed_donor_high = donor_edges.back();
  const Real collapsed_mediator_low = mediator_edges.front();
  const Real collapsed_mediator_high = mediator_edges.back();
  const Real collapsed_receiver_low = receiver_edges.front();
  const Real collapsed_receiver_high = receiver_edges.back();

  const auto qnames = QuantityNamesForTerms(cfg.terms);
  const auto tags = BaseFieldClosure(qnames);

  // cfg.mode is global -- shared by every term in cfg.terms (see
  // ShellTransferConfig::mode). If mediator decomposition is requested, it
  // applies to the whole call, so every requested term must have a
  // decomposable mediator (some terms' only mediator is a scalar
  // normalization like rho, which can't be shell-restricted) -- checked
  // once, up front, for the whole call before any FFT/allocation work
  // happens, so a bad request fails immediately rather than partway
  // through. Terms requesting mediator decomposition also need their
  // quantities' mediator_only_base_fields (e.g. FT_U for
  // U_dot_grad_W/U_dot_grad_B, otherwise unneeded).
  const auto &term_table_for_reqs = BuiltinTerms();
  const auto &quantity_table_for_reqs = BuiltinQuantities();
  std::set<std::string> mediator_tags;
  if (cfg.mode.mediator_resolved) {
    for (auto &term_name : cfg.terms) {
      const auto &term = term_table_for_reqs.at(term_name);
      const auto &q_side = quantity_table_for_reqs.at(term.q_side_quantity);
      const auto &k_side = quantity_table_for_reqs.at(term.k_side_quantity);
      PARTHENON_REQUIRE_THROWS(
          q_side.has_decomposable_mediator || k_side.has_decomposable_mediator,
          "energy_transfer: term '" + term_name +
              "' has no decomposable mediator field (its mediator, if any, is a scalar "
              "normalization like rho, which can't be shell-decomposed) -- "
              "ShellTransferConfig::mode.mediator_resolved must be false for this call, "
              "since it applies to every requested term.");
      for (auto &t : q_side.mediator_only_base_fields) mediator_tags.insert(t);
      for (auto &t : k_side.mediator_only_base_fields) mediator_tags.insert(t);
    }
  }

  const bool needs_U = tags.count("U") > 0 || mediator_tags.count("U") > 0;
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
    // Computed unconditionally alongside b_flat (not just when Divb is also
    // needed) so any term shell-decomposing the b mediator (BUT/UBTb/
    // UBTbA) can filter it without recomputing this forward transform.
    ft.FT_b = parthenon::ParArray1D<Kokkos::complex<Real>>("FT_b", 3 * fft_size_outbox);
    for (int n = 0; n < 3; n++) {
      FFTMgr->Forward(b_flat.data() + n * fft_size_inbox, ft.FT_b.data() + n * fft_size_outbox);
    }
  }
  if (needs_DivU) {
    PARTHENON_REQUIRE_THROWS(needs_U, "energy_transfer: internal error -- DivU requires FT_U");
    aux.DivU = parthenon::ParArray1D<Real>("DivU", fft_size_inbox);
    parthenon::ParArray1D<Kokkos::complex<Real>> scratch("DivU_scratch", fft_size_outbox);
    SpectralDivergence(FFTMgr, ft.FT_U, scratch, aux.DivU, two_pi_over_L);
  }
  if (needs_Divb) {
    PARTHENON_REQUIRE_THROWS(needs_b, "energy_transfer: internal error -- Divb requires b_flat");
    aux.Divb = parthenon::ParArray1D<Real>("Divb", fft_size_inbox);
    parthenon::ParArray1D<Kokkos::complex<Real>> scratch("Divb_scratch", fft_size_outbox);
    SpectralDivergence(FFTMgr, ft.FT_b, scratch, aux.Divb, two_pi_over_L);
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

  TransferResult result;
  result.n_donor_shells = n_donor_shells;
  result.n_mediator_shells = n_mediator_shells;
  result.n_receiver_shells = n_receiver_shells;
  result.donor_binning = cfg.donor_binning;
  result.mediator_binning = cfg.mediator_binning;
  result.receiver_binning = cfg.receiver_binning;
  result.donor_edges = donor_edges;
  result.mediator_edges = mediator_edges;
  result.receiver_edges = receiver_edges;

  // cfg.mode is shared by every term, so one (Q,M,K) shell sweep serves all
  // of them -- Q outer, mediator middle, K inner, terms innermost, mirroring
  // both the original single-file driver and the Python reference tool this
  // library was ported from (neither ever nests a nother loop per term, nor
  // caches a K-side quantity across different Q values). Q-side buffers
  // depend only on (qi,mi) and are recomputed fresh every mi; K-side buffers
  // depend only on (mi,ki) and are recomputed fresh every ki. Both live in
  // ordinary local std::maps freed by RAII at the end of their scope, so
  // peak memory is bounded by the number of *distinct* quantity names any
  // requested term actually needs (typically well under 10), independent of
  // shell count or term count -- this is the fix for the unbounded
  // whole-call cache that used to accumulate one array per (name, shell,
  // mediator-shell) triple ever visited and caused real OOMs on long sweeps.
  // The accepted trade-off: a K-side quantity is no longer reused across
  // different qi within one term (as the old cache did), so this recomputes
  // it up to n_q times more than before for the "many donor AND many
  // receiver shells" case -- matching both ancestors' long-proven behavior,
  // not a new regression relative to them.
  const bool q_resolved = cfg.mode.donor_resolved;
  const bool k_resolved = cfg.mode.receiver_resolved;
  const bool m_resolved = cfg.mode.mediator_resolved;
  const int n_q = q_resolved ? n_donor_shells : 1;
  const int n_k = k_resolved ? n_receiver_shells : 1;
  const int n_m = m_resolved ? n_mediator_shells : 1;
  const auto split_names = SplitQuantityNamesForTerms(cfg.terms);

  for (auto &term_name : cfg.terms) {
    result.matrices.emplace(term_name,
                            parthenon::HostArray3D<TransferReal>(term_name, n_q, n_m, n_k));
  }

  const bool report_progress = parthenon::Globals::my_rank == 0;
  if (report_progress) {
    std::cout << "energy_transfer: " << cfg.terms.size() << " term(s) -- " << n_q << " donor x "
              << n_m << " mediator x " << n_k << " receiver shells (" << (n_q * n_m * n_k)
              << " entries each)" << std::endl;
  }

  for (int qi = 0; qi < n_q; qi++) {
    const Real q_low = q_resolved ? donor_edges[qi] : collapsed_donor_low;
    const Real q_high = q_resolved ? donor_edges[qi + 1] : collapsed_donor_high;
    for (int mi = 0; mi < n_m; mi++) {
      const Real m_low = m_resolved ? mediator_edges[mi] : collapsed_mediator_low;
      const Real m_high = m_resolved ? mediator_edges[mi + 1] : collapsed_mediator_high;
      if (report_progress) {
        std::cout << "energy_transfer: donor shell " << (qi + 1) << "/" << n_q << " (k in ("
                  << q_low << ", " << q_high << "]), mediator shell " << (mi + 1) << "/" << n_m
                  << " (k in (" << m_low << ", " << m_high << "]) -- sweeping " << n_k
                  << " receiver shell(s) x " << cfg.terms.size() << " term(s)" << std::endl;
      }

      std::map<std::string, parthenon::ParArray1D<Real>> q_side_bufs;
      for (auto &qname : split_names.q_side) {
        ShellWorkspace ws = ws_template;
        ws.k_low = q_low;
        ws.k_high = q_high;
        ws.mediator_active = m_resolved;
        ws.m_low = m_low;
        ws.m_high = m_high;
        q_side_bufs.emplace(qname, quantity_table.at(qname).fn(ws));
      }

      for (int ki = 0; ki < n_k; ki++) {
        const Real k_low = k_resolved ? receiver_edges[ki] : collapsed_receiver_low;
        const Real k_high = k_resolved ? receiver_edges[ki + 1] : collapsed_receiver_high;

        std::map<std::string, parthenon::ParArray1D<Real>> k_side_bufs;
        for (auto &kname : split_names.k_side) {
          ShellWorkspace ws = ws_template;
          ws.k_low = k_low;
          ws.k_high = k_high;
          ws.mediator_active = m_resolved;
          ws.m_low = m_low;
          ws.m_high = m_high;
          k_side_bufs.emplace(kname, quantity_table.at(kname).fn(ws));
        }

        for (auto &term_name : cfg.terms) {
          const auto &term = term_table.at(term_name);
          result.matrices.at(term_name)(qi, mi, ki) =
              term.prefactor * DotProductReduce(k_side_bufs.at(term.k_side_quantity),
                                                q_side_bufs.at(term.q_side_quantity),
                                                3 * fft_size_inbox);
        }
      } // ki -- k_side_bufs freed here
    }   // mi -- q_side_bufs freed here
  }
  if (report_progress) {
    std::cout << "energy_transfer: done." << std::endl;
  }

  const auto &spectrum_table = BuiltinSpectra();
  const auto &bundle_table = BuiltinSpectrumBundles();
  for (auto &name : cfg.spectrum_names) {
    auto spec_it = spectrum_table.find(name);
    if (spec_it != spectrum_table.end()) {
      result.spectra.emplace(name, spec_it->second.fn(pmesh, fields, W_flat));
      continue;
    }
    for (auto &[sub_name, arr] : bundle_table.at(name).fn(pmesh, fields, W_flat)) {
      result.spectra.emplace(sub_name, arr);
    }
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
                                            const FileFieldNaming &naming,
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

  FlatFields fields;
  switch (DetectInputFileFormat(input_file)) {
  case InputFileFormat::ADIOS2:
    fields = ReadADIOS2Fields(pmesh, input_file, naming);
    break;
  case InputFileFormat::ParthenonHDF5:
    fields = ReadPHDFFields(pmesh, input_file, naming);
    break;
  }
  ConvertConservedToPrimitive(fields);
  return ComputeShellTransfer(pmesh, fields, cfg);
}

namespace {

// Named tokens covering all 2^3 donor/mediator/receiver resolved-vs-
// collapsed combinations, for the single energy_transfer/mode= input key
// (shared by every term in energy_transfer/terms=). The four non-"_mediator"
// names are the legacy tokens (mediator always collapsed, matching behavior
// before mediator decomposition existed); the "_mediator"/"mediator_only"
// names add the mediator axis.
DecompositionMode ParseDecompositionMode(const std::string &s) {
  if (s == "full") return DecompositionMode::Full();
  if (s == "by_sender") return DecompositionMode::BySender();
  if (s == "by_receiver") return DecompositionMode::ByReceiver();
  if (s == "total") return DecompositionMode::Total();
  if (s == "full_mediator") return DecompositionMode::FullWithMediator();
  if (s == "by_sender_mediator") return DecompositionMode::BySenderWithMediator();
  if (s == "by_receiver_mediator") return DecompositionMode::ByReceiverWithMediator();
  if (s == "mediator_only") return DecompositionMode::MediatorOnly();
  PARTHENON_FAIL("energy_transfer: unknown decomposition mode '" + s + "'");
  return DecompositionMode::Full();
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

BinningSpec ParseBinningSpecFromKeys(parthenon::ParameterInput *pin,
                                     const std::string &binning_key,
                                     const std::string &num_shells_key,
                                     const std::string &shell_edges_key) {
  const auto binning_str = pin->GetOrAddString("energy_transfer", binning_key, "lin");
  const auto num_shells = pin->GetOrAddInteger("energy_transfer", num_shells_key, 20);
  if (binning_str == "lin") return BinningSpec::Linear(num_shells);
  if (binning_str == "log") return BinningSpec::Log(num_shells);
  if (binning_str == "custom") {
    const auto edges_str = pin->GetOrAddString("energy_transfer", shell_edges_key, "");
    std::vector<Real> edges;
    for (const auto &token : SplitCommaList(edges_str)) {
      edges.push_back(static_cast<Real>(std::stod(token)));
    }
    PARTHENON_REQUIRE_THROWS(edges.size() >= 2,
                             "energy_transfer/" + binning_key +
                                 "=custom requires energy_transfer/" + shell_edges_key +
                                 " to list at least 2 comma-separated bin-edge values, e.g. " +
                                 shell_edges_key + " = 0.5,1.5,2.5,16.0,26.5,28.5,32.0");
    return BinningSpec::Custom(std::move(edges));
  }
  PARTHENON_FAIL(("energy_transfer/" + binning_key + " must be 'lin', 'log', or 'custom'").c_str());
  return BinningSpec::Linear(num_shells);
}

} // namespace

ShellTransferConfig ShellTransferConfig::FromInput(parthenon::ParameterInput *pin) {
  ShellTransferConfig cfg;

  // Shared default: binning=/num_shells=/shell_edges= (no prefix), used by
  // any axis without its own explicit donor_/mediator_/receiver_ override --
  // an input deck that only sets these keeps all three axes sharing one
  // binning, exactly as before mediator decomposition existed.
  const auto default_binning = ParseBinningSpecFromKeys(pin, "binning", "num_shells", "shell_edges");

  cfg.donor_binning = pin->DoesParameterExist("energy_transfer", "donor_binning")
                          ? ParseBinningSpecFromKeys(pin, "donor_binning", "donor_num_shells",
                                                     "donor_shell_edges")
                          : default_binning;
  cfg.mediator_binning =
      pin->DoesParameterExist("energy_transfer", "mediator_binning")
          ? ParseBinningSpecFromKeys(pin, "mediator_binning", "mediator_num_shells",
                                     "mediator_shell_edges")
          : default_binning;
  cfg.receiver_binning =
      pin->DoesParameterExist("energy_transfer", "receiver_binning")
          ? ParseBinningSpecFromKeys(pin, "receiver_binning", "receiver_num_shells",
                                     "receiver_shell_edges")
          : default_binning;

  const auto terms_str = pin->GetOrAddString("energy_transfer", "terms", "UUA,UUC");
  cfg.terms = SplitCommaList(terms_str);

  const auto mode_str = pin->GetOrAddString("energy_transfer", "mode", "full");
  cfg.mode = ParseDecompositionMode(mode_str);

  const auto spectra_str = pin->GetOrAddString("energy_transfer", "spectra", "spec_U");
  cfg.spectrum_names = SplitCommaList(spectra_str);

  return cfg;
}

} // namespace energy_transfer
