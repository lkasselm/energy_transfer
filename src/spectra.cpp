#include "energy_transfer/spectra.hpp"

#include <sstream>

#include <kokkos_abstraction.hpp>
#include <utils/calc_spectrum.hpp>
#include <utils/error_checking.hpp>

#include "energy_transfer/decomposition.hpp"
#include "energy_transfer/spectral_kernels.hpp"

namespace energy_transfer {

namespace {

// ---- Plain power spectra ------------------------------------------------

parthenon::HostArray2D<TransferReal> SpecU(parthenon::Mesh *pm, const FlatFields &f,
                                           const parthenon::ParArray1D<Real> & /*W_flat*/) {
  auto spectra = parthenon::utils::fft::CalcSpectrum(pm, f.mom_or_vel, 3);
  return spectra.GetHostMirrorAndCopy();
}

parthenon::HostArray2D<TransferReal> SpecRho(parthenon::Mesh *pm, const FlatFields &f,
                                             const parthenon::ParArray1D<Real> & /*W_flat*/) {
  auto spectra = parthenon::utils::fft::CalcSpectrum(pm, f.rho, 1);
  return spectra.GetHostMirrorAndCopy();
}

parthenon::HostArray2D<TransferReal> SpecW(parthenon::Mesh *pm, const FlatFields & /*f*/,
                                           const parthenon::ParArray1D<Real> &W_flat) {
  auto spectra = parthenon::utils::fft::CalcSpectrum(pm, W_flat, 3);
  return spectra.GetHostMirrorAndCopy();
}

parthenon::HostArray2D<TransferReal> SpecB(parthenon::Mesh *pm, const FlatFields &f,
                                           const parthenon::ParArray1D<Real> & /*W_flat*/) {
  auto spectra = parthenon::utils::fft::CalcSpectrum(pm, f.mag, 3);
  return spectra.GetHostMirrorAndCopy();
}

// ---- Decomposed spectra (compressive / plus / minus, fused with full) --
// One bundle per vector field (U, W, B); rho is scalar and has no direction
// to decompose against. Each shares a single Forward() FFT across all four
// outputs (see decomposition.hpp's DecomposeFourierField) -- there's rarely
// a reason to request just one direction, and doing so separately would
// redundantly re-FFT the same field three times over.

parthenon::ParArray1D<Kokkos::complex<Real>>
ForwardTransformVector(parthenon::Mesh *pm, const parthenon::ParArray1D<Real> &field) {
  auto FFTMgr = pm->GetFFTManager();
  const auto fft_size_inbox = FFTMgr->size_real_space_box();
  const auto fft_size_outbox = FFTMgr->size_fourier_space_box();
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_field("FT_field", 3 * fft_size_outbox);
  for (int n = 0; n < 3; n++) {
    FFTMgr->Forward(field.data() + n * fft_size_inbox, FT_field.data() + n * fft_size_outbox);
  }
  return FT_field;
}

std::map<std::string, parthenon::HostArray2D<TransferReal>>
ComputeDecomposedSpectrumBundle(parthenon::Mesh *pm, const std::string &base_name,
                                const parthenon::ParArray1D<Real> &field) {
  auto FT_field = ForwardTransformVector(pm, field);
  auto decomp = DecomposeFourierField(pm, FT_field);
  return {
      {base_name, BinFourierSpectrum(pm, FT_field, 3).GetHostMirrorAndCopy()},
      {base_name + "_compressive", BinFourierSpectrum(pm, decomp.compressive, 1).GetHostMirrorAndCopy()},
      {base_name + "_plus", BinFourierSpectrum(pm, decomp.plus, 1).GetHostMirrorAndCopy()},
      {base_name + "_minus", BinFourierSpectrum(pm, decomp.minus, 1).GetHostMirrorAndCopy()},
  };
}

std::map<std::string, parthenon::HostArray2D<TransferReal>>
SpecUDecomp(parthenon::Mesh *pm, const FlatFields &f, const parthenon::ParArray1D<Real> &) {
  return ComputeDecomposedSpectrumBundle(pm, "spec_U", f.mom_or_vel);
}

std::map<std::string, parthenon::HostArray2D<TransferReal>>
SpecWDecomp(parthenon::Mesh *pm, const FlatFields &, const parthenon::ParArray1D<Real> &W_flat) {
  return ComputeDecomposedSpectrumBundle(pm, "spec_W", W_flat);
}

std::map<std::string, parthenon::HostArray2D<TransferReal>>
SpecBDecomp(parthenon::Mesh *pm, const FlatFields &f, const parthenon::ParArray1D<Real> &) {
  return ComputeDecomposedSpectrumBundle(pm, "spec_B", f.mag);
}

// ---- Registry tables (private -- ComputeSpectra/SpectrumNeedsMag are the
// only public surface; there's no need for callers to see these) ---------

using SpectrumFn = parthenon::HostArray2D<TransferReal> (*)(
    parthenon::Mesh *, const FlatFields &, const parthenon::ParArray1D<Real> &W_flat);
struct Spectrum {
  SpectrumFn fn;
  bool needs_mag = false;
};

using SpectrumBundleFn = std::map<std::string, parthenon::HostArray2D<TransferReal>> (*)(
    parthenon::Mesh *, const FlatFields &, const parthenon::ParArray1D<Real> &W_flat);
struct SpectrumBundle {
  SpectrumBundleFn fn;
  bool needs_mag = false;
};

const std::map<std::string, Spectrum> &BuiltinSpectra() {
  static const std::map<std::string, Spectrum> table = {
      {"spec_U", {&SpecU, false}},
      {"spec_rho", {&SpecRho, false}},
      {"spec_W", {&SpecW, false}},
      {"spec_B", {&SpecB, true}},
  };
  return table;
}

const std::map<std::string, SpectrumBundle> &BuiltinSpectrumBundles() {
  static const std::map<std::string, SpectrumBundle> table = {
      {"spec_U_decomp", {&SpecUDecomp, false}},
      {"spec_W_decomp", {&SpecWDecomp, false}},
      {"spec_B_decomp", {&SpecBDecomp, true}},
  };
  return table;
}

} // namespace

bool SpectrumNeedsMag(const std::string &name) {
  const auto &spectrum_table = BuiltinSpectra();
  auto spec_it = spectrum_table.find(name);
  if (spec_it != spectrum_table.end()) return spec_it->second.needs_mag;

  const auto &bundle_table = BuiltinSpectrumBundles();
  auto bundle_it = bundle_table.find(name);
  PARTHENON_REQUIRE_THROWS(bundle_it != bundle_table.end(),
                           "energy_transfer: unknown spectrum '" + name + "'");
  return bundle_it->second.needs_mag;
}

std::vector<std::string> ParseSpectrumNames(parthenon::ParameterInput *pin) {
  const auto spectra_str = pin->GetOrAddString("energy_transfer", "spectra", "spec_U");
  std::vector<std::string> names;
  std::stringstream ss(spectra_str);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) names.push_back(token);
  }
  return names;
}

std::map<std::string, parthenon::HostArray2D<TransferReal>>
ComputeSpectra(parthenon::Mesh *pm, const FlatFields &fields,
              const std::vector<std::string> &spectrum_names) {
  // W = sqrt(rho) * velocity -- computed unconditionally (cheap, no FFT) so
  // spec_W/spec_W_decomp work without any extra plumbing from the caller.
  const auto fft_size_inbox = pm->GetFFTManager()->size_real_space_box();
  auto rho = fields.rho;
  auto vel = fields.mom_or_vel;
  parthenon::ParArray1D<Real> W_flat("W_flat", 3 * fft_size_inbox);
  parthenon::par_for(
      "ComputeW_forSpectra", std::size_t(0), fft_size_inbox - 1,
      KOKKOS_LAMBDA(const std::size_t idx) {
        const Real sqrt_rho = Kokkos::sqrt(rho(idx));
        for (int n = 0; n < 3; n++) {
          W_flat(n * fft_size_inbox + idx) = sqrt_rho * vel(n * fft_size_inbox + idx);
        }
      });
  Kokkos::fence();

  const auto &spectrum_table = BuiltinSpectra();
  const auto &bundle_table = BuiltinSpectrumBundles();

  std::map<std::string, parthenon::HostArray2D<TransferReal>> result;
  for (auto &name : spectrum_names) {
    auto spec_it = spectrum_table.find(name);
    if (spec_it != spectrum_table.end()) {
      result.emplace(name, spec_it->second.fn(pm, fields, W_flat));
      continue;
    }
    auto bundle_it = bundle_table.find(name);
    PARTHENON_REQUIRE_THROWS(bundle_it != bundle_table.end(),
                             "energy_transfer: unknown spectrum '" + name + "'");
    for (auto &[sub_name, arr] : bundle_it->second.fn(pm, fields, W_flat)) {
      result.emplace(sub_name, arr);
    }
  }
  return result;
}

} // namespace energy_transfer
