#include "energy_transfer/spectra.hpp"

#include <sstream>

#include <kokkos_abstraction.hpp>
#include <utils/calc_spectrum.hpp>
#include <utils/error_checking.hpp>

#include "energy_transfer/decomposition.hpp"
#include "energy_transfer/helicity.hpp"
#include "energy_transfer/spectral_kernels.hpp"

namespace energy_transfer {

namespace {

// "spec_helicity"/"spec_helicity_variance" don't fit the field-selector
// table below (that pattern is "pick one field, call CalcSpectrum"; these
// need the vector-potential machinery in helicity.hpp and, for
// spec_helicity, a co-spectrum of two different Fourier fields) -- they're
// handled as two explicit names instead, checked before the table lookup.
bool IsHelicitySpectrumName(const std::string &name) {
  return name == "spec_helicity" || name == "spec_helicity_variance";
}

// ---- Field selectors -----------------------------------------------------
// The only code that knows a spectrum name refers to a particular field --
// everything downstream (CalcSpectrum, ComputeDecomposedSpectrumBundle) is
// completely field-agnostic, only caring whether it's a 1- or 3-component
// real-space array.

using FieldSelectorFn = const parthenon::ParArray1D<Real> &(*)(
    const FlatFields &, const parthenon::ParArray1D<Real> &W_flat);

const parthenon::ParArray1D<Real> &SelectU(const FlatFields &f,
                                           const parthenon::ParArray1D<Real> &) {
  return f.mom_or_vel;
}
const parthenon::ParArray1D<Real> &SelectRho(const FlatFields &f,
                                             const parthenon::ParArray1D<Real> &) {
  return f.rho;
}
const parthenon::ParArray1D<Real> &SelectW(const FlatFields &,
                                           const parthenon::ParArray1D<Real> &W_flat) {
  return W_flat;
}
const parthenon::ParArray1D<Real> &SelectB(const FlatFields &f,
                                           const parthenon::ParArray1D<Real> &) {
  return f.mag;
}

// One row per plain spectrum name -- "<name>_decomp" isn't a separate row;
// it's parsed (see HasDecompSuffix below) and only valid when decomposable.
struct Spectrum {
  FieldSelectorFn select;
  int n_comp;                 // 1 (scalar) or 3 (vector)
  bool needs_mag = false;
  bool decomposable = false;  // "<name>_decomp" valid iff true (only for n_comp==3)
};

const std::map<std::string, Spectrum> &BuiltinSpectra() {
  static const std::map<std::string, Spectrum> table = {
      {"spec_U", {&SelectU, 3, false, true}},
      {"spec_rho", {&SelectRho, 1, false, false}},
      {"spec_W", {&SelectW, 3, false, true}},
      {"spec_B", {&SelectB, 3, true, true}},
  };
  return table;
}

bool HasDecompSuffix(const std::string &name) {
  static const std::string suffix = "_decomp";
  return name.size() > suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string StripDecompSuffix(const std::string &name) {
  static const std::string suffix = "_decomp";
  return name.substr(0, name.size() - suffix.size());
}

const Spectrum &LookupSpectrum(const std::string &name) {
  const auto &table = BuiltinSpectra();
  const auto base = HasDecompSuffix(name) ? StripDecompSuffix(name) : name;
  auto it = table.find(base);
  PARTHENON_REQUIRE_THROWS(it != table.end(), "energy_transfer: unknown spectrum '" + name + "'");
  if (HasDecompSuffix(name)) {
    PARTHENON_REQUIRE_THROWS(it->second.decomposable, "energy_transfer: '" + name + "' -- '" +
                                                           base +
                                                           "' has no direction to decompose");
  }
  return it->second;
}

// ---- Decomposed spectra (compressive / plus / minus, fused with full) --
// Shares a single Forward() FFT across all four outputs (see
// decomposition.hpp's DecomposeFourierField) -- there's rarely a reason to
// want just one direction, and computing them separately would redundantly
// re-FFT the same field three times over. Field-agnostic: only needs a
// 3-component real-space array and the base name to build its 4 output keys.

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

} // namespace

bool SpectrumNeedsMag(const std::string &name) {
  if (IsHelicitySpectrumName(name)) return true;
  return LookupSpectrum(name).needs_mag;
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

  std::map<std::string, parthenon::HostArray2D<TransferReal>> result;
  for (auto &name : spectrum_names) {
    if (name == "spec_helicity") {
      PARTHENON_REQUIRE_THROWS(fields.mag.size() > 0,
                               "energy_transfer: spec_helicity needs the magnetic field.");
      auto vp = ComputeVectorPotentialFourier(pm, fields.mag);
      result.emplace(name,
                     BinFourierCospectrum(pm, vp.FT_A, vp.FT_B, 3).GetHostMirrorAndCopy());
      continue;
    }
    if (name == "spec_helicity_variance") {
      PARTHENON_REQUIRE_THROWS(
          fields.mag.size() > 0,
          "energy_transfer: spec_helicity_variance needs the magnetic field.");
      auto H = CalcHelicity(pm, fields.mag);
      result.emplace(name, parthenon::utils::fft::CalcSpectrum(pm, H, 1).GetHostMirrorAndCopy());
      continue;
    }

    const auto &spec = LookupSpectrum(name);
    const auto &field = spec.select(fields, W_flat);
    if (!HasDecompSuffix(name)) {
      result.emplace(name, parthenon::utils::fft::CalcSpectrum(pm, field, spec.n_comp)
                               .GetHostMirrorAndCopy());
    } else {
      for (auto &[sub_name, arr] :
          ComputeDecomposedSpectrumBundle(pm, StripDecompSuffix(name), field)) {
        result.emplace(sub_name, arr);
      }
    }
  }
  return result;
}

} // namespace energy_transfer
