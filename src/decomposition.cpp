#include "energy_transfer/decomposition.hpp"

#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

namespace energy_transfer {

DecomposedFourierField
DecomposeFourierField(parthenon::Mesh *pm,
                      const parthenon::ParArray1D<Kokkos::complex<Real>> &FT_field_3comp) {
  PARTHENON_REQUIRE_THROWS(pm != nullptr, "DecomposeFourierField: mesh pointer must not be null");

  auto FFTMgr = pm->GetFFTManager();
  const auto fft_size_outbox = FFTMgr->size_fourier_space_box();
  PARTHENON_REQUIRE_THROWS(
      FT_field_3comp.size() == 3 * fft_size_outbox,
      "DecomposeFourierField: input array must be exactly 3 * size_fourier_space_box() -- "
      "only 3-component vector fields can be directionally decomposed.");

  DecomposedFourierField out;
  out.compressive =
      parthenon::ParArray1D<Kokkos::complex<Real>>("decomp compressive", fft_size_outbox);
  out.plus = parthenon::ParArray1D<Kokkos::complex<Real>>("decomp plus", fft_size_outbox);
  out.minus = parthenon::ParArray1D<Kokkos::complex<Real>>("decomp minus", fft_size_outbox);

  auto FT_field = FT_field_3comp;
  auto compressive = out.compressive;
  auto plus = out.plus;
  auto minus = out.minus;

  auto fb = FFTMgr->fourier_space_box();
  auto kernel_helper = FFTMgr->GetKernelHelper();
  parthenon::par_for(
      "DecomposeFourierField", fb.low[2], fb.high[2], fb.low[1], fb.high[1], fb.low[0],
      fb.high[0], KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto k_vec = kernel_helper.Wavevector(k, j, i);
        const auto outidx = kernel_helper.FourierFlatIndex(k, j, i);

        const auto Vx = FT_field[outidx + 0 * fft_size_outbox];
        const auto Vy = FT_field[outidx + 1 * fft_size_outbox];
        const auto Vz = FT_field[outidx + 2 * fft_size_outbox];

        const Real kx = Real(ComponentWavenumber(k_vec, 0));
        const Real ky = Real(ComponentWavenumber(k_vec, 1));
        const Real kz = Real(ComponentWavenumber(k_vec, 2));
        const auto basis = BuildHelicalBasis(kx, ky, kz);
        const auto proj = ProjectMode(basis, Vx, Vy, Vz);

        compressive(outidx) = proj.compressive;
        plus(outidx) = proj.plus;
        minus(outidx) = proj.minus;
      });
  Kokkos::fence();

  return out;
}

} // namespace energy_transfer
