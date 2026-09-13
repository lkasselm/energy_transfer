#include "energy_transfer/helicity.hpp"

#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

#include "energy_transfer/spectral_kernels.hpp"

namespace energy_transfer {

VectorPotentialFourier ComputeVectorPotentialFourier(parthenon::Mesh *pm,
                                                     const parthenon::ParArray1D<Real> &B) {
  PARTHENON_REQUIRE_THROWS(pm != nullptr,
                           "ComputeVectorPotentialFourier: mesh pointer must not be null");

  auto FFTMgr = pm->GetFFTManager();
  const auto fft_size_inbox = FFTMgr->size_real_space_box();
  const auto fft_size_outbox = FFTMgr->size_fourier_space_box();
  PARTHENON_REQUIRE_THROWS(B.size() == 3 * fft_size_inbox,
                           "ComputeVectorPotentialFourier: B must be 3 * size_real_space_box().");

  VectorPotentialFourier vp;
  vp.FT_A = parthenon::ParArray1D<Kokkos::complex<Real>>("FT_A", 3 * fft_size_outbox);
  vp.FT_B = parthenon::ParArray1D<Kokkos::complex<Real>>("FT_B", 3 * fft_size_outbox);
  for (int n = 0; n < 3; n++) {
    FFTMgr->Forward(B.data() + n * fft_size_inbox, vp.FT_B.data() + n * fft_size_outbox);
  }

  auto FT_A = vp.FT_A;
  auto FT_B = vp.FT_B;
  const Kokkos::complex<Real> imag_unit(0.0, 1.0);

  auto fb = FFTMgr->fourier_space_box();
  auto kernel_helper = FFTMgr->GetKernelHelper();
  parthenon::par_for(
      "ComputeVectorPotentialFourier", fb.low[2], fb.high[2], fb.low[1], fb.high[1], fb.low[0],
      fb.high[0], KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto k_vec = kernel_helper.Wavevector(k, j, i);
        const Real kx = Real(ComponentWavenumber(k_vec, 0));
        const Real ky = Real(ComponentWavenumber(k_vec, 1));
        const Real kz = Real(ComponentWavenumber(k_vec, 2));
        const Real k2 = kx * kx + ky * ky + kz * kz;
        const auto idx = kernel_helper.FourierFlatIndex(k, j, i);

        if (k2 == Real(0.0)) {
          // DC mode: direction is undefined -- vector potential undefined too.
          FT_A[idx + 0 * fft_size_outbox] = 0.0;
          FT_A[idx + 1 * fft_size_outbox] = 0.0;
          FT_A[idx + 2 * fft_size_outbox] = 0.0;
          return;
        }

        const auto Bx = FT_B[idx + 0 * fft_size_outbox];
        const auto By = FT_B[idx + 1 * fft_size_outbox];
        const auto Bz = FT_B[idx + 2 * fft_size_outbox];
        // A_hat = i (k x B_hat) / |k|^2
        FT_A[idx + 0 * fft_size_outbox] = imag_unit * (ky * Bz - kz * By) / k2;
        FT_A[idx + 1 * fft_size_outbox] = imag_unit * (kz * Bx - kx * Bz) / k2;
        FT_A[idx + 2 * fft_size_outbox] = imag_unit * (kx * By - ky * Bx) / k2;
      });
  Kokkos::fence();

  return vp;
}

parthenon::ParArray1D<Real> CalcHelicity(parthenon::Mesh *pm, const parthenon::ParArray1D<Real> &B) {
  auto vp = ComputeVectorPotentialFourier(pm, B);

  auto FFTMgr = pm->GetFFTManager();
  const auto fft_size_inbox = FFTMgr->size_real_space_box();
  const auto fft_size_outbox = FFTMgr->size_fourier_space_box();

  parthenon::ParArray1D<Real> A("A_helicity", 3 * fft_size_inbox);
  for (int n = 0; n < 3; n++) {
    FFTMgr->Backward(vp.FT_A.data() + n * fft_size_outbox, A.data() + n * fft_size_inbox);
  }

  parthenon::ParArray1D<Real> H("H", fft_size_inbox);
  parthenon::par_for(
      "ComputeHelicityDensity", std::size_t(0), fft_size_inbox - 1,
      KOKKOS_LAMBDA(const std::size_t idx) {
        Real h = 0.0;
        for (int n = 0; n < 3; n++) h += A(n * fft_size_inbox + idx) * B(n * fft_size_inbox + idx);
        H(idx) = h;
      });
  Kokkos::fence();

  return H;
}

} // namespace energy_transfer
