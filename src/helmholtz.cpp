#include "energy_transfer/helmholtz.hpp"

#include <cmath>

#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

#include "energy_transfer/spectral_kernels.hpp"

namespace energy_transfer {

HelmholtzFourier ComputeHelmholtzFourier(parthenon::Mesh *pm,
                                         const parthenon::ParArray1D<Real> &F) {
  PARTHENON_REQUIRE_THROWS(pm != nullptr, "ComputeHelmholtzFourier: mesh pointer must not be null");

  auto FFTMgr = pm->GetFFTManager();
  const auto fft_size_inbox = FFTMgr->size_real_space_box();
  const auto fft_size_outbox = FFTMgr->size_fourier_space_box();
  PARTHENON_REQUIRE_THROWS(F.size() == 3 * fft_size_inbox,
                           "ComputeHelmholtzFourier: F must be 3 * size_real_space_box().");

  // Mode indices from Wavevector()/ComponentWavenumber() are raw integers,
  // not physical wavenumbers -- scale by two_pi_over_L the same way
  // SpectralDivergence/ShellFilterDerivative (spectral_kernels.cpp) do.
  const auto &mesh_size = pm->mesh_size;
  const Real Lx = mesh_size.xmax(parthenon::X1DIR) - mesh_size.xmin(parthenon::X1DIR);
  const Real two_pi_over_L = 2.0 * M_PI / Lx;

  HelmholtzFourier h;
  h.FT_F = parthenon::ParArray1D<Kokkos::complex<Real>>("FT_F_helmholtz", 3 * fft_size_outbox);
  h.FT_phi = parthenon::ParArray1D<Kokkos::complex<Real>>("FT_phi", fft_size_outbox);
  h.FT_A = parthenon::ParArray1D<Kokkos::complex<Real>>("FT_A", 3 * fft_size_outbox);
  for (int n = 0; n < 3; n++) {
    FFTMgr->Forward(F.data() + n * fft_size_inbox, h.FT_F.data() + n * fft_size_outbox);
  }

  auto FT_F = h.FT_F;
  auto FT_phi = h.FT_phi;
  auto FT_A = h.FT_A;
  const Kokkos::complex<Real> imag_unit(0.0, 1.0);

  auto fb = FFTMgr->fourier_space_box();
  auto kernel_helper = FFTMgr->GetKernelHelper();
  parthenon::par_for(
      "ComputeHelmholtzFourier", fb.low[2], fb.high[2], fb.low[1], fb.high[1], fb.low[0],
      fb.high[0], KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto k_vec = kernel_helper.Wavevector(k, j, i);
        const Real kx = Real(ComponentWavenumber(k_vec, 0)) * two_pi_over_L;
        const Real ky = Real(ComponentWavenumber(k_vec, 1)) * two_pi_over_L;
        const Real kz = Real(ComponentWavenumber(k_vec, 2)) * two_pi_over_L;
        const Real k2 = kx * kx + ky * ky + kz * kz;
        const auto idx = kernel_helper.FourierFlatIndex(k, j, i);

        if (k2 == Real(0.0)) {
          // DC mode: the mean of F belongs to neither potential.
          FT_phi[idx] = 0.0;
          FT_A[idx + 0 * fft_size_outbox] = 0.0;
          FT_A[idx + 1 * fft_size_outbox] = 0.0;
          FT_A[idx + 2 * fft_size_outbox] = 0.0;
          return;
        }

        const auto Fx = FT_F[idx + 0 * fft_size_outbox];
        const auto Fy = FT_F[idx + 1 * fft_size_outbox];
        const auto Fz = FT_F[idx + 2 * fft_size_outbox];
        // phi_hat = -i (k . F_hat) / |k|^2
        FT_phi[idx] = -imag_unit * (kx * Fx + ky * Fy + kz * Fz) / k2;
        // A_hat = i (k x F_hat) / |k|^2
        FT_A[idx + 0 * fft_size_outbox] = imag_unit * (ky * Fz - kz * Fy) / k2;
        FT_A[idx + 1 * fft_size_outbox] = imag_unit * (kz * Fx - kx * Fz) / k2;
        FT_A[idx + 2 * fft_size_outbox] = imag_unit * (kx * Fy - ky * Fx) / k2;
      });
  Kokkos::fence();

  return h;
}

HelmholtzPotentials CalcHelmholtzPotentials(parthenon::Mesh *pm,
                                            const parthenon::ParArray1D<Real> &F) {
  auto h = ComputeHelmholtzFourier(pm, F);

  auto FFTMgr = pm->GetFFTManager();
  const auto fft_size_inbox = FFTMgr->size_real_space_box();
  const auto fft_size_outbox = FFTMgr->size_fourier_space_box();

  HelmholtzPotentials p;
  p.phi = parthenon::ParArray1D<Real>("phi_helmholtz", fft_size_inbox);
  p.A = parthenon::ParArray1D<Real>("A_helmholtz", 3 * fft_size_inbox);
  FFTMgr->Backward(h.FT_phi.data(), p.phi.data());
  for (int n = 0; n < 3; n++) {
    FFTMgr->Backward(h.FT_A.data() + n * fft_size_outbox, p.A.data() + n * fft_size_inbox);
  }
  Kokkos::fence();
  return p;
}

} // namespace energy_transfer
