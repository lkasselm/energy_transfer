#include "energy_transfer/lorentz_force.hpp"

#include <cmath>

#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

#include "energy_transfer/spectral_kernels.hpp"

namespace energy_transfer {

parthenon::ParArray1D<Real> CalcLorentzForce(parthenon::Mesh *pm,
                                             const parthenon::ParArray1D<Real> &B) {
  PARTHENON_REQUIRE_THROWS(pm != nullptr, "CalcLorentzForce: mesh pointer must not be null");

  auto FFTMgr = pm->GetFFTManager();
  const auto fft_size_inbox = FFTMgr->size_real_space_box();
  const auto fft_size_outbox = FFTMgr->size_fourier_space_box();
  PARTHENON_REQUIRE_THROWS(B.size() == 3 * fft_size_inbox,
                           "CalcLorentzForce: B must be 3 * size_real_space_box().");

  // Mode indices from Wavevector()/ComponentWavenumber() are raw integers,
  // not physical wavenumbers -- scale by two_pi_over_L here the same way
  // SpectralDivergence/ShellFilterDerivative do (spectral_kernels.cpp),
  // rather than leaving it out the way helicity.cpp's vector potential
  // currently does (fine there only because that file's one caller/test
  // happens to always use an L=2*pi box).
  const auto &mesh_size = pm->mesh_size;
  const Real Lx = mesh_size.xmax(parthenon::X1DIR) - mesh_size.xmin(parthenon::X1DIR);
  const Real two_pi_over_L = 2.0 * M_PI / Lx;

  parthenon::ParArray1D<Kokkos::complex<Real>> FT_B("FT_B_lorentz", 3 * fft_size_outbox);
  for (int n = 0; n < 3; n++) {
    FFTMgr->Forward(B.data() + n * fft_size_inbox, FT_B.data() + n * fft_size_outbox);
  }

  parthenon::ParArray1D<Kokkos::complex<Real>> FT_J("FT_J", 3 * fft_size_outbox);
  auto FT_J_view = FT_J;
  auto FT_B_view = FT_B;
  const Kokkos::complex<Real> imag_unit(0.0, 1.0);

  auto fb = FFTMgr->fourier_space_box();
  auto kernel_helper = FFTMgr->GetKernelHelper();
  parthenon::par_for(
      "ComputeCurrentDensityFourier", fb.low[2], fb.high[2], fb.low[1], fb.high[1], fb.low[0],
      fb.high[0], KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto k_vec = kernel_helper.Wavevector(k, j, i);
        const Real kx = Real(ComponentWavenumber(k_vec, 0)) * two_pi_over_L;
        const Real ky = Real(ComponentWavenumber(k_vec, 1)) * two_pi_over_L;
        const Real kz = Real(ComponentWavenumber(k_vec, 2)) * two_pi_over_L;
        const auto idx = kernel_helper.FourierFlatIndex(k, j, i);

        const auto Bx = FT_B_view[idx + 0 * fft_size_outbox];
        const auto By = FT_B_view[idx + 1 * fft_size_outbox];
        const auto Bz = FT_B_view[idx + 2 * fft_size_outbox];
        // J_hat = i k x B_hat -- the curl of B in Fourier space. No DC
        // special-case needed: at k=0 every term above is zero already,
        // unlike a quantity that divides by |k|^2.
        FT_J_view[idx + 0 * fft_size_outbox] = imag_unit * (ky * Bz - kz * By);
        FT_J_view[idx + 1 * fft_size_outbox] = imag_unit * (kz * Bx - kx * Bz);
        FT_J_view[idx + 2 * fft_size_outbox] = imag_unit * (kx * By - ky * Bx);
      });
  Kokkos::fence();

  parthenon::ParArray1D<Real> J("J_current", 3 * fft_size_inbox);
  for (int n = 0; n < 3; n++) {
    FFTMgr->Backward(FT_J.data() + n * fft_size_outbox, J.data() + n * fft_size_inbox);
  }

  parthenon::ParArray1D<Real> F("lorentz_force", 3 * fft_size_inbox);
  parthenon::par_for(
      "ComputeLorentzForceDensity", std::size_t(0), fft_size_inbox - 1,
      KOKKOS_LAMBDA(const std::size_t idx) {
        const Real Jx = J(0 * fft_size_inbox + idx);
        const Real Jy = J(1 * fft_size_inbox + idx);
        const Real Jz = J(2 * fft_size_inbox + idx);
        const Real Bx = B(0 * fft_size_inbox + idx);
        const Real By = B(1 * fft_size_inbox + idx);
        const Real Bz = B(2 * fft_size_inbox + idx);
        F(0 * fft_size_inbox + idx) = Jy * Bz - Jz * By;
        F(1 * fft_size_inbox + idx) = Jz * Bx - Jx * Bz;
        F(2 * fft_size_inbox + idx) = Jx * By - Jy * Bx;
      });
  Kokkos::fence();

  return F;
}

} // namespace energy_transfer
