#include "energy_transfer/spectral_kernels.hpp"

#include <utils/uniform_grid_helper.hpp>

namespace energy_transfer {

void ShellFilter(parthenon::FFTManager *fft_mgr, int n_comp,
                 const parthenon::ParArray1D<Kokkos::complex<Real>> &FT_field,
                 parthenon::ParArray1D<Kokkos::complex<Real>> &FT_scratch,
                 parthenon::ParArray1D<Real> &real_out, Real k_low, Real k_high) {
  auto fb = fft_mgr->fourier_space_box();
  auto kernel_helper = fft_mgr->GetKernelHelper();
  const auto fft_size_outbox = fft_mgr->size_fourier_space_box();
  const auto fft_size_inbox = fft_mgr->size_real_space_box();

  auto FT_in = FT_field.data();
  auto FT_out = FT_scratch.data();
  const int nc = n_comp;

  parthenon::par_for(
      "ShellFilter", fb.low[2], fb.high[2], fb.low[1], fb.high[1], fb.low[0], fb.high[0],
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto k_vec = kernel_helper.Wavevector(k, j, i);
        auto k_mag = Kokkos::sqrt(
            Real(k_vec[0] * k_vec[0] + k_vec[1] * k_vec[1] + k_vec[2] * k_vec[2]));
        auto idx = kernel_helper.FourierFlatIndex(k, j, i);
        bool in_shell = (k_mag > k_low) && (k_mag <= k_high);
        for (int n = 0; n < nc; n++) {
          FT_out[idx + n * fft_size_outbox] = in_shell ? FT_in[idx + n * fft_size_outbox]
                                                       : Kokkos::complex<Real>(0.0, 0.0);
        }
      });

  for (int n = 0; n < n_comp; n++) {
    fft_mgr->Backward(FT_scratch.data() + n * fft_size_outbox,
                      real_out.data() + n * fft_size_inbox);
  }
  Kokkos::fence();
}

void ShellFilterDerivative(parthenon::FFTManager *fft_mgr,
                           const parthenon::ParArray1D<Kokkos::complex<Real>> &FT_field_full,
                           int comp_offset,
                           parthenon::ParArray1D<Kokkos::complex<Real>> &FT_scratch,
                           int scratch_offset, parthenon::ParArray1D<Real> &deriv_out,
                           int out_offset, Real k_low, Real k_high, int dir,
                           Real two_pi_over_L) {
  auto fb = fft_mgr->fourier_space_box();
  auto kernel_helper = fft_mgr->GetKernelHelper();

  auto FT_in = FT_field_full.data();
  auto FT_out = FT_scratch.data();
  const Kokkos::complex<Real> imag_unit(0.0, 1.0);
  const int d = dir;
  const Real scale = two_pi_over_L;
  const std::size_t in_off = comp_offset;
  const std::size_t out_off = scratch_offset;

  parthenon::par_for(
      "ShellFilterDeriv", fb.low[2], fb.high[2], fb.low[1], fb.high[1], fb.low[0],
      fb.high[0], KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto k_vec = kernel_helper.Wavevector(k, j, i);
        auto k_mag = Kokkos::sqrt(
            Real(k_vec[0] * k_vec[0] + k_vec[1] * k_vec[1] + k_vec[2] * k_vec[2]));
        auto idx = kernel_helper.FourierFlatIndex(k, j, i);
        bool in_shell = (k_mag > k_low) && (k_mag <= k_high);
        Real k_phys = scale * ComponentWavenumber(k_vec, d);
        FT_out[idx + out_off] = in_shell ? imag_unit * k_phys * FT_in[idx + in_off]
                                         : Kokkos::complex<Real>(0.0, 0.0);
      });

  fft_mgr->Backward(FT_scratch.data() + scratch_offset, deriv_out.data() + out_offset);
  Kokkos::fence();
}

void SpectralDivergence(parthenon::FFTManager *fft_mgr,
                        const parthenon::ParArray1D<Kokkos::complex<Real>> &FT_vec,
                        parthenon::ParArray1D<Kokkos::complex<Real>> &FT_scratch,
                        parthenon::ParArray1D<Real> &div_out, Real two_pi_over_L) {
  auto fb = fft_mgr->fourier_space_box();
  auto kernel_helper = fft_mgr->GetKernelHelper();
  const auto fft_size_outbox = fft_mgr->size_fourier_space_box();

  auto FT_in = FT_vec.data();
  auto FT_out = FT_scratch.data();
  const Kokkos::complex<Real> imag_unit(0.0, 1.0);
  const Real scale = two_pi_over_L;

  parthenon::par_for(
      "SpectralDiv", fb.low[2], fb.high[2], fb.low[1], fb.high[1], fb.low[0], fb.high[0],
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto k_vec = kernel_helper.Wavevector(k, j, i);
        auto idx = kernel_helper.FourierFlatIndex(k, j, i);
        auto sum = Kokkos::complex<Real>(0.0, 0.0);
        for (int d = 0; d < 3; d++) {
          Real k_phys = scale * ComponentWavenumber(k_vec, d);
          sum += k_phys * FT_in[idx + d * fft_size_outbox];
        }
        FT_out[idx] = imag_unit * sum;
      });

  fft_mgr->Backward(FT_scratch.data(), div_out.data());
  Kokkos::fence();
}

} // namespace energy_transfer
