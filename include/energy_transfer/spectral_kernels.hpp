#ifndef ENERGY_TRANSFER_SPECTRAL_KERNELS_HPP_
#define ENERGY_TRANSFER_SPECTRAL_KERNELS_HPP_

#include <array>

#include <basic_types.hpp>
#include <kokkos_types.hpp>
#include <utils/fft_manager.hpp>

namespace energy_transfer {

using parthenon::Real;

KOKKOS_INLINE_FUNCTION
inline int ComponentWavenumber(const std::array<int, 3> &kji_vec, const int dir) {
  return kji_vec[2 - dir];
}

// Shell-filters a field in Fourier space and IFFTs to real space. Extracts
// modes with k_low < |k| <= k_high. FT_field/FT_scratch are n_comp *
// fft_size_outbox; real_out is n_comp * fft_size_inbox.
void ShellFilter(parthenon::FFTManager *fft_mgr, int n_comp,
                 const parthenon::ParArray1D<Kokkos::complex<Real>> &FT_field,
                 parthenon::ParArray1D<Kokkos::complex<Real>> &FT_scratch,
                 parthenon::ParArray1D<Real> &real_out, Real k_low, Real k_high);

// Fused shell-filter + spectral derivative: computes d(field_Q)/dx_dir for
// modes in shell (k_low, k_high], writing a single component into deriv_out
// at out_offset. comp_offset/scratch_offset index into the (possibly
// multi-component) FT_field_full/FT_scratch arrays.
void ShellFilterDerivative(parthenon::FFTManager *fft_mgr,
                           const parthenon::ParArray1D<Kokkos::complex<Real>> &FT_field_full,
                           int comp_offset,
                           parthenon::ParArray1D<Kokkos::complex<Real>> &FT_scratch,
                           int scratch_offset, parthenon::ParArray1D<Real> &deriv_out,
                           int out_offset, Real k_low, Real k_high, int dir,
                           Real two_pi_over_L);

// Spectral divergence of a 3-component vector field. FT_vec is 3 *
// fft_size_outbox; div_out is fft_size_inbox.
void SpectralDivergence(parthenon::FFTManager *fft_mgr,
                        const parthenon::ParArray1D<Kokkos::complex<Real>> &FT_vec,
                        parthenon::ParArray1D<Kokkos::complex<Real>> &FT_scratch,
                        parthenon::ParArray1D<Real> &div_out, Real two_pi_over_L);

} // namespace energy_transfer

#endif // ENERGY_TRANSFER_SPECTRAL_KERNELS_HPP_
