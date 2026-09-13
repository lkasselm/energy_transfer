#include "energy_transfer/spectral_kernels.hpp"

#include <cmath>

#include <globals.hpp>
#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>
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

parthenon::ParArray2D<parthenon::utils::fft::SpecReal>
BinFourierCospectrum(parthenon::Mesh *pm,
                     const parthenon::ParArray1D<Kokkos::complex<Real>> &FT_a,
                     const parthenon::ParArray1D<Kokkos::complex<Real>> &FT_b, int n_comp) {
  using parthenon::utils::fft::SpecReal;

  PARTHENON_REQUIRE_THROWS(pm != nullptr, "BinFourierCospectrum: mesh pointer must not be null");
  PARTHENON_REQUIRE_THROWS(n_comp > 0, "BinFourierCospectrum: n_comp must be positive");

  auto FFTMgr = pm->GetFFTManager();
  const auto fft_size_outbox = FFTMgr->size_fourier_space_box();
  PARTHENON_REQUIRE_THROWS(
      FT_a.size() == static_cast<std::size_t>(n_comp) * fft_size_outbox &&
          FT_b.size() == static_cast<std::size_t>(n_comp) * fft_size_outbox,
      "BinFourierCospectrum: input arrays have the wrong size for n_comp * "
      "size_fourier_space_box().");

  auto mesh_size = pm->mesh_size;
  const auto nx = mesh_size.nx(parthenon::X1DIR);
  const auto ny = mesh_size.nx(parthenon::X2DIR);
  const auto nz = mesh_size.nx(parthenon::X3DIR);
  const auto k_max = std::sqrt(Real(nx / 2) * Real(nx / 2) + Real(ny / 2) * Real(ny / 2) +
                               Real(nz / 2) * Real(nz / 2));
  const auto num_bins = static_cast<int>(std::ceil(k_max)) + 1;

  parthenon::ParArray2D<SpecReal> spectra("BinFourierCospectrum output", num_bins, 3);
  auto scatter_spectra = Kokkos::Experimental::ScatterView<SpecReal **, parthenon::LayoutWrapper>(
      spectra.KokkosView());

  auto FT_a_in = FT_a.data();
  auto FT_b_in = FT_b.data();
  const int nc = n_comp;
  auto fb = FFTMgr->fourier_space_box();
  auto kernel_helper = FFTMgr->GetKernelHelper();
  parthenon::par_for(
      "BinFourierCospectrum", fb.low[2], fb.high[2], fb.low[1], fb.high[1], fb.low[0], fb.high[0],
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto k_vec = kernel_helper.Wavevector(k, j, i);
        auto k_mag =
            Kokkos::sqrt(Real(k_vec[0] * k_vec[0] + k_vec[1] * k_vec[1] + k_vec[2] * k_vec[2]));
        auto k_mag_int = static_cast<int>(Kokkos::floor(k_mag));
        auto outidx = kernel_helper.FourierFlatIndex(k, j, i);

        Real val = 0.0;
        for (int n = 0; n < nc; n++) {
          const auto a = FT_a_in[outidx + n * fft_size_outbox];
          const auto b = FT_b_in[outidx + n * fft_size_outbox];
          val += a.real() * b.real() + a.imag() * b.imag(); // Re(a * conj(b))
        }

        // k_vec[2] is kx (r2c-stored, always >=0); doubles the contribution
        // for modes whose Hermitian-conjugate partner (at -k, kx<0) isn't
        // separately stored -- identical to CalcSpectrum's own convention.
        const auto fac = ((k_vec[2] > 0) && (2 * k_vec[2] != nx)) ? 2.0 : 1.0;
        auto s = scatter_spectra.access();
        s(k_mag_int, 0) += fac * SpecReal(val);
        s(k_mag_int, 1) += fac * k_mag;
        s(k_mag_int, 2) += fac * 1.0;
      });

  Kokkos::Experimental::contribute(spectra.KokkosView(), scatter_spectra);
  Kokkos::fence();

#ifdef MPI_PARALLEL
  PARTHENON_REQUIRE_THROWS(sizeof(SpecReal) == sizeof(double),
                           "Need to fix comm data types manually.");
  if (parthenon::Globals::my_rank == 0) {
    PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, spectra.data(), spectra.size(), MPI_DOUBLE,
                                   MPI_SUM, 0, MPI_COMM_WORLD));
  } else {
    PARTHENON_MPI_CHECK(MPI_Reduce(spectra.data(), spectra.data(), spectra.size(), MPI_DOUBLE,
                                   MPI_SUM, 0, MPI_COMM_WORLD));
  }
#endif

  return spectra;
}

parthenon::ParArray2D<parthenon::utils::fft::SpecReal>
BinFourierSpectrum(parthenon::Mesh *pm,
                  const parthenon::ParArray1D<Kokkos::complex<Real>> &FT_field, int n_comp) {
  return BinFourierCospectrum(pm, FT_field, FT_field, n_comp);
}

} // namespace energy_transfer
