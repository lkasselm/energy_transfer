#include <cmath>
#include <iostream>

#include <globals.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon_manager.hpp>
#include <utils/error_checking.hpp>

#include "energy_transfer/spectral_kernels.hpp"

using parthenon::Real;

// Single-rank smoke test: fills a real-space scalar field with one known
// Fourier mode (kx=3) and checks that ShellFilter preserves it when the
// shell contains |k|=3 and zeroes it out when the shell doesn't.
int main(int argc, char *argv[]) {
  parthenon::ParthenonManager pman;
  pman.app_input->ProblemGenerator = [](parthenon::MeshBlock *, parthenon::ParameterInput *) {};

  auto status = pman.ParthenonInitEnv(argc, argv);
  if (status != parthenon::ParthenonStatus::ok) {
    pman.ParthenonFinalize();
    return status == parthenon::ParthenonStatus::complete ? 0 : 1;
  }
  pman.ParthenonInitPackagesAndMesh();

  int result = 0;
  {
    auto *pmesh = pman.pmesh.get();
    auto FFTMgr = pmesh->GetFFTManager();
    const auto n_in = FFTMgr->size_real_space_box();
    const auto n_out = FFTMgr->size_fourier_space_box();

    auto helper = FFTMgr->GetKernelHelper();
    auto inbox = FFTMgr->real_space_box();
    const auto &mesh_size = pmesh->mesh_size;
    const int Nx = mesh_size.nx(parthenon::X1DIR);
    const int kx_mode = 3;

    parthenon::ParArray1D<Real> field("field", n_in);
    parthenon::par_for(
        "FillMode", inbox.low[2], inbox.high[2], inbox.low[1], inbox.high[1], inbox.low[0],
        inbox.high[0], KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const auto idx = helper.RealFlatIndex(k, j, i);
          field(idx) = Kokkos::cos(2.0 * M_PI * kx_mode * i / Real(Nx));
        });
    Kokkos::fence();

    parthenon::ParArray1D<Kokkos::complex<Real>> FT_field("FT_field", n_out);
    FFTMgr->Forward(field.data(), FT_field.data());

    parthenon::ParArray1D<Kokkos::complex<Real>> scratch("scratch", n_out);
    parthenon::ParArray1D<Real> filtered_in("filtered_in", n_in);
    parthenon::ParArray1D<Real> filtered_out("filtered_out", n_in);

    energy_transfer::ShellFilter(FFTMgr, 1, FT_field, scratch, filtered_in, 2.5, 3.5);
    energy_transfer::ShellFilter(FFTMgr, 1, FT_field, scratch, filtered_out, 4.5, 5.5);

    Real sum_sq_in = 0.0, sum_sq_out = 0.0, sum_sq_original = 0.0;
    Kokkos::parallel_reduce(
        "CheckShellFilter", Kokkos::RangePolicy<>(0, n_in),
        KOKKOS_LAMBDA(const std::size_t idx, Real &a, Real &b, Real &c) {
          a += filtered_in(idx) * filtered_in(idx);
          b += filtered_out(idx) * filtered_out(idx);
          c += field(idx) * field(idx);
        },
        sum_sq_in, sum_sq_out, sum_sq_original);

    const bool preserved = sum_sq_in > 0.4 * sum_sq_original;
    const bool excluded = sum_sq_out < 1e-6 * sum_sq_original;

    if (parthenon::Globals::my_rank == 0) {
      std::cout << "sum_sq_original=" << sum_sq_original << " in-shell=" << sum_sq_in
                << " out-of-shell=" << sum_sq_out << std::endl;
      if (preserved && excluded) {
        std::cout << "PASS: ShellFilter isolates the requested shell.\n";
      } else {
        std::cout << "FAIL: ShellFilter did not isolate the requested shell.\n";
        result = 1;
      }
    }
  }

  pman.ParthenonFinalize();
  return result;
}
