#include <cmath>
#include <iostream>
#include <memory>

#include <globals.hpp>
#include <interface/metadata.hpp>
#include <interface/state_descriptor.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon_manager.hpp>
#include <utils/error_checking.hpp>
#include <utils/uniform_grid_helper.hpp>

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#include "energy_transfer/helmholtz.hpp"
#include "energy_transfer/ingest.hpp"
#include "energy_transfer/spectral_kernels.hpp"

using parthenon::Real;

namespace {

// "test_vec" components: an analytic field F = grad(phi0) + curl(A0) + mean,
// its known potentials phi0 and A0, and an unstructured pseudo-random field.
enum { IF = 0, IPHI0 = 3, IA0 = 4, IR = 7, NCOMP = 10 };

// Field mean, deliberately nonzero: it must end up in neither potential.
constexpr Real kMean[3] = {0.7, -0.2, 0.1};

parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &) {
  parthenon::Packages_t packages;
  auto package = std::make_shared<parthenon::StateDescriptor>("test_helmholtz");
  auto m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::OneCopy},
                               std::vector<int>{NCOMP});
  package->AddField("test_vec", m);
  packages.Add(package);
  return packages;
}

Real GlobalMax(Real v) {
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_PARTHENON_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  return v;
}

Real MaxAbsDiff(const parthenon::ParArray1D<Real> &a, const parthenon::ParArray1D<Real> &b) {
  Real result = 0.0;
  Kokkos::parallel_reduce(
      "MaxAbsDiff", Kokkos::RangePolicy<>(0, a.extent(0)),
      KOKKOS_LAMBDA(const std::size_t idx, Real &m) {
        const Real d = Kokkos::fabs(a(idx) - b(idx));
        if (d > m) m = d;
      },
      Kokkos::Max<Real>(result));
  return GlobalMax(result);
}

} // namespace

// Unit test for energy_transfer::ComputeHelmholtzFourier/CalcHelmholtzPotentials.
// Run on a box with L != 2*pi (see CMakeLists.txt) so a missing 2*pi/L
// wavenumber scaling can't go unnoticed.
//  1. Analytic: F built from known potentials (phi0 with zero mean, A0
//     divergence-free with zero mean, plus a constant) must give back exactly
//     phi0 and A0 -- this pins signs, scaling, and the k=0 handling.
//  2. Random field: per Fourier mode (k != 0), i k phi_hat + i k x A_hat must
//     reconstruct F_hat, and k . A_hat must vanish (Coulomb gauge).
int main(int argc, char *argv[]) {
  parthenon::ParthenonManager pman;
  pman.app_input->ProcessPackages = ProcessPackages;
  pman.app_input->ProblemGenerator = [](parthenon::MeshBlock *pmb, parthenon::ParameterInput *) {
    auto &rc = pmb->meshblock_data.Get();
    auto v = rc->PackVariables(std::vector<std::string>{"test_vec"});
    auto ib = pmb->cellbounds.GetBoundsI(parthenon::IndexDomain::interior);
    auto jb = pmb->cellbounds.GetBoundsJ(parthenon::IndexDomain::interior);
    auto kb = pmb->cellbounds.GetBoundsK(parthenon::IndexDomain::interior);
    auto coords = pmb->coords;
    const auto &ms = pmb->pmy_mesh->mesh_size;
    const Real a = 2.0 * M_PI / (ms.xmax(parthenon::X1DIR) - ms.xmin(parthenon::X1DIR));
    const Real m0 = kMean[0], m1 = kMean[1], m2 = kMean[2];
    parthenon::par_for(
        "FillTestVec", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real x = coords.Xc<1>(i);
          const Real y = coords.Xc<2>(j);
          const Real z = coords.Xc<3>(k);
          // phi0 = sin(a x) cos(2 a y)
          // A0   = (sin(2 a z), 0, cos(a x) sin(3 a y))   -- div A0 = 0
          v(IPHI0, k, j, i) = Kokkos::sin(a * x) * Kokkos::cos(2 * a * y);
          v(IA0 + 0, k, j, i) = Kokkos::sin(2 * a * z);
          v(IA0 + 1, k, j, i) = 0.0;
          v(IA0 + 2, k, j, i) = Kokkos::cos(a * x) * Kokkos::sin(3 * a * y);
          // F = grad(phi0) + curl(A0) + mean
          v(IF + 0, k, j, i) = a * Kokkos::cos(a * x) * Kokkos::cos(2 * a * y) +
                               3 * a * Kokkos::cos(a * x) * Kokkos::cos(3 * a * y) + m0;
          v(IF + 1, k, j, i) = -2 * a * Kokkos::sin(a * x) * Kokkos::sin(2 * a * y) +
                               2 * a * Kokkos::cos(2 * a * z) +
                               a * Kokkos::sin(a * x) * Kokkos::sin(3 * a * y) + m1;
          v(IF + 2, k, j, i) = m2;
          // Deterministic pseudo-random noise, a function of position only
          // so it doesn't depend on the block decomposition.
          for (int n = 0; n < 3; n++) {
            const Real s = Kokkos::sin(12.9898 * x + 78.233 * y + 37.719 * z + 4.1 * n) * 43758.5453;
            v(IR + n, k, j, i) = s - Kokkos::floor(s) - 0.5;
          }
        });
  };

  auto status = pman.ParthenonInitEnv(argc, argv);
  if (status != parthenon::ParthenonStatus::ok) {
    pman.ParthenonFinalize();
    return status == parthenon::ParthenonStatus::complete ? 0 : 1;
  }
  pman.ParthenonInitPackagesAndMesh();

  int result_code = 0;
  {
    auto *pmesh = pman.pmesh.get();
    auto FFTMgr = pmesh->GetFFTManager();
    const auto fft_size_inbox = FFTMgr->size_real_space_box();
    const auto fft_size_outbox = FFTMgr->size_fourier_space_box();
    constexpr Real tol = 1e-10;
    bool ok = true;

    // ---- 1. Analytic potentials ----
    auto F = energy_transfer::GatherLiveVector(pmesh, "test_vec", {IF, IF + 1, IF + 2});
    auto A0 = energy_transfer::GatherLiveVector(pmesh, "test_vec", {IA0, IA0 + 1, IA0 + 2});
    parthenon::ParArray1D<Real> phi0("phi0", fft_size_inbox);
    pmesh->GetUniformGridHelper()->GatherField("test_vec", IPHI0, phi0);
    Kokkos::fence();

    auto pot = energy_transfer::CalcHelmholtzPotentials(pmesh, F);
    const Real err_phi = MaxAbsDiff(pot.phi, phi0);
    const Real err_A = MaxAbsDiff(pot.A, A0);
    if (parthenon::Globals::my_rank == 0) {
      std::cout << "analytic: max|phi - phi0| = " << err_phi << ", max|A - A0| = " << err_A
                << std::endl;
    }
    if (!(err_phi < tol && err_A < tol)) ok = false;

    // ---- 2. Random field: per-mode reconstruction and Coulomb gauge ----
    auto R = energy_transfer::GatherLiveVector(pmesh, "test_vec", {IR, IR + 1, IR + 2});
    auto h = energy_transfer::ComputeHelmholtzFourier(pmesh, R);
    const auto &ms = pmesh->mesh_size;
    const Real two_pi_over_L =
        2.0 * M_PI / (ms.xmax(parthenon::X1DIR) - ms.xmin(parthenon::X1DIR));
    auto FT_F = h.FT_F;
    auto FT_phi = h.FT_phi;
    auto FT_A = h.FT_A;
    auto fb = FFTMgr->fourier_space_box();
    auto kh = FFTMgr->GetKernelHelper();
    const Kokkos::complex<Real> I(0.0, 1.0);
    Real max_resid = 0.0, max_div_A = 0.0, max_F = 0.0;
    Kokkos::parallel_reduce(
        "HelmholtzResidual",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({fb.low[2], fb.low[1], fb.low[0]},
                                               {fb.high[2] + 1, fb.high[1] + 1, fb.high[0] + 1}),
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &resid, Real &divA, Real &fmax) {
          auto kv = kh.Wavevector(k, j, i);
          const Real kx = Real(energy_transfer::ComponentWavenumber(kv, 0)) * two_pi_over_L;
          const Real ky = Real(energy_transfer::ComponentWavenumber(kv, 1)) * two_pi_over_L;
          const Real kz = Real(energy_transfer::ComponentWavenumber(kv, 2)) * two_pi_over_L;
          const auto idx = kh.FourierFlatIndex(k, j, i);
          const Kokkos::complex<Real> Fh[3] = {FT_F[idx], FT_F[idx + fft_size_outbox],
                                               FT_F[idx + 2 * fft_size_outbox]};
          for (int n = 0; n < 3; n++) fmax = Kokkos::fmax(fmax, Kokkos::abs(Fh[n]));
          if (kx * kx + ky * ky + kz * kz == Real(0.0)) return;
          const Kokkos::complex<Real> Ah[3] = {FT_A[idx], FT_A[idx + fft_size_outbox],
                                               FT_A[idx + 2 * fft_size_outbox]};
          const auto p = FT_phi[idx];
          const Kokkos::complex<Real> rec[3] = {I * kx * p + I * (ky * Ah[2] - kz * Ah[1]),
                                                I * ky * p + I * (kz * Ah[0] - kx * Ah[2]),
                                                I * kz * p + I * (kx * Ah[1] - ky * Ah[0])};
          for (int n = 0; n < 3; n++) resid = Kokkos::fmax(resid, Kokkos::abs(rec[n] - Fh[n]));
          divA = Kokkos::fmax(divA, Kokkos::abs(kx * Ah[0] + ky * Ah[1] + kz * Ah[2]));
        },
        Kokkos::Max<Real>(max_resid), Kokkos::Max<Real>(max_div_A), Kokkos::Max<Real>(max_F));
    max_resid = GlobalMax(max_resid);
    max_div_A = GlobalMax(max_div_A);
    max_F = GlobalMax(max_F);
    const Real rel_resid = max_resid / max_F;
    const Real rel_div_A = max_div_A / max_F;
    if (parthenon::Globals::my_rank == 0) {
      std::cout << "random: max|recon - F_hat|/max|F_hat| = " << rel_resid
                << ", max|k.A_hat|/max|F_hat| = " << rel_div_A << std::endl;
    }
    if (!(rel_resid < tol && rel_div_A < tol)) ok = false;

    if (parthenon::Globals::my_rank == 0) {
      std::cout << (ok ? "PASS" : "FAIL") << ": Helmholtz decomposition" << std::endl;
    }
    result_code = ok ? 0 : 1;
  }

  pman.ParthenonFinalize();
  return result_code;
}
