#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

#include <globals.hpp>
#include <interface/metadata.hpp>
#include <interface/state_descriptor.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon_manager.hpp>
#include <utils/error_checking.hpp>

#include "energy_transfer/convert.hpp"
#include "energy_transfer/field_spec.hpp"
#include "energy_transfer/helicity.hpp"
#include "energy_transfer/ingest.hpp"
#include "energy_transfer/spectra.hpp"

using parthenon::Real;

namespace {

enum { IDN = 0, IV1 = 1, IV2 = 2, IV3 = 3, IPR = 4, IB1 = 5, IB2 = 6, IB3 = 7, NPRIM = 8 };

parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &) {
  parthenon::Packages_t packages;
  auto package = std::make_shared<parthenon::StateDescriptor>("test_helicity_smoke");
  auto m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::Derived,
                                parthenon::Metadata::OneCopy, parthenon::Metadata::Vector},
                               std::vector<int>{NPRIM});
  package->AddField("prim", m);
  packages.Add(package);
  return packages;
}

// sum over a device array via Kokkos::parallel_reduce -- avoids needing a
// host mirror just to loop over values.
Real DeviceSum(const parthenon::ParArray1D<Real> &v, std::size_t n) {
  Real total = 0.0;
  Kokkos::parallel_reduce(
      "DeviceSum", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(const std::size_t idx, Real &s) { s += v(idx); }, total);
  return total;
}

Real DeviceSumSquares(const parthenon::ParArray1D<Real> &v, std::size_t n) {
  Real total = 0.0;
  Kokkos::parallel_reduce(
      "DeviceSumSquares", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(const std::size_t idx, Real &s) { s += v(idx) * v(idx); }, total);
  return total;
}

int CountNonFinite(const parthenon::ParArray1D<Real> &v, std::size_t n) {
  int count = 0;
  Kokkos::parallel_reduce(
      "CountNonFinite", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(const std::size_t idx, int &c) {
        const Real h = v(idx);
        // Portable finite check (no library isfinite needed in device code):
        // NaN fails self-equality; Inf fails the magnitude bound.
        const bool finite = (h == h) && (h > Real(-1e300)) && (h < Real(1e300));
        if (!finite) c += 1;
      },
      count);
  return count;
}

Real SumBinColumn0(const parthenon::HostArray2D<energy_transfer::TransferReal> &spec) {
  energy_transfer::TransferReal total = 0.0;
  for (int b = 0; b < static_cast<int>(spec.extent(0)); b++) total += spec(b, 0);
  return static_cast<Real>(total);
}

} // namespace

// Single-rank smoke test for energy_transfer::CalcHelicity and the
// spec_helicity / spec_helicity_variance spectra. No external reference
// needed -- everything is checked via Parseval identities against this
// library's own already-validated plain power spectrum (spec_B), which
// pins down the exact real-space-sum <-> binned-Fourier-sum normalization:
// FFTManager::Forward() applies heffte::scale::full (divides by
// N = Nx*Ny*Nz), so sum_x|field(x)|^2 == fft_size_inbox * sum_bins(pow_sum)
// -- NOT a 1:1 ratio. spec_B (already exercised/validated elsewhere this
// session, including against the Python reference tool) confirms that
// factor empirically before it's used to check the two new spectra.
int main(int argc, char *argv[]) {
  parthenon::ParthenonManager pman;
  pman.app_input->ProcessPackages = ProcessPackages;
  pman.app_input->ProblemGenerator = [](parthenon::MeshBlock *pmb, parthenon::ParameterInput *) {
    auto &rc = pmb->meshblock_data.Get();
    auto prim = rc->PackVariables(std::vector<std::string>{"prim"});
    auto ib = pmb->cellbounds.GetBoundsI(parthenon::IndexDomain::interior);
    auto jb = pmb->cellbounds.GetBoundsJ(parthenon::IndexDomain::interior);
    auto kb = pmb->cellbounds.GetBoundsK(parthenon::IndexDomain::interior);
    const int Nx = pmb->pmy_mesh->mesh_size.nx(parthenon::X1DIR);
    parthenon::par_for(
        "FillPrim", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          prim(IDN, k, j, i) = 1.0;
          prim(IPR, k, j, i) = 1.0;
          prim(IV1, k, j, i) = 0.0;
          prim(IV2, k, j, i) = 0.0;
          prim(IV3, k, j, i) = 0.0;
          // A multi-mode field with genuine helical structure -- not a pure
          // Beltrami mode, so H(x) actually fluctuates in space rather than
          // being a spatially uniform constant.
          prim(IB1, k, j, i) = Kokkos::cos(2.0 * M_PI * 2 * i / Real(Nx)) +
                              0.3 * Kokkos::sin(2.0 * M_PI * 1 * k / Real(Nx));
          prim(IB2, k, j, i) = 0.5 * Kokkos::sin(2.0 * M_PI * 3 * j / Real(Nx));
          prim(IB3, k, j, i) = 0.2 * Kokkos::cos(2.0 * M_PI * 1 * i / Real(Nx)) +
                              0.4 * Kokkos::sin(2.0 * M_PI * 2 * j / Real(Nx));
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
    auto &md = pmesh->mesh_data.Get();
    auto spec = energy_transfer::MakeAthenaPKPrimitiveLiveSpec(IDN, IV1, IV2, IV3, IPR,
                                                                /*has_bfield=*/true, IB1, IB2, IB3);
    auto fields = energy_transfer::GatherLiveFields(pmesh, md.get(), spec);
    energy_transfer::ConvertConservedToPrimitive(fields);

    const Real fft_size_inbox = static_cast<Real>(pmesh->GetFFTManager()->size_real_space_box());

    auto H = energy_transfer::CalcHelicity(pmesh, fields.mag);
    const bool all_finite = CountNonFinite(H, H.extent(0)) == 0;
    const Real sum_H = DeviceSum(H, H.extent(0));
    const Real sum_H2 = DeviceSumSquares(H, H.extent(0));
    const Real sum_B2 = DeviceSumSquares(fields.mag, fields.mag.extent(0));

    auto spectra = energy_transfer::ComputeSpectra(
        pmesh, fields, {"spec_B", "spec_helicity", "spec_helicity_variance"});

    const bool have_all = spectra.count("spec_B") == 1 && spectra.count("spec_helicity") == 1 &&
                          spectra.count("spec_helicity_variance") == 1;
    bool checks_ok = true;
    Real max_rel_diff = 0.0;

    if (have_all) {
      const Real sum_B2_from_bins = fft_size_inbox * SumBinColumn0(spectra.at("spec_B"));
      const Real sum_H_from_bins = fft_size_inbox * SumBinColumn0(spectra.at("spec_helicity"));
      const Real sum_H2_from_bins =
          fft_size_inbox * SumBinColumn0(spectra.at("spec_helicity_variance"));

      auto check = [&](Real direct, Real from_bins, const char *label) {
        const Real diff = std::abs(direct - from_bins);
        const Real rel = diff / std::max(std::abs(direct), Real(1.0));
        max_rel_diff = std::max(max_rel_diff, rel);
        if (rel > 1e-8) {
          std::cout << "  FAIL " << label << ": direct=" << direct << " from_bins=" << from_bins
                    << " rel_diff=" << rel << "\n";
          checks_ok = false;
        }
      };
      check(sum_B2, sum_B2_from_bins, "spec_B Parseval (normalization sanity check)");
      check(sum_H, sum_H_from_bins, "spec_helicity integrates to total helicity");
      check(sum_H2, sum_H2_from_bins, "spec_helicity_variance Parseval");
    }

    if (parthenon::Globals::my_rank == 0) {
      std::cout << "helicity checks: max_rel_diff=" << max_rel_diff << std::endl;
      if (have_all && all_finite && checks_ok) {
        std::cout << "PASS: CalcHelicity is finite everywhere, and both spec_helicity and "
                     "spec_helicity_variance satisfy their Parseval identities.\n";
      } else {
        std::cout << "FAIL: have_all=" << have_all << " all_finite=" << all_finite
                  << " checks_ok=" << checks_ok << "\n";
        result_code = 1;
      }
    }
  }

  pman.ParthenonFinalize();
  return result_code;
}
