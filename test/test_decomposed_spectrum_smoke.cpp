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
#include "energy_transfer/ingest.hpp"
#include "energy_transfer/spectra.hpp"

using parthenon::Real;

namespace {

enum { IDN = 0, IV1 = 1, IV2 = 2, IV3 = 3, IPR = 4, NPRIM = 5 };

parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &) {
  parthenon::Packages_t packages;
  auto package = std::make_shared<parthenon::StateDescriptor>("test_decomposed_spectrum");
  auto m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::Derived,
                                parthenon::Metadata::OneCopy, parthenon::Metadata::Vector},
                               std::vector<int>{NPRIM});
  package->AddField("prim", m);
  packages.Add(package);
  return packages;
}

} // namespace

// Single-rank smoke test for the compressive/plus/minus decomposed spectra
// (energy_transfer::ComputeSpectra, requested via the single "spec_U_decomp"
// bundle name, which shares one Forward() FFT across full/compressive/plus/
// minus -- see decomposition.hpp's DecomposeFourierField and
// spectral_kernels.hpp's BinFourierSpectrum). Checks:
// (1) spec_U_compressive + spec_U_plus + spec_U_minus reconstructs spec_U at
//     every bin >= 1 (Parseval/orthonormality -- see decomposition.hpp).
//     Bin 0 (the DC mode) is NOT expected to satisfy this: direction is
//     undefined at k=0, so compressive/plus/minus are all exactly zero
//     there by construction, with no special-cased convention routing the
//     DC mode's actual power anywhere.
// (2) the k-sum/count columns are bit-identical across all four spectra at
//     every bin including bin 0 -- they depend only on |k|, never on the
//     field values or which component.
// (3) bin 0 specifically: compressive == plus == minus == 0 (exactly).
// The synthetic field includes a pure-kz mode (IV3 varies only with the
// z-index) so BuildHelicalBasis's kperp==0 fallback branch is actually
// exercised, unlike test_mediator_decomposition_smoke.cpp's field.
int main(int argc, char *argv[]) {
  parthenon::ParthenonManager pman;
  pman.app_input->ProcessPackages = ProcessPackages;
  pman.app_input->ProblemGenerator = [](parthenon::MeshBlock *pmb,
                                        parthenon::ParameterInput *) {
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
          prim(IV1, k, j, i) = Kokkos::cos(2.0 * M_PI * 3 * i / Real(Nx)) +
                              0.5 * Kokkos::sin(2.0 * M_PI * 5 * i / Real(Nx));
          prim(IV2, k, j, i) = 0.3 * Kokkos::cos(2.0 * M_PI * 2 * j / Real(Nx));
          // Pure-kz mode (kx=ky=0) -- exercises BuildHelicalBasis's kperp==0
          // fallback, which the mediator smoke test's field never touches.
          prim(IV3, k, j, i) = 0.4 * Kokkos::sin(2.0 * M_PI * 4 * k / Real(Nx));
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
                                                                /*has_bfield=*/false);

    auto fields = energy_transfer::GatherLiveFields(pmesh, md.get(), spec);
    energy_transfer::ConvertConservedToPrimitive(fields);
    auto spectra = energy_transfer::ComputeSpectra(pmesh, fields, {"spec_U_decomp"});

    bool have_all = spectra.count("spec_U") == 1 && spectra.count("spec_U_compressive") == 1 &&
                    spectra.count("spec_U_plus") == 1 && spectra.count("spec_U_minus") == 1;
    bool sums_match = true, k_and_count_match = true, dc_ok = true;
    Real max_sum_abs_diff = 0.0;

    if (have_all) {
      const auto &full = spectra.at("spec_U");
      const auto &comp = spectra.at("spec_U_compressive");
      const auto &plus = spectra.at("spec_U_plus");
      const auto &minus = spectra.at("spec_U_minus");
      const int num_bins = static_cast<int>(full.extent(0));
      have_all = have_all && comp.extent(0) == full.extent(0) && plus.extent(0) == full.extent(0) &&
                minus.extent(0) == full.extent(0);

      // Bin 0 (the DC mode) is excluded from the sum-identity check -- see
      // the file-level comment above.
      for (int b = 1; b < num_bins; b++) {
        const Real summed = comp(b, 0) + plus(b, 0) + minus(b, 0);
        const Real diff = std::abs(summed - full(b, 0));
        max_sum_abs_diff = std::max(max_sum_abs_diff, diff);
        if (diff > 1e-8 * std::max(std::abs(full(b, 0)), Real(1.0))) sums_match = false;
      }

      for (int b = 0; b < num_bins; b++) {
        for (int col = 1; col < 3; col++) {
          if (comp(b, col) != full(b, col) || plus(b, col) != full(b, col) ||
              minus(b, col) != full(b, col)) {
            k_and_count_match = false;
          }
        }
      }

      if (num_bins > 0) {
        if (comp(0, 0) != Real(0.0) || plus(0, 0) != Real(0.0) || minus(0, 0) != Real(0.0)) {
          dc_ok = false;
        }
      }
    }

    if (parthenon::Globals::my_rank == 0) {
      std::cout << "decomposed spectrum sum vs full (bins >= 1): max_abs_diff=" << max_sum_abs_diff
                << std::endl;
      if (have_all && sums_match && k_and_count_match && dc_ok) {
        std::cout << "PASS: compressive+plus+minus reconstructs spec_U at every bin >= 1, "
                     "k-sum/count columns match at every bin, and compressive/plus/minus are "
                     "all exactly zero at the DC mode (bin 0).\n";
      } else {
        std::cout << "FAIL: have_all=" << have_all << " sums_match=" << sums_match
                  << " k_and_count_match=" << k_and_count_match << " dc_ok=" << dc_ok << "\n";
        result_code = 1;
      }
    }
  }

  pman.ParthenonFinalize();
  return result_code;
}
