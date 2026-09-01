#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <globals.hpp>
#include <interface/metadata.hpp>
#include <interface/state_descriptor.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon_manager.hpp>
#include <utils/error_checking.hpp>

#include "energy_transfer/field_spec.hpp"
#include "energy_transfer/shell_transfer.hpp"

using parthenon::Real;

namespace {

enum { IDN = 0, IV1 = 1, IV2 = 2, IV3 = 3, IPR = 4, NPRIM = 5 };

parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &) {
  parthenon::Packages_t packages;
  auto package = std::make_shared<parthenon::StateDescriptor>("test_mediator_decomposition");
  auto m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::Derived,
                                parthenon::Metadata::OneCopy, parthenon::Metadata::Vector},
                               std::vector<int>{NPRIM});
  package->AddField("prim", m);
  packages.Add(package);
  return packages;
}

} // namespace

// Single-rank smoke test for DecompositionMode::mediator_resolved: checks
// (1) a mediator-decomposed UUA matrix has the expected 3D (n_q, n_m, n_k)
// shape and is finite, (2) summing it over the mediator axis reproduces the
// non-mediator-decomposed result -- U_dot_grad_W is linear in its U
// mediator, and shells partition Fourier space, so shell-filtering U into
// n_shells pieces and summing must reconstruct the unfiltered-U result
// exactly (up to floating point), and (3) requesting mediator decomposition
// on a term with no decomposable mediator (PU, whose only mediator is a
// rho scaling) throws rather than silently ignoring the request.
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
          prim(IV3, k, j, i) = 0.0;
        });
  };

  auto status = pman.ParthenonInitEnv(argc, argv);
  if (status != parthenon::ParthenonStatus::ok) {
    pman.ParthenonFinalize();
    return status == parthenon::ParthenonStatus::complete ? 0 : 1;
  }
  pman.ParthenonInitPackagesAndMesh();

  int result = 0;
  {
    auto *pmesh = pman.pmesh.get();
    auto &md = pmesh->mesh_data.Get();
    auto spec = energy_transfer::MakeAthenaPKPrimitiveLiveSpec(IDN, IV1, IV2, IV3, IPR,
                                                                /*has_bfield=*/false);

    energy_transfer::ShellTransferConfig cfg_plain;
    cfg_plain.donor_binning = cfg_plain.mediator_binning = cfg_plain.receiver_binning =
        energy_transfer::BinningSpec::Linear(4);
    cfg_plain.terms = {{"UUA", energy_transfer::DecompositionMode::Full()}};
    auto res_plain = energy_transfer::ComputeShellTransferLive(pmesh, md.get(), spec, cfg_plain);

    energy_transfer::ShellTransferConfig cfg_mediator;
    cfg_mediator.donor_binning = cfg_mediator.mediator_binning = cfg_mediator.receiver_binning =
        energy_transfer::BinningSpec::Linear(4);
    cfg_mediator.terms = {{"UUA", energy_transfer::DecompositionMode::FullWithMediator()}};
    auto res_mediator =
        energy_transfer::ComputeShellTransferLive(pmesh, md.get(), spec, cfg_mediator);

    bool shape_ok = false, all_finite = true, sums_match = true;
    Real max_abs_diff = 0.0;
    if (res_plain.matrices.count("UUA") == 1 && res_mediator.matrices.count("UUA") == 1) {
      const auto &plain = res_plain.matrices.at("UUA");
      const auto &med = res_mediator.matrices.at("UUA");
      shape_ok = plain.extent(0) == 4 && plain.extent(1) == 1 && plain.extent(2) == 4 &&
                med.extent(0) == 4 && med.extent(1) == 4 && med.extent(2) == 4;
      if (shape_ok) {
        for (int qi = 0; qi < 4; qi++) {
          for (int ki = 0; ki < 4; ki++) {
            Real summed = 0.0;
            for (int mi = 0; mi < 4; mi++) {
              const Real v = med(qi, mi, ki);
              if (!std::isfinite(v)) all_finite = false;
              summed += v;
            }
            const Real plain_val = plain(qi, 0, ki);
            const Real diff = std::abs(summed - plain_val);
            max_abs_diff = std::max(max_abs_diff, diff);
            if (diff > 1e-8 * std::max(std::abs(plain_val), Real(1.0))) sums_match = false;
          }
        }
      }
    }

    bool pu_rejected = false;
    try {
      energy_transfer::ShellTransferConfig cfg_pu;
      cfg_pu.donor_binning = cfg_pu.mediator_binning = cfg_pu.receiver_binning =
          energy_transfer::BinningSpec::Linear(4);
      cfg_pu.terms = {{"PU", energy_transfer::DecompositionMode::FullWithMediator()}};
      energy_transfer::ComputeShellTransferLive(pmesh, md.get(), spec, cfg_pu);
    } catch (const std::runtime_error &) {
      pu_rejected = true;
    }

    if (parthenon::Globals::my_rank == 0) {
      std::cout << "UUA mediator-sum vs plain: max_abs_diff=" << max_abs_diff << std::endl;
      if (shape_ok && all_finite && sums_match && pu_rejected) {
        std::cout << "PASS: mediator decomposition shapes/values are consistent, and PU "
                     "correctly rejects mediator_resolved=true.\n";
      } else {
        std::cout << "FAIL: shape_ok=" << shape_ok << " all_finite=" << all_finite
                  << " sums_match=" << sums_match << " pu_rejected=" << pu_rejected << "\n";
        result = 1;
      }
    }
  }

  pman.ParthenonFinalize();
  return result;
}
