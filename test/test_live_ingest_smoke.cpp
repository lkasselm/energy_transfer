#include <cmath>
#include <iostream>
#include <memory>

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
  auto package = std::make_shared<parthenon::StateDescriptor>("test_live_ingest");
  auto m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::Derived,
                                parthenon::Metadata::OneCopy, parthenon::Metadata::Vector},
                               std::vector<int>{NPRIM});
  package->AddField("prim", m);
  packages.Add(package);
  return packages;
}

} // namespace

// Single-rank smoke test for the on-the-fly (live) entry point: fills a
// packed "prim" field the way AthenaPK would (constant density/pressure,
// velocity as one Fourier mode), then calls ComputeShellTransferLive with no
// package/StateDescriptor callbacks involved beyond the bare field -- proving
// the live path needs nothing more than a Mesh*/MeshData<Real>* an app
// already has inside its own UserWorkBeforeOutput.
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
          prim(IV1, k, j, i) = Kokkos::cos(2.0 * M_PI * 3 * i / Real(Nx));
          prim(IV2, k, j, i) = 0.0;
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
    energy_transfer::ShellTransferConfig cfg;
    cfg.binning = energy_transfer::BinningSpec::Linear(4);
    cfg.terms = {"UUA"};

    auto res = energy_transfer::ComputeShellTransferLive(pmesh, md.get(), spec, cfg);

    const bool has_uua = res.matrices.count("UUA") == 1;
    bool all_finite = true;
    if (has_uua) {
      const auto &m = res.matrices.at("UUA");
      for (int q = 0; q < static_cast<int>(m.extent(0)); q++) {
        for (int k = 0; k < static_cast<int>(m.extent(1)); k++) {
          if (!std::isfinite(m(q, k))) all_finite = false;
        }
      }
    }

    if (parthenon::Globals::my_rank == 0) {
      if (has_uua && all_finite && res.n_shells == 4) {
        std::cout << "PASS: ComputeShellTransferLive produced a finite " << res.n_shells
                  << "x" << res.n_shells << " UUA matrix.\n";
      } else {
        std::cout << "FAIL: ComputeShellTransferLive result missing/non-finite/wrong shape.\n";
        result = 1;
      }
    }
  }

  pman.ParthenonFinalize();
  return result;
}
