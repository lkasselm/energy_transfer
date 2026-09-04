# energy_transfer

Shell-to-shell energy transfer analysis for turbulent (M)HD flows on a
uniform-grid Parthenon mesh. A standalone library (depends on Parthenon, but
lives outside both the Parthenon submodule and any single application's
source tree) with two ways to use it:

1. **On the fly**, linked into any Parthenon application: call
   `energy_transfer::ComputeShellTransferLive(...)` from your own hook
   (e.g. `UserWorkBeforeOutput`) with the `Mesh*`/`MeshData<Real>*` you
   already have, then `energy_transfer::WriteResult(...)` to dump the
   result. No `StateDescriptor`/package registration required.
2. **Offline**, via the `energy-transfer-offline` executable built alongside
   the library, which reads either an ADIOS2/bp5 snapshot (e.g. converted
   from Enzo via `scripts/enzo_to_bp5.py`) or a Parthenon HDF5
   (`.phdf`/`.h5`/`.hdf5`) output file -- the format AthenaPK itself writes,
   read via raw HDF5 hyperslab reads of just the region each rank needs --
   and runs the same computation. The format is picked automatically from
   `input_file`'s extension.

See `docs/plan.md` for the physics background and the historical single-file
prototype this library was extracted from
(https://github.com/parthenon-hpc-lab/parthenon/tree/pgrete/energy-transfer).

## Getting the source

Parthenon is vendored as a git submodule (`external/parthenon`, pinned to a
commit on `develop`, `parthenon-hpc-lab/parthenon`'s default branch -- the
FFT/spectral machinery this library depends on has been merged upstream, so
both this repo and AthenaPK's own submodule now track the same mainline
branch), mirroring the same nested-submodule convention AthenaPK itself
uses for Kokkos/Parthenon. `.gitmodules` doesn't pin a `branch =` (`develop`
already is the remote's default, so `git submodule update --remote` resolves
it via `origin/HEAD` on its own) -- either way, the checkout stays pinned to
whatever commit was current until a `--remote` update manually bumps it.
After cloning this repo:

```
git submodule update --init --recursive
```

(Not needed when this repo is added as `add_subdirectory()` from an app that
already builds its own Parthenon, e.g. AthenaPK -- see "In-situ" below; that
vendored copy is simply never configured in that case.)

## Build prerequisite: Parthenon needs `-DPARTHENON_ENABLE_FFT=ON`

`Mesh::GetFFTManager()`/`GetUniformGridHelper()`, `FFTManager`,
`UniformGridHelper`, and `CalcSpectrum` -- everything this library depends
on -- only exist in Parthenon when it was configured with
`-DPARTHENON_ENABLE_FFT=ON` (the single flag for the whole FFT/HeFFTe
machinery; older docs/branches may still call it `PARTHENON_ENABLE_HEFFTE`,
but that name no longer exists in Parthenon's CMake). The top-level
`CMakeLists.txt` here checks this and fails with a clear message rather than
a wall of missing-member compiler errors, whenever the flag's value is
visible to CMake (i.e. when Parthenon is added via `add_subdirectory`,
which is the common case).

## Constraints

Uniform grid, single mesh partition (`parthenon/mesh/pack_size = -1`),
periodic boundary conditions, and a cubic domain (`x1max-x1min ==
x2max-x2min == x3max-x3min`) -- all four are checked at runtime by
`ComputeShellTransfer` and throw a clear error otherwise.

## Public API (see `include/energy_transfer/`)

- `field_spec.hpp` -- `LiveFieldSpec` describes where your app keeps
  density/velocity-or-momentum/magnetic-field/pressure-or-energy/
  acceleration, as a variable name + component index (or indices). Works
  for both separate named fields and a single packed field addressed by
  component (e.g. AthenaPK's `"cons"`/`"prim"`). Use `MakeAthenaPKPrimitiveLiveSpec`/
  `MakeAthenaPKConservedLiveSpec`, passing your own `IDN`/`IV1`/... enum
  values, or `MakeSeparateFieldsLiveSpec` for the historical rho/vel/mag/
  acc/pres layout.
- `shell_transfer.hpp` -- the main entry points:
  - `ShellTransferConfig` -- `donor_binning`/`mediator_binning`/
    `receiver_binning` (each an independent `BinningSpec::Linear/Log/Custom`
    -- e.g. a narrow custom band pinning donor and receiver to two specific
    scales while mediator sweeps a fine `Log()` binning across whichever
    scales mediate that pair's transfer), `terms` (a plain list of names
    selected from the library's fixed built-in set -- see below), `mode`
    (one `DecompositionMode` shared by every term in `terms` -- not
    per-term, see below), and `spectrum_names`.
  - `ComputeShellTransferLive(Mesh*, MeshData<Real>*, LiveFieldSpec, ShellTransferConfig)`
  - `ComputeShellTransferFromFile(Mesh*, input_file, FileFieldNaming, ShellTransferConfig)`
    -- dispatches to the ADIOS2 or Parthenon HDF5 reader based on
    `input_file`'s extension (`ingest.hpp`'s `DetectInputFileFormat`).
  - `ComputeFieldRequirements(cfg)` -- tells you which of magnetic field /
    pressure-or-energy / acceleration the requested terms actually need, so
    you can build a minimal `LiveFieldSpec`/`FileFieldNaming` (the offline
    tool does this automatically).
- `io_openpmd.hpp` -- `WriteResult(...)` writes every computed term/spectrum
  as a named openPMD mesh record to a `.bp` file.

### Built-in terms

`UUA, UUC, BBA, BBC, BUT, UBTb, UBTbA, UBTbC, BUPbb, UBPbb, PU, FU` (see
`docs/plan.md` for their physical meaning) plus spectra `spec_U, spec_rho,
spec_W, spec_B`, and a directionally-decomposed variant of each vector field
requested via a single bundle name -- `spec_U_decomp`, `spec_W_decomp`,
`spec_B_decomp` (`rho` is scalar, no direction to decompose against; `B`'s
bundle pulls in the magnetic field the same way plain `spec_B` does).
Requesting e.g. `spec_B_decomp` computes and writes **four** spectra
together -- `spec_B`, `spec_B_compressive`, `spec_B_plus`, `spec_B_minus`
(compressive = parallel to the wavevector `k`; plus/minus = the two
circularly-polarized "helical" parts perpendicular to `k`) -- in one fused
Forward()+mode-loop pass, since there's rarely a reason to want just one
direction in isolation, and computing them separately would redundantly
re-FFT the same field and redundantly recompute the same per-mode projection
three times over (see `include/energy_transfer/decomposition.hpp` for the
math, including why all three components -- plus/minus included -- are
Hermitian-symmetric and could validly be reconstructed to real space with
this library's existing r2c/c2r FFT if a future extension needs that; the
current spectrum-only use never leaves Fourier space simply because a power
spectrum never needs to, not as a workaround for anything). At every bin,
`spec_B_compressive + spec_B_plus + spec_B_minus` reconstructs `spec_B`
exactly (Parseval/orthonormality of the projection basis) -- including the
`k=0` bin, via the convention that the whole DC mode is assigned to
compressive and plus/minus are zero there, since direction is undefined at
`k=0`. Input decks request a bundle the same way as any other spectrum:
`spectra = spec_U,spec_B_decomp` (see
`tools/energy_transfer_offline/parthinput.example`) -- the individual
`spec_B_compressive` etc. names are not separately requestable.

`DecompositionMode{donor_resolved, mediator_resolved, receiver_resolved}` (a
plain struct, `ShellTransferConfig::mode`'s default is `DecompositionMode::Full()`)
controls which of the three shell axes are resolved per-shell vs collapsed
to the whole domain -- collapsing an axis skips that axis's per-shell loop
entirely rather than summing a full matrix after the fact, so e.g. `Total()`
is O(1) in the number of shells, not O(n_shells^2). **`mode` applies to
every term in `ShellTransferConfig::terms` -- it is not per-term.** This is
deliberate: it lets `ComputeShellTransfer` share one `(Q,M,K)` shell sweep
across all requested terms instead of a separate sweep per term, which is
what keeps peak memory bounded to a small, fixed number of in-flight
shell-filtered fields (the distinct quantity names any requested term
actually needs, typically well under 10) regardless of shell count or term
count -- an earlier per-term-mode version of this API let different terms in
one call want different sweep shapes, which forced an unbounded whole-call
cache and caused real OOMs on long sweeps.

"Donor" (Q) and "receiver" (K) are the fields shell-filtered on each side of
a term's dot product; "mediator" is the field each side's derived quantity
reads directly to relate them (e.g. the advecting velocity `U` in `UUA`, or
the tension field `b` in `BUT`) -- normally read full/unfiltered, but can
optionally be shell-restricted too. Static factories cover all eight
combinations: `Full()`/`BySender()`/`ByReceiver()`/`Total()` (mediator
always collapsed -- the only four that existed before mediator
decomposition), and `FullWithMediator()`/`BySenderWithMediator()`/
`ByReceiverWithMediator()`/`MediatorOnly()` (mediator also resolved). Not
every term has a decomposable mediator -- some terms' only mediator is a
scalar normalization (density, via `sqrt(rho)` scaling) rather than a field
being transported, and setting `mode.mediator_resolved=true` while `terms`
includes such a term (currently `PU` and `FU`) throws immediately, for the
whole call, before any computation starts; see `has_decomposable_mediator`
in `src/registry.cpp`'s `BuiltinQuantities()` for the authoritative list.
`TransferResult::matrices` is always 3D, `(n_q, n_m, n_k)` -- the same shape
for every term, since `mode` is shared -- with `n_m == 1` whenever
`mode.mediator_resolved == false`.

Input-file keys: `terms=` is a plain comma-separated name list (no per-term
suffix), e.g. `terms=UUA,BBA,BUT`; `mode=` sets the one `DecompositionMode`
shared by all of them -- `full` (default), `by_sender`, `by_receiver`,
`total`, or the mediator-resolving variants `full_mediator`,
`by_sender_mediator`, `by_receiver_mediator`, `mediator_only`.

`binning=`/`num_shells=`/`shell_edges=` (unprefixed, unchanged) set a
default binning shared by donor/mediator/receiver, exactly as before
per-axis binning existed. `donor_binning=`/`mediator_binning=`/
`receiver_binning=` (each with its own `_num_shells=`/`_shell_edges=`)
override just that one axis, e.g. to pin donor and receiver at two narrow
custom bands while mediator sweeps a fine `log` binning:
```
donor_binning = custom
donor_shell_edges = 18,28
mediator_binning = log
mediator_num_shells = 20
receiver_binning = custom
receiver_shell_edges = 8,12
```

**Adding a new term is a source change, not a runtime registration**: add an
entry to the fixed tables in `src/registry.cpp` (`BuiltinQuantities()` for a
new derived quantity, `BuiltinTerms()` to combine two of them into a named
term), following the pattern of an existing entry.

## Building

There are two distinct ways to build this, matching the two ways to use it.

### Offline / standalone (the `energy-transfer-offline` tool)

Build this repo on its own -- the vendored `external/parthenon` submodule is
used automatically since no other `Parthenon::parthenon` target exists yet:

```
git submodule update --init --recursive
cmake -S . -B build -DPARTHENON_ENABLE_FFT=ON -DADIOS2_DIR=...
cmake --build build
./build/tools/energy_transfer_offline/energy-transfer-offline -i my_input.in
```

`-DPARTHENON_ENABLE_FFT=ON` is forwarded straight through to Parthenon's own
`add_subdirectory` (it's a normal Parthenon CMake option), and is required
per the prerequisite above. Pass through whatever else your Parthenon build
normally needs (Kokkos arch flags, `-GNinja`, etc.) the same way -- e.g.
mirroring an existing AthenaPK configure line:

```
cmake -S . -B build -GNinja \
  -DPARTHENON_ENABLE_FFT=ON -DKokkos_ENABLE_CUDA=OFF -DKokkos_ARCH_NATIVE=OFF \
  -DADIOS2_DIR=/path/to/ADIOS2/install/lib64/cmake/adios2
cmake --build build
```

`ADIOS2_DIR` (or an `ADIOS2_ROOT`/module-provided `adios2` package) is
needed because `energy_transfer` calls the raw ADIOS2 API directly for the
offline ingestion path. `openPMD_DIR` is normally **not** needed: Parthenon's
own CMakeLists already links `openPMD::openPMD` `PUBLIC` onto
`Parthenon::parthenon` whenever it's built via `add_subdirectory` (true here,
via the vendored submodule), so that target is reused automatically. It's
only required if you link against a separately *installed* Parthenon via
`find_package(parthenon)`, since `parthenonConfig.cmake` does not re-export
openPMD as a dependency in that case.

### In-situ, linked into AthenaPK (or any other Parthenon app)

The app's own build already provides `Parthenon::parthenon`, so this
library's own vendored submodule is simply never configured (per the guard
in "Getting the source" above) -- wiring this in is just: add this repo as a
subdirectory, link the `energy_transfer` target, then call the library from
your own code.

1. **Make this checkout visible to AthenaPK's CMake**, as a git submodule at
   `athenapk/external/energy_transfer` (matching the existing
   `external/parthenon`/`external/Kokkos` convention):

   ```
   git submodule add <url> external/energy_transfer
   ```

   (or point `add_subdirectory` at a plain local checkout path instead, if
   you don't want to vendor it as a submodule yet).

2. **Add that `add_subdirectory` call to `athenapk/CMakeLists.txt`**, right
   after the existing Parthenon block so `Parthenon::parthenon` already
   exists when this library's CMakeLists runs its `if(NOT TARGET ...)`
   guards, and before `add_subdirectory(src)`:

   ```cmake
   # athenapk/CMakeLists.txt, between the existing lines
   #   if(EXISTS .../external/parthenon/CMakeLists.txt)
   #     add_subdirectory(.../external/parthenon parthenon)
   #   else()
   #     find_package(parthenon REQUIRED)
   #   endif()
   # and
   #   add_subdirectory(src)
   set(ENERGY_TRANSFER_BUILD_TOOLS OFF CACHE BOOL "" FORCE)  # skip the offline tool -- AthenaPK doesn't need it
   set(ENERGY_TRANSFER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
   add_subdirectory(external/energy_transfer energy_transfer)  # or the direct path from step 1
   ```

3. **Link it into the `athenaPK` target**, in `athenapk/src/CMakeLists.txt`:

   ```cmake
   # was: target_link_libraries(athenaPK PRIVATE parthenon)
   target_link_libraries(athenaPK PRIVATE parthenon energy_transfer)
   ```

4. **Call it from your problem generator's `UserWorkBeforeOutput`** (or
   `UserMeshWorkBeforeOutput`/`UserWorkAfterLoop`/wherever you want it to
   run -- the library doesn't care about cadence, that's entirely up to
   you). `decaying_turbulence.cpp`'s `UserWorkBeforeOutput`
   (`athenapk/src/pgen/decaying_turbulence.cpp:130`) already has exactly the
   `Mesh*`/`ParameterInput*` this needs and already computes an FFT-based
   diagnostic (magnetic helicity) the same way -- add the energy-transfer
   call alongside it:

   ```cpp
   #include "energy_transfer/field_spec.hpp"
   #include "energy_transfer/io_openpmd.hpp"
   #include "energy_transfer/shell_transfer.hpp"

   void UserWorkBeforeOutput(Mesh *pmesh, ParameterInput *pin,
                             const parthenon::SimTime &tm) {
     // ... existing helicity calculation ...

     auto &md = pmesh->mesh_data.Get();
     // AthenaPK's "prim" layout: IDN=0, IV1=1, IV2=2, IV3=3, IPR=4, IB1=5, IB2=6, IB3=7
     // (from athenapk/src/main.hpp; pass your own enum values, this library
     // has no AthenaPK dependency of its own).
     auto spec = energy_transfer::MakeAthenaPKPrimitiveLiveSpec(
         IDN, IV1, IV2, IV3, IPR, /*has_bfield=*/true, IB1, IB2, IB3);

     energy_transfer::ShellTransferConfig cfg;
     cfg.donor_binning = cfg.mediator_binning = cfg.receiver_binning =
         energy_transfer::BinningSpec::Log(20);
     cfg.terms = {"UUA", "UUC"}; // mode defaults to Full() -- shared by both terms

     auto result = energy_transfer::ComputeShellTransferLive(pmesh, md.get(), spec, cfg);
     energy_transfer::WriteResult(result, "transfer", tm.ncycle);
   }
   ```

   Nothing here touches `ProcessPackages`/`StateDescriptor` -- the whole
   point of the live entry point is that it only needs the `Mesh*`/
   `MeshData<Real>*` a hook like this already has.
