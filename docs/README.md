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
   the library, which reads an ADIOS2/bp5 snapshot (e.g. converted from Enzo
   via `scripts/enzo_to_bp5.py`) and runs the same computation.

See `docs/plan.md` for the physics background and the historical single-file
prototype this library was extracted from
(`athenapk/external/parthenon/example/energy_transfer/`, left untouched as a
reference for verifying this refactor's numerical output).

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
  - `ShellTransferConfig` -- `binning` (`BinningSpec::Linear/Log/Custom`)
    and `terms` (a list of `TermRequest{name, DecompositionMode}`, names
    selected from the library's fixed built-in set -- see below) and
    `spectrum_names`.
  - `ComputeShellTransferLive(Mesh*, MeshData<Real>*, LiveFieldSpec, ShellTransferConfig)`
  - `ComputeShellTransferFromFile(Mesh*, input_file, ADIOS2FieldNaming, ShellTransferConfig)`
  - `ComputeFieldRequirements(cfg)` -- tells you which of magnetic field /
    pressure-or-energy / acceleration the requested terms actually need, so
    you can build a minimal `LiveFieldSpec`/`ADIOS2FieldNaming` (the offline
    tool does this automatically).
- `io_openpmd.hpp` -- `WriteResult(...)` writes every computed term/spectrum
  as a named openPMD mesh record to a `.bp` file.

### Built-in terms

`UUA, UUC, BBA, BBC, BUT, UBTb, UBTbA, UBTbC, BUPbb, UBPbb, PU, FU` (see
`docs/plan.md` for their physical meaning) plus spectra `spec_U, spec_rho,
spec_W, spec_B`. Each term can be requested at one of four resolutions
(`DecompositionMode::Full/BySender/ByReceiver/Total`) -- collapsing a side
skips that side's per-shell loop entirely rather than summing a full matrix
after the fact, so `Total` is O(1) in the number of shells, not O(n_shells^2).

**Adding a new term is a source change, not a runtime registration**: add an
entry to the fixed tables in `src/registry.cpp` (`BuiltinQuantities()` for a
new derived quantity, `BuiltinTerms()` to combine two of them into a named
term), following the pattern of an existing entry.

## Building

There are two distinct ways to build this, matching the two ways to use it.

### Offline / standalone (the `energy-transfer-offline` tool)

Build this repo on its own, pointed at a Parthenon checkout:

```
cmake -S . -B build \
  -DENERGY_TRANSFER_PARTHENON_SOURCE_DIR=/path/to/parthenon \
  -DPARTHENON_ENABLE_FFT=ON \
  -DADIOS2_DIR=...
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
  -DENERGY_TRANSFER_PARTHENON_SOURCE_DIR=/path/to/athenapk/external/parthenon \
  -DPARTHENON_ENABLE_FFT=ON -DKokkos_ENABLE_CUDA=OFF -DKokkos_ARCH_NATIVE=OFF \
  -DADIOS2_DIR=/path/to/ADIOS2/install/lib64/cmake/adios2
cmake --build build
```

`ADIOS2_DIR` (or an `ADIOS2_ROOT`/module-provided `adios2` package) is
needed because `energy_transfer` calls the raw ADIOS2 API directly for the
offline ingestion path. `openPMD_DIR` is normally **not** needed: whenever
Parthenon itself is built via `add_subdirectory` -- which
`ENERGY_TRANSFER_PARTHENON_SOURCE_DIR` does here -- Parthenon's own
CMakeLists already links `openPMD::openPMD` `PUBLIC` onto
`Parthenon::parthenon`, and that target is reused automatically. It's only
required if you link against a separately *installed* Parthenon via
`find_package(parthenon)`, since `parthenonConfig.cmake` does not re-export
openPMD as a dependency in that case.

### In-situ, linked into AthenaPK (or any other Parthenon app)

The app's own build already provides `Parthenon::parthenon`
(`ENERGY_TRANSFER_PARTHENON_SOURCE_DIR` is then not needed -- that target is
reused automatically), so wiring this in is just: add this repo as a
subdirectory, link the `energy_transfer` target, then call the library from
your own code.

1. **Make this checkout visible to AthenaPK's CMake.** Either point at it
   directly (fastest for local development, no extra setup):

   ```cmake
   # in athenapk/CMakeLists.txt
   add_subdirectory(/e/project1/jureap19/lenard/energy_transfer energy_transfer)
   ```

   or vendor it properly once you've pushed this repo somewhere, as a git
   submodule at `athenapk/external/energy_transfer` (matching the existing
   `external/parthenon`/`external/Kokkos` convention):

   ```
   git submodule add <url> external/energy_transfer
   ```

   and then `add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/energy_transfer energy_transfer)`.

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
     cfg.binning = energy_transfer::BinningSpec::Log(20);
     cfg.terms = {"UUA", "UUC",
                  {"BBA", energy_transfer::DecompositionMode::Total}};

     auto result = energy_transfer::ComputeShellTransferLive(pmesh, md.get(), spec, cfg);
     energy_transfer::WriteResult(result, "transfer", tm.ncycle);
   }
   ```

   Nothing here touches `ProcessPackages`/`StateDescriptor` -- the whole
   point of the live entry point is that it only needs the `Mesh*`/
   `MeshData<Real>*` a hook like this already has.
