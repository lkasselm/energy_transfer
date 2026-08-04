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

```
cmake -S . -B build \
  -DENERGY_TRANSFER_PARTHENON_SOURCE_DIR=/path/to/parthenon \
  -DADIOS2_DIR=... -DopenPMD_DIR=...
cmake --build build
```

If built as `add_subdirectory()` from an application that already defines a
`Parthenon::parthenon` target (e.g. AthenaPK, right after its own
`add_subdirectory(external/parthenon)`), that target is reused automatically
and `ENERGY_TRANSFER_PARTHENON_SOURCE_DIR` is not needed.
