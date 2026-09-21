# energy_transfer

(M)HD turbulence analysis library powered by Parthenon. 

## Features

Two families of analysis, both computed spectrally (FFT-based, exact
derivatives) on a uniform periodic grid and callable independently of each
other:

1. **Shell-to-shell transfer** of kinetic energy, magnetic energy and
   magnetic helicity, optionally resolving the mediating scale as well
   (triadic transfer).
2. **Spectra** of energy and helicity, including a decomposition of vector
   fields into compressive and left/right-handed (helical) solenoidal parts.

### Conventions

- Fields: `ρ` density, `U` velocity, `B` magnetic field (in units where the
  magnetic pressure is `B²/2`), `P` thermal pressure, `a` external
  acceleration (forcing). Two derived fields appear throughout: the
  density-weighted velocity `W = √ρ U` and the Alfvén-like field
  `b = B/√ρ`. `W` is the natural variable for compressible flows because the
  kinetic energy density is exactly `½|W|²` (whereas `½|U|²` is not the
  kinetic energy).
- Shells partition Fourier space by `|k|`, in units of the fundamental
  wavenumber `2π/L`; each shell is the half-open band `(k_low, k_high]`.
  `F_Q` denotes the field `F` with only the Fourier modes of shell `Q` kept.
- `⟨a, b⟩ = Σ a·b` is a plain sum over all grid cells (and MPI ranks) with
  no volume normalization: divide by the number of cells for a domain
  average, multiply by the cell volume for a domain integral.

### 1. Shell-to-shell transfer

The transfer function `T(K,Q)` is the rate at which the energy in the
**receiver** shell `K` changes because of nonlinear interaction with the
**donor** shell `Q` (positive: `K` gains from `Q`). It follows the
formulation of Grete et al. (2017, Phys. Plasmas 24, 092311,
[doi:10.1063/1.4990613](https://doi.org/10.1063/1.4990613)) for compressible
MHD, extended here by a third, optional axis (the **mediator**, see below).
The ideal equations for `W` and `B` are

```
∂ₜW = −(U·∇)W − ½W(∇·U)  +  (b·∇)B − ∇(B·B)/(2√ρ)  −  ∇P/√ρ  +  √ρ a
∂ₜB = −(U·∇)B −  B(∇·U)  +  ∇·(b W)                       [∇·(bW) = (B·∇)U]
```

and the energies of shell `K` evolve as `dE_u(K)/dt = ⟨W_K, ∂ₜW⟩` and
`dE_b(K)/dt = ⟨B_K, ∂ₜB⟩`. Splitting every nonlinear term on the right-hand
side by the shell `Q` its donor field lives in gives the built-in terms:

| Term | `T(K,Q)` | Process | Mediator | Extra fields |
|------|----------|---------|----------|--------------|
| `UUA`   | `−⟨W_K, (U·∇)W_Q⟩`                | kinetic → kinetic, advection            | `U`   | -   |
| `UUC`   | `−½⟨W_K·W_Q, ∇·U⟩`                | kinetic → kinetic, compression          | `∇·U` | -   |
| `BBA`   | `−⟨B_K, (U·∇)B_Q⟩`                | magnetic → magnetic, advection          | `U`   | `B` |
| `BBC`   | `−½⟨B_K·B_Q, ∇·U⟩`                | magnetic → magnetic, compression        | `∇·U` | `B` |
| `BUT`   | `+⟨W_K, (b·∇)B_Q⟩`                | magnetic → kinetic, tension             | `b`   | `B` |
| `UBTb`  | `+⟨B_K, ∇·(b W_Q)⟩`               | kinetic → magnetic, tension             | `b`   | `B` |
| `UBTbA` | `+⟨B_K, (b·∇)W_Q⟩`                | part of `UBTb` (derivative of `W`)      | `b`   | `B` |
| `UBTbC` | `+⟨B_K, W_Q ∇·b⟩`                 | part of `UBTb` (divergence of `b`)      | `∇·b` | `B` |
| `BUPbb` | `−½⟨W_K/√ρ, ∇(B·B_Q)⟩`            | magnetic → kinetic, magnetic pressure   | `B`   | `B` |
| `UBPbb` | `−⟨B_K·B, ∇·(W_Q/(2√ρ))⟩`         | kinetic → magnetic, compression against `B` (magnetic pressure) | `B` | `B` |
| `PU`    | `−⟨W_K/√ρ, ∇P_Q⟩`                 | thermal → kinetic, pressure gradient    | -     | `P` |
| `FU`    | `+⟨W_K, √ρ a_Q⟩`                  | external forcing → kinetic              | -     | `a` |
| `H`     | `+2⟨B_K, U×B_Q⟩`                  | magnetic helicity transfer              | `U`   | `B` |

`ρ` and `U` are always required. Names read `<donor><receiver><mechanism>`
(`U` kinetic, `B` magnetic; `A` advection, `C` compression, `T` tension,
`P` pressure); the lowercase suffixes follow the upstream Python reference
tool.

**What each group does and what it conserves.** Summed over all donor shells
the computed terms give the ideal-MHD rate of change of each shell's energy
(up to donor modes outside the binning range, such as the `k=0` mean):

```
dE_u(K)/dt = Σ_Q [ UUA + UUC + BUT + BUPbb + PU + FU ](K,Q)
dE_b(K)/dt = Σ_Q [ BBA + BBC + UBTb + UBPbb ](K,Q)
```

- **Kinetic and magnetic cascade (`UUA`+`UUC`, `BBA`+`BBC`).** Advection
  carries a field's energy from scale to scale; compression (`∇·U ≠ 0`)
  adds the part that has no incompressible counterpart. Only the *sums* are
  antisymmetric, `T(K,Q) = −T(Q,K)`, and sum to zero over all shells: they
  redistribute energy between scales without creating or destroying it. Neither
  term is antisymmetric alone -- the factor `½` on the compressive term is
  exactly what makes the sum so. (`docs/analytic_synthetic_field_derivation.md`
  works this out by hand for a synthetic field.) In the induction equation
  the compression term `−B∇·U` is split evenly: half is `BBC` (magnetic
  donor), the other half is `UBPbb` (velocity donor, see below).
- **Exchange between the reservoirs (`BUT`/`UBTb`, `BUPbb`/`UBPbb`).** The
  Lorentz force `(B·∇)B − ∇(B²/2)` moves energy from magnetic to kinetic
  through *tension* (field-line curvature, `BUT`) and *magnetic pressure*
  (`BUPbb`); the induction equation returns it through stretching of field
  lines by velocity gradients along the field (`UBTb`) and through
  compression against `B` (`UBPbb`). Each pair is the same physical
  exchange seen from either end: `BUT(K,Q) = −UBTb(Q,K)` and
  `BUPbb(K,Q) = −UBPbb(Q,K)` exactly, so the kinetic energy that shell `K` gains from
  magnetic shell `Q` is precisely what `Q` loses. `UBTb` is split by the
  product rule into `UBTbA` (derivative of `W` along `b`) and `UBTbC`
  (`W ∇·b`), with `UBTbA + UBTbC = UBTb`. Because
  `∇·b = −B·∇ρ / (2ρ^{3/2})`, `UBTbC` vanishes unless the density varies
  along the field, so it is the part of the tension exchange that is specific
  to compressible flow.
- **Sources (`PU`, `FU`).** `PU` is the kinetic energy exchanged with
  the thermal reservoir through the pressure-gradient force; `FU` is the
  energy injected by the forcing acceleration `a`. Neither has a counterpart
  term, since no internal-energy budget is computed. Viscous and resistive
  dissipation are not included either.
- **Magnetic helicity (`H`).** The helicity of shell `K` is
  `⟨A_K, B_K⟩` with `A` the Coulomb-gauge vector potential. Ideal induction
  gives `∂ₜA = U×B − ∇φ`; the gauge term `∇φ` drops out against the
  solenoidal `B_K`, and since `⟨A_K, ∂ₜB_K⟩ = ⟨∂ₜA_K, B_K⟩` the shell
  helicity changes at `2⟨B_K, U×B⟩`. Splitting `B` into donor shells gives
  `H(K,Q)`. It is antisymmetric and sums to zero, reflecting ideal
  conservation of total helicity, so it only redistributes helicity between
  scales (e.g. an inverse transfer shows up as `H(K,Q) > 0` for `K < Q`).
  Formulation as in Teissier & Müller (2021, J. Fluid Mech. 921,
  [doi:10.1017/jfm.2021.496](https://doi.org/10.1017/jfm.2021.496)). Where
  the helicity resides is given by `spec_helicity` below; `H` shows how it moves.

**Resolving donor, mediator and receiver.** Every term is a triple
product: a receiver field, a donor field, and a third mediating field that
couples them (e.g. the advecting `U` in `UUA`, or the tension field `b` in
`BUT`). By default the donor and receiver are resolved into shells and the
mediator is left whole, which gives the classical two-shell transfer
`T(K,Q)`. Resolving the mediator too gives the triadic transfer
`T(K,M,Q)`: how much of the transfer from `Q` to `K` is carried by the
mediating field at scale `M`, for instance to separate transfer by
large-scale flows from transfer by small-scale ones. Each axis has its own
binning (linear, logarithmic or custom edges) and can be resolved
independently, so e.g. donor and receiver can be pinned to two narrow bands
while the mediator sweeps all scales. For the terms that are linear in the
mediator, summing over the mediator shells reproduces the unresolved result;
`PU` and `FU` have no mediating field (only a density normalization) and
cannot be mediator-resolved. Note that none of the built-in binnings
contains the mean (`k=0`) mode: an unresolved axis uses the full field
including its mean, a resolved one does not, so transfer mediated by a
uniform background field (e.g. a mean magnetic field acting through `b·∇`)
is only captured with the mediator left unresolved. See
"Built-in terms" below for how to select the axes and binnings.

### 2. Spectra

**Power spectra** (`spec_U`, `spec_rho`, `spec_W`, `spec_B`) are binned in
unit-width shells of `|k|` and returned as three columns per spectrum:
`pow_sum` (sum of `|f̂|²` over the modes in the bin), `k_sum` and
`count_sum` (so the mean wavenumber of a bin is `k_sum/count_sum`). The Fourier
transform is normalized so that `Σ_bins pow_sum = ⟨|f|²⟩`, the
volume average of the squared field. These are raw squared amplitudes, not energies:
the magnetic energy spectrum is `½ spec_B`, and the kinetic energy spectrum of a
compressible flow is `½ spec_W` (density-weighted), not `½ spec_U`.

**Compressive and helical decomposition** (`spec_U_decomp`,
`spec_W_decomp`, `spec_B_decomp`). Each Fourier mode of a vector field is
projected onto the orthonormal complex basis `{k̂, h₊, h₋}` with
`h± = (e₁ ± i e₂)/√2` and `e₁, e₂ ⟂ k̂`:

- **compressive**: the part parallel to `k` (curl-free, longitudinal);
- **plus / minus**: the two circularly polarized parts transverse to `k`
  (divergence-free, helical: `∇×h± = ±|k| h±`), carrying positive and
  negative helicity respectively.

A bundle returns four spectra from one shared forward FFT: the full
spectrum and its `_compressive`, `_plus` and `_minus` parts. At every bin
except `k=0`, the three parts sum to the full spectrum exactly
(orthonormality of the basis); at `k=0` the direction is undefined, so all
three are zero there and only the full spectrum carries the mean. Typical uses:

- `spec_U_compressive` against `spec_U_plus + spec_U_minus` (or the `W`
  equivalent) is the scale-dependent compressive-to-solenoidal ratio of the flow.
- `spec_B_compressive` is zero for an exactly solenoidal field, so for
  simulation data it measures the spectral `∇·B` error at each scale.
- `spec_*_plus` against `spec_*_minus` quantifies the helicity imbalance
  of a field; for `U`, `k (spec_U_plus − spec_U_minus)` is the kinetic
  helicity spectrum.

Decomposing the transfer terms themselves into these components is not
implemented yet.

**Helicity spectra** (both need `B`, implemented in `helicity.hpp`):

- `spec_helicity` is the *signed* magnetic helicity spectrum `Re(Â·B̂*)`,
  with the Coulomb-gauge potential `Â = i(k×B̂)/|k|²` (zero at `k=0`). It is
  a co-spectrum of two different fields, not a power spectrum, so its
  bins can be negative, and it sums to the volume-averaged helicity
  `⟨A·B⟩`. Each mode contributes `(|b₊|² − |b₋|²)/|k|`, where `b±` are its
  helical amplitudes, which ties it directly to the decomposition above and
  bounds it by `|Re(Â·B̂*)| ≤ |B̂|²/|k|` per mode: the relative helicity
  `k·spec_helicity/spec_B` lies in `[−1, 1]` (up to binning effects).
- `spec_helicity_variance` is the ordinary power spectrum of the
  real-space local helicity density `h(x) = A(x)·B(x)`. It measures the
  scales on which the helicity density fluctuates in space, a different
  quantity from `spec_helicity`, which measures how much of the *total*
  helicity sits at each `k`.

`CalcHelicity(pm, B)` returns `h(x)` itself and is usable outside the
spectra machinery, e.g. to write the field back into an application's mesh.


## Usage

1. **On the fly**, linked into any Parthenon application: call
   `energy_transfer::GatherLiveFields(...)` from your own hook (e.g.
   `UserWorkBeforeOutput`) with the `Mesh*`/`MeshData<Real>*` you already
   have, then `energy_transfer::ComputeEnergyTransfer(...)` and/or
   `energy_transfer::ComputeSpectra(...)` (two independent, separately
   callable computations -- call either, both, or neither, on whatever
   cadence you like), then `energy_transfer::WriteResult(...)` to dump the
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
`ComputeEnergyTransfer` and throw a clear error otherwise.

## Public API (see `include/energy_transfer/`)

- `field_spec.hpp` -- `LiveFieldSpec` describes where your app keeps
  density/velocity-or-momentum/magnetic-field/pressure-or-energy/
  acceleration, as a variable name + component index (or indices). Works
  for both separate named fields and a single packed field addressed by
  component (e.g. AthenaPK's `"cons"`/`"prim"`). Use `MakeAthenaPKPrimitiveLiveSpec`/
  `MakeAthenaPKConservedLiveSpec`, passing your own `IDN`/`IV1`/... enum
  values, or `MakeSeparateFieldsLiveSpec` for the historical rho/vel/mag/
  acc/pres layout.
- `shell_transfer.hpp` -- shell-to-shell energy transfer, entirely
  independent of spectra (see `spectra.hpp` below -- the two are separate
  concerns with separate entry points, on purpose: an in-situ caller can
  compute either, both, or neither, on whatever cadence it wants, and the
  offline driver calls them one after the other rather than through one
  fused call):
  - `ShellTransferConfig` -- `donor_binning`/`mediator_binning`/
    `receiver_binning` (each an independent `BinningSpec::Linear/Log/Custom`
    -- e.g. a narrow custom band pinning donor and receiver to two specific
    scales while mediator sweeps a fine `Log()` binning across whichever
    scales mediate that pair's transfer), `terms` (a plain list of names
    selected from the library's fixed built-in set -- see below), and `mode`
    (one `DecompositionMode` shared by every term in `terms` -- not
    per-term, see below).
  - `ComputeEnergyTransfer(Mesh*, FlatFields&, ShellTransferConfig)` --
    fields must already be ingested and in primitive form (`ingest.hpp`'s
    `GatherLiveFields`/`ReadADIOS2Fields`/`ReadPHDFFields` +
    `convert.hpp`'s `ConvertConservedToPrimitive`). Populates only
    `TransferResult::matrices`.
  - `ComputeFieldRequirements(cfg, spectrum_names)` -- tells you which of
    magnetic field / pressure-or-energy / acceleration the requested terms
    *and* spectra actually need, so you can build a minimal
    `LiveFieldSpec`/`FileFieldNaming` (the offline tool does this
    automatically).
- `spectra.hpp` -- power spectra, independent of shell-to-shell transfer:
  - `ComputeSpectra(Mesh*, const FlatFields&, spectrum_names)` -- same
    ingested/primitive `FlatFields` as `ComputeEnergyTransfer` above; returns
    a plain `map<string, HostArray2D<TransferReal>>` (a caller wanting both
    fills it into `TransferResult::spectra` itself, e.g. before calling
    `WriteResult`).
- `io_openpmd.hpp` -- `WriteResult(...)` writes every computed term/spectrum
  as a named openPMD mesh record to a `.bp` file.

### Built-in terms

`UUA, UUC, BBA, BBC, BUT, UBTb, UBTbA, UBTbC, BUPbb, UBPbb, PU, FU, H` (see
"Features" above for their physical meaning) plus spectra `spec_U, spec_rho,
spec_W, spec_B`, and a directionally-decomposed variant of each vector field
requested via a single bundle name -- `spec_U_decomp`, `spec_W_decomp`,
`spec_B_decomp` (`rho` is scalar, no direction to decompose against; `B`'s
bundle pulls in the magnetic field the same way plain `spec_B` does).
Requesting e.g. `spec_B_decomp` computes and writes **four** spectra
together -- `spec_B`, `spec_B_compressive`, `spec_B_plus`, `spec_B_minus`
(compressive = parallel to the wavevector `k`; plus/minus = the two
circularly-polarized "helical" parts perpendicular to `k`) -- sharing a
single `Forward()` FFT across all four (see
`include/energy_transfer/decomposition.hpp`'s `DecomposeFourierField`, a
pure per-mode Fourier-space decomposition with no notion of "spectrum" at
all, and `include/energy_transfer/spectral_kernels.hpp`'s
`BinFourierSpectrum`, a generic already-in-Fourier-space binning utility
applied separately to the full field and to each decomposed component),
since there's rarely a reason to want just one direction in isolation, and
computing them separately would redundantly re-FFT the same field three
times over. `decomposition.hpp` also explains why all three components --
plus/minus included -- are Hermitian-symmetric and could validly be
reconstructed to real space with this library's existing r2c/c2r FFT if a
future extension needs that; the current spectrum-only use never leaves
Fourier space simply because a power spectrum never needs to, not as a
workaround for anything. At every bin **except `k=0`**,
`spec_B_compressive + spec_B_plus + spec_B_minus` reconstructs `spec_B`
exactly (Parseval/orthonormality of the projection basis); at the DC mode
(`k=0`) direction is undefined, so all three are exactly zero there instead
-- no convention routes that mode's power to any of them, it's simply not
decomposed (only `spec_B` itself carries the true DC power). Input decks
request a bundle the same way as any other spectrum:
`spectra = spec_U,spec_B_decomp` (see
`tools/energy_transfer_offline/parthinput.example`) -- the individual
`spec_B_compressive` etc. names are not separately requestable.

Two more spectra, both needing the magnetic field and both implemented in
`include/energy_transfer/helicity.hpp`/`src/helicity.cpp` (a small,
standalone module -- `CalcHelicity(pm, B)` is usable on its own outside the
spectra machinery, e.g. by an in-situ app wanting the raw real-space field
to `ScatterField` back into its own mesh, the way
`athenapk/src/pgen/decaying_turbulence.cpp` computes it inline today):
- `spec_helicity` -- the *signed* magnetic helicity spectrum,
  `Re(Â(k)·B̂*(k))` binned by `|k|`, where `Â` is the Coulomb-gauge vector
  potential reconstructed spectrally from `B̂` (`Â = i(k×B̂)/|k|²`, zero at
  `k=0`). This is a **co-spectrum** of two different fields (`Â` and `B̂`),
  not an ordinary power spectrum -- its bins can be negative, and by
  Parseval it sums to the real-space total helicity `∫A·B dV`. Computed via
  `include/energy_transfer/spectral_kernels.hpp`'s `BinFourierCospectrum`, a
  generic already-in-Fourier-space co-spectrum binning utility of which
  `BinFourierSpectrum` (used for every other spectrum in this library) is
  just the special case of a field against itself.
- `spec_helicity_variance` -- the ordinary power spectrum of the real-space
  scalar field `H(x) = A(x)·B(x)` itself (`CalcSpectrum(H, 1)`) -- measures
  the spatial fluctuation scale of local helicity density, a genuinely
  different quantity from `spec_helicity` above (which measures how much of
  the *total* helicity resides at each `k`).

`DecompositionMode{donor_resolved, mediator_resolved, receiver_resolved}` (a
plain struct of three bools -- just pick which axes are decomposed; the
default is `{true, false, true}`) controls which of the three shell axes are
resolved per-shell vs collapsed to the whole domain -- collapsing an axis
skips that axis's per-shell loop entirely rather than summing a full matrix
after the fact, so decomposing nothing (`{false, false, false}`, one global
number per term) is O(1) in the number of shells, not O(n_shells^2).
**`mode` applies to
every term in `ShellTransferConfig::terms` -- it is not per-term.** This is
deliberate: it lets `ComputeEnergyTransfer` share one `(Q,M,K)` shell sweep
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
the tension field `b` in `BUT`). All three axes are on equal footing, in the
API and in the implementation: each is independently decomposed or not, and
"not decomposed" means exactly the same thing on each of them -- the field
passes through completely unfiltered (`k=0` included) at zero extra FFT
cost, rather than being filtered against a whole-domain band. Internally
that's one `ShellRestriction{active, low, high}` per axis, fed to one shared
`FilterVector`/`FilterScalar` pair (`src/registry.cpp`); no axis has a
special code path. Not
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
suffix), e.g. `terms=UUA,BBA,BUT`; `mode=` lists which axes are decomposed --
any comma-separated subset of `donor`, `mediator`, `receiver`, in any order.
`mode = donor,receiver` is the default; `mode = donor,mediator,receiver`
decomposes all three; an empty `mode =` decomposes none, giving one global
number per term.

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
   #include "energy_transfer/convert.hpp"
   #include "energy_transfer/field_spec.hpp"
   #include "energy_transfer/ingest.hpp"
   #include "energy_transfer/io_openpmd.hpp"
   #include "energy_transfer/shell_transfer.hpp"
   #include "energy_transfer/spectra.hpp"

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
     cfg.terms = {"UUA", "UUC"};
     cfg.mode = {/*donor=*/true, /*mediator=*/false, /*receiver=*/true}; // the default; shared by both terms

     // GatherLiveFields is a cheap, in-memory reshuffle of fields already
     // resident in md -- no file I/O, unlike the offline driver's ingestion.
     auto fields = energy_transfer::GatherLiveFields(pmesh, md.get(), spec);
     energy_transfer::ConvertConservedToPrimitive(fields);

     auto result = energy_transfer::ComputeEnergyTransfer(pmesh, fields, cfg);
     // Independent call -- compute spectra every cycle, energy transfer only
     // some cycles, or vice versa, entirely up to you:
     result.spectra = energy_transfer::ComputeSpectra(pmesh, fields, {"spec_U"});
     energy_transfer::WriteResult(result, "transfer", tm.ncycle);
   }
   ```

   Nothing here touches `ProcessPackages`/`StateDescriptor` -- the whole
   point of the live entry point is that it only needs the `Mesh*`/
   `MeshData<Real>*` a hook like this already has.
