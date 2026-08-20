# gold_standard_dd0024

Validates `energy_transfer`'s C++ output against the upstream Python reference tool's
own checked-in test fixture:
[github.com/pgrete/energy-transfer-analysis](https://github.com/pgrete/energy-transfer-analysis),
`testing/0024-All-Forc-Pres-testing-128_HDF-gold.pkl` (checked into this directory as
`gold_standard_0024.pkl`).

## What it validates

All 12 of this library's built-in transfer terms (`UUA, UUC, BBA, BBC, BUT, UBTb,
UBTbA, UBTbC, BUPbb, UBPbb, PU, FU`) against the gold pickle's identically-named
entries, using the exact same 6-shell binning (`[0.5, 1.5, 2.5, 16.0, 26.5, 28.5,
32.0]`) and `gamma=1.0001` the gold data was produced with. The upstream pickle has 9
additional terms (`UU, BB, UBTbTot, UBTa, BUP, UBPb, UBPbA, UBPbC, UBPbTot`) this
library doesn't implement -- `UU`/`BB` (convenience sums of terms this library does
have) are checked as a non-gating bonus; the rest are a genuinely different term
family and aren't checked at all.

## Data

The input snapshot is a driven-MHD-turbulence Enzo run (128^3, `DD0024`,
`run/MHD/3D/StochasticForcing`), **not checked into any repo** -- `fetch_and_convert.py`
downloads it from `https://pgrete.de/dl/EnTrans/EnTransTesting-DD0024.tgz` (370MB,
checksummed against the md5/sha1 given in the upstream `testing/README.md`), extracts
it, and converts `DD0024/data0024` to ADIOS2/bp5 via `../../../../scripts/enzo_to_bp5.py`.
Everything is cached under `$ENERGY_TRANSFER_DD0024_CACHE` (set from the CMake cache
variable `ENERGY_TRANSFER_GOLD_DATA_CACHE_DIR`, default
`<build_dir>/external_data/dd0024`) so this only happens once.

## Enabling

Off by default (`-DENERGY_TRANSFER_ENABLE_GOLD_STANDARD_TEST=OFF`), since it needs
that external download plus `yt`. Enable with:
```
cmake -S . -B build -DENERGY_TRANSFER_BUILD_TOOLS=ON -DENERGY_TRANSFER_ENABLE_GOLD_STANDARD_TEST=ON ...
cmake --build build
ctest --test-dir build -R gold_standard_dd0024 -V
```
If the dataset can't be fetched/converted (no network, missing `yt`, checksum
mismatch), `Prepare()` exits with code 77, which ctest reports as **skipped**, not
failed.

## On the derivative-method discrepancy

The upstream tool computes every gradient/divergence via 2nd-order central finite
differences (real space); this library uses exact spectral (FFT) derivatives -- a
deliberate design choice, not a bug. Real disagreement beyond floating-point noise is
therefore possible, and should grow with wavenumber (worst near the highest-k shell,
`28.50-32.00`, closest to the grid's Nyquist limit). The test currently uses the exact
same tolerance the upstream tool's own `testing/runTest.py` uses to compare two runs
of itself (`rtol=1e-14, atol=1e-7`) rather than a pre-loosened one. If this test fails
on a real run, check whether failures cluster at high-k shells (expected FD-vs-spectral
signature) versus looking uniformly wrong (would point to an actual bug) before
concluding the tolerance needs revisiting.

**In practice, the first real run surfaced a different, much larger effect first**: a
near-uniform `~1/(2*pi)` relative error across every derivative-based term (`FU`, the
one term with no derivative, was correctly unaffected) -- not the wavenumber-growing
signature described above. That turned out to be `dd0024.in`'s domain size being wrong
(`x1max` etc. were `2*pi`, copied from this repo's other example decks, when this
Enzo snapshot's actual domain is a unit box per `enzo_to_bp5.py`'s own reported
`Domain: [0.0, 1.0]^3`) -- `two_pi_over_L = 2*pi/Lx` feeds directly into every spectral
derivative, so a wrong `Lx` silently rescales every derivative-based term by a constant
factor. Fixed by matching `x1max`/`x2max`/`x3max` to `1.0`. Worth remembering if this
ever needs to run against a *different* snapshot: the mesh block's domain size must be
read off from that snapshot's own conversion output, not copied from another deck.
