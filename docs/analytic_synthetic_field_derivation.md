# Analytic derivation: `UUA`/`UUC` for a synthetic two-mode field

## Purpose

The gold-standard regression test (`test/regression/test_suites/gold_standard_dd0024/`) validates against an external finite-difference reference, which means it can never be exact -- the residual is a real, expected FD-vs-spectral gap (see that test's `README.md`). This document takes the opposite approach: pick a synthetic field simple enough to shell-filter and differentiate **by hand**, and derive the exact numbers `ComputeShellTransfer` must produce for it. Since everything here is expressed as finite sums of exact discrete Fourier harmonics, the spectral derivatives used by the code are *exact* for this input (no truncation error at all), so the derived values are a legitimate machine-precision assertion for a unit test -- no external data, no tolerance tuning, no dependence on the FD-vs-spectral question at all.

## Code conventions this derivation relies on

- Grid: `field(idx) = ...` at `x_m = m L/N`, `m = 0..N-1`, matching `test_shell_filter_synthetic.cpp:45`'s own convention (`cos(2*pi*kx_mode*i/Nx)`). `kappa := two_pi_over_L = 2*pi/L` (`shell_transfer.cpp:130`).
- `ShellFilter` keeps a Fourier mode iff `k_low < |k| <= k_high` (`spectral_kernels.cpp:27`) -- half-open, upper-inclusive.
- `ShellFilterDerivative` computes `i * kappa * n_dir * FT_field` for modes inside the shell, i.e. the *exact* continuum derivative of whatever discrete Fourier content survives the filter (`spectral_kernels.cpp:67-69`).
- `DotProductReduce` is a plain grid-point sum, `sum_idx a(idx)*b(idx)`, no `1/N` or volume-element factor (`registry.cpp:385-400`).
- Term definitions and prefactors, from `registry.cpp:359-362`:
  - `UUA = -1 * <W_filter_K, U_dot_grad_W_Q>`, where `U_dot_grad_W_Q = (U . grad) W_Q` (full, unfiltered `U`; only `W` is shell-restricted to `Q` -- see `UdotGradW`, `registry.cpp:57-77`).
  - `UUC = -0.5 * <W_filter_K, W_times_DivU_Q>`, where `W_times_DivU_Q = W_Q * div(U)` (pointwise product; `div(U)` is the full, *unfiltered* divergence -- see `WTimesDivU`, `registry.cpp:145-156`).
- Assume `rho == 1` everywhere, so `W == U` exactly (`W = sqrt(rho)*vel`, `shell_transfer.cpp:163-174`) -- this is what lets the derivation work with a single field instead of tracking `rho` and `vel` separately.

## Discrete orthogonality lemma

For grid points `x_m = m L/N` and integer mode number `p` (not a multiple of `N`, i.e. not aliased to the zero/Nyquist mode):

```         
sum_{m=0}^{N-1} cos(2*pi*p*m/N) = 0
sum_{m=0}^{N-1} sin(2*pi*p*m/N) = 0
sum_{m=0}^{N-1} cos^2(2*pi*p*m/N) = sum_{m=0}^{N-1} sin^2(2*pi*p*m/N) = N/2
```

and for `p != q` (both nonzero, not aliased), any product `sin(p.)cos(q.)`, `sin(p.)sin(q.)`, `cos(p.)cos(q.)` with `p != q` (mod `N`) sums to exactly zero, via `sum_m exp(2*pi*i*p*m/N) = N` iff `N | p`, else `0`. All the wavenumbers used below (`k0=1, 2k0=2, 3k0=3, 4k0=4`) stay far from `N/2` for any `N >= 16`, so no aliasing subtlety arises.

## The synthetic field

Single spatial dependence (x only), single nonzero velocity component, `rho == 1` everywhere:

```         
U_x(x) = U0 * cos(theta) + U1 * sin(2*theta),   theta := kappa * k0 * x
U_y = U_z = 0
```

This field is **compressible** (`div(U) = -U0*kappa*k0*sin(theta) + 2*U1*kappa*k0*cos(2*theta) != 0`) -- deliberately, so the derivation exercises both `UUA` and `UUC`, not just the incompressible case.

Shells: pick edges `[0.5, 1.5, 2.5, 3.5]`, so shell 1 = `(0.5,1.5]` contains exactly `k0=1`, shell 2 = `(1.5,2.5]` contains exactly `2k0=2`, shell 3 = `(2.5,3.5]` contains exactly `3k0=3` (present only as *generated* content, see below -- the base field has no seed there).

## Step 1: shell-filtered `W_Q` (== `U_Q`, since `rho=1`)

Since `U_x` is exactly two harmonics, filtering is trivial:

```         
W^{Q=1}_x(x) = U0*cos(theta)          (shell 1 keeps only the k0 term)
W^{Q=2}_x(x) = U1*sin(2*theta)        (shell 2 keeps only the 2k0 term)
W^{Q=3}_x(x) = 0                      (no seed content at 3k0)
```

## Step 2: `U . grad(W_Q)` (for `UUA`)

Only the x-derivative of `W_Q` is nonzero (no y/z dependence), so `(U.grad W_Q)_x = U_x(x) * d(W_Q_x)/dx`; other components are 0.

**Q=1**: `d(W^{Q=1}_x)/dx = -U0*kappa*k0*sin(theta)`. Using `cos(a)sin(a) = sin(2a)/2` and `sin(2a)cos(a) = [sin(3a)-sin(a)]/2`:

```         
G^{Q=1}_x(x) = U_x * d(W^{Q=1}_x)/dx
             = -(U0^2*kappa*k0/2)*sin(2*theta)
               - (U0*U1*kappa*k0/2)*cos(theta)
               + (U0*U1*kappa*k0/2)*cos(3*theta)
```

**Q=2**: `d(W^{Q=2}_x)/dx = 2*U1*kappa*k0*cos(2*theta)`. Using `cos(a)cos(2a) = [cos(a)+cos(3a)]/2` and `sin(2a)cos(2a) = sin(4a)/2`:

```         
G^{Q=2}_x(x) = U0*U1*kappa*k0*cos(theta)
             + U0*U1*kappa*k0*cos(3*theta)
             + U1^2*kappa*k0*sin(4*theta)
```

**Q=3**: `W^{Q=3} = 0` identically, so `G^{Q=3} = 0` (no computation needed).

## Step 3: `UUA(Q,K) = -sum_x[ W_K(x) * G^Q(x) ]`

Every cross term `sin(p.)cos(q.)` or matching-frequency `sin.cos` product above integrates to zero by the lemma; only matching `sin.sin` / `cos.cos` (same `p`) survive, contributing `N/2` each. Working through all four populated `(Q,K)` combinations:

| Q\K     | K=1 (`k0`)              | K=2 (`2k0`)             | K=3 (`3k0`) |
|---------|-------------------------|-------------------------|-------------|
| **Q=1** | `+U0^2*U1*kappa*k0*N/4` | `+U0^2*U1*kappa*k0*N/4` | `0`         |
| **Q=2** | `-U0^2*U1*kappa*k0*N/2` | `0`                     | `0`         |
| **Q=3** | `0`                     | `0`                     | `0`         |

(Row/column 3 is zero purely structurally: `W_filter_K` is a linear filter of the base field, so if the base field has no power in shell K, `W_filter_K` is identically zero and *every* term using it as the `K`-side quantity vanishes for *any* `Q` -- no derivative or cancellation argument needed. This is the cheapest possible "first principles" check: it needs no arithmetic at all, just "shell with no seed content -\> that row/column of every term's matrix is exactly zero.")

## Step 4: `UUC(Q,K) = -0.5 * sum_x[ W_K(x) * W_Q(x) * div(U)(x) ]`

`div(U)(x) = -U0*kappa*k0*sin(theta) + 2*U1*kappa*k0*cos(2*theta)` (full, unfiltered -- same for every `Q`). Forming `W_Q * div(U)` and reducing with the same identities:

```         
P^{Q=1}_x = -(U0^2*kappa*k0/2)*sin(2*theta) + U0*U1*kappa*k0*cos(theta) + U0*U1*kappa*k0*cos(3*theta)
P^{Q=2}_x = -(U0*U1*kappa*k0/2)*cos(theta) + (U0*U1*kappa*k0/2)*cos(3*theta) + U1^2*kappa*k0*sin(4*theta)
```

| Q\K     | K=1                     | K=2                     | K=3 |
|---------|-------------------------|-------------------------|-----|
| **Q=1** | `-U0^2*U1*kappa*k0*N/4` | `+U0^2*U1*kappa*k0*N/8` | `0` |
| **Q=2** | `+U0^2*U1*kappa*k0*N/8` | `0`                     | `0` |
| **Q=3** | `0`                     | `0`                     | `0` |

## Bonus cross-check: `UUA + UUC` is exactly antisymmetric and sums to zero

Adding the two matrices entrywise:

```         
(Q=1,K=1): U0^2*U1*kappa*k0*N * ( 1/4 - 1/4)        = 0
(Q=1,K=2): U0^2*U1*kappa*k0*N * ( 1/4 + 1/8)        = +3/8 * U0^2*U1*kappa*k0*N
(Q=2,K=1): U0^2*U1*kappa*k0*N * (-1/2 + 1/8)        = -3/8 * U0^2*U1*kappa*k0*N
(Q=2,K=2): 0
```

`(UUA+UUC)(1,2) = -(UUA+UUC)(2,1)` exactly, and the full sum over the 3x3 block is exactly zero -- the textbook "shell-to-shell transfer conserves energy" identity for the *combined* advective+compressive nonlinear self-interaction, reproduced here from a hand-derived example rather than assumed. (Neither `UUA` nor `UUC` individually is antisymmetric or zero-sum on its own for this compressible field -- that only emerges once they're added back together, which makes sense: the split into "advective" vs "compressive" pieces is bookkeeping internal to the method, only their sum reconstitutes the actual physical nonlinear term.)

## Concrete numbers for a unit test

Pick `L = 2*pi` (so `kappa = 1`), `k0 = 1`, `N = 64` (any `N >= 16` avoids aliasing for the `1,2,3,4` harmonics used here), `U0 = U1 = 1`, `rho = 1` everywhere, shells `= [0.5, 1.5, 2.5, 3.5]`:

```         
U0^2 * U1 * kappa * k0 * N / 8  =  8      (base unit for the table below)

UUA: [[ 16,  16, 0],
      [-32,   0, 0],
      [  0,   0, 0]]

UUC: [[-16,   8, 0],
      [  8,   0, 0],
      [  0,   0, 0]]
```

## Suggested test structure

Follow `test/test_shell_filter_synthetic.cpp`'s pattern (single-rank, `ParthenonManager` bring-up, no MPI needed): fill `FlatFields::rho` with 1.0 everywhere, `FlatFields::mom_or_vel`'s x-component with `U0*cos(kappa*k0*x_m) + U1*sin(2*kappa*k0*x_m)` (y/z components zero), leave `mag`/`pres_or_energy`/`acc` unset (not needed -- `ComputeFieldRequirements` for `terms=UUA,UUC` doesn't request them), call `ComputeShellTransfer` with `binning = Custom({0.5,1.5,2.5,3.5})`, and `EXPECT_NEAR` each of the 18 matrix entries above against machine precision (`~1e-10`, not the `1e-7` used against the FD gold data -- there is no discretization error to budget for here, only floating-point round-off and the FFT's own accuracy).