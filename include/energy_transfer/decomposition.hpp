#ifndef ENERGY_TRANSFER_DECOMPOSITION_HPP_
#define ENERGY_TRANSFER_DECOMPOSITION_HPP_

#include <basic_types.hpp>
#include <kokkos_types.hpp>
#include <mesh/mesh.hpp>
#include <utils/fft_manager.hpp>

#include "energy_transfer/spectral_kernels.hpp"

namespace energy_transfer {

using parthenon::Real;

// Directional decomposition of a 3-component Fourier-space vector field:
// "compressive" (parallel to the wavevector k) and the two circularly
// polarized "helical" parts transverse to k (often called left/right-handed
// or plus/minus). Ported from the Python reference tool's
// DecompHelperFuncs.py::getHelicalDecomposition (which only covers the
// plus/minus split of an already-solenoidal field) plus the compressive part
// that split silently drops, following the same k-hat projection convention
// FlowAnalysis.py::vector_power_spectrum_from_fft uses for its
// compressive/solenoidal (Helmholtz) spectrum. Also matches
// athenapk/src/utils/generate_stochastic_b_field.cpp's helical-basis
// construction (ex1/ex2/ep/em) almost exactly -- that file performs the
// inverse operation (synthesizing a field with prescribed helicity) with the
// same basis, minus the random per-mode phase rotation it adds only for its
// own generation use case.
//
// Pure per-mode Fourier-space math: this file knows nothing about power
// spectra. It decomposes an already-Fourier-transformed field and returns
// the decomposed field, nothing more -- no FFT execution (the caller already
// forward-transformed the input) and no binning (see spectral_kernels.hpp's
// BinFourierSpectrum for that, applied to whichever of these outputs, or to
// the original input, a caller wants a spectrum of).
//
// Real-space reconstruction: both the compressive component AND the
// individual plus/minus helical components are Hermitian-symmetric
// (B(-k) = conj(B(k))), so all of them CAN validly be inverse-transformed to
// a real field with this library's existing r2c/c2r ShellFilter/Backward()
// machinery -- a future extension needs no new C2C transform infrastructure.
// This follows from a general fact: for ANY per-mode unit vector e(k) with
// e(-k) = s*conj(e(k)) for a fixed sign s (true here for k_hat, with s=-1,
// since it's real; and for h_plus, also with s=-1, from e1(-k)=-e1(k),
// e2(-k)=e2(k) below plus h_minus(k)=conj(h_plus(k))), the projection
// B(k) := <e(k),V(k)> * e(k) of a real field V satisfies B(-k)=conj(B(k))
// regardless of the value of s, since the two factors of s square to 1.
// (Verified both symbolically and against a direct numerical example.)
//
// DC mode (k=0): direction is genuinely undefined there, so
// BuildHelicalBasis returns an all-zero basis and DecomposeFourierField
// returns exactly zero for compressive/plus/minus at that one mode -- no
// artificial convention is applied to route the DC mode's power anywhere.
// A caller wanting the DC mode's actual power gets it from the original
// (undecomposed) input field, not from these outputs.

// k_hat (real) and h_plus/h_minus (complex), all transverse to k_hat except
// k_hat itself. At k=0 direction is undefined -- every field is left zero;
// callers must special-case k=0 themselves if they need to (see
// DecomposeFourierField's DC handling above) rather than relying on
// k_hat=0 producing the right answer through the ordinary projection
// formulas below.
struct HelicalBasis {
  Real khat[3] = {0.0, 0.0, 0.0};
  Kokkos::complex<Real> h_plus[3] = {};
  Kokkos::complex<Real> h_minus[3] = {};
};

// kx,ky,kz must already be in physical x,y,z order (e.g. via
// ComponentWavenumber(k_vec, 0/1/2)), not FFTManager::KernelHelper::
// Wavevector's own raw {kz,ky,kx} storage order.
KOKKOS_INLINE_FUNCTION
HelicalBasis BuildHelicalBasis(Real kx, Real ky, Real kz) {
  HelicalBasis basis;
  const Real kperp2 = kx * kx + ky * ky;
  const Real kmag = Kokkos::sqrt(kperp2 + kz * kz);
  if (kmag == Real(0.0)) return basis; // k=0: undefined direction, see above

  Real e1[3];
  if (kperp2 == Real(0.0)) {
    e1[0] = 1.0;
    e1[1] = 0.0;
    e1[2] = 0.0; // pure-kz fallback (matches the Python reference's mask branch)
  } else {
    const Real kperp = Kokkos::sqrt(kperp2);
    e1[0] = -ky / kperp;
    e1[1] = kx / kperp;
    e1[2] = 0.0;
  }

  const Real invk = Real(1.0) / kmag;
  const Real khat[3] = {kx * invk, ky * invk, kz * invk};
  // e2 = khat x e1, already a unit vector since khat and e1 are orthonormal.
  const Real e2[3] = {khat[1] * e1[2] - khat[2] * e1[1], khat[2] * e1[0] - khat[0] * e1[2],
                      khat[0] * e1[1] - khat[1] * e1[0]};

  const Real s = Real(1.0) / Kokkos::sqrt(Real(2.0));
  for (int c = 0; c < 3; c++) {
    basis.khat[c] = khat[c];
    basis.h_plus[c] = Kokkos::complex<Real>(e1[c] * s, e2[c] * s);
    basis.h_minus[c] = Kokkos::complex<Real>(e1[c] * s, -e2[c] * s);
  }
  return basis;
}

// Projects one Fourier mode's complex field vector onto compressive/plus/
// minus. khat, h_plus, h_minus form an orthonormal complex basis, so
// |compressive|^2 + |plus|^2 + |minus|^2 == |Vx|^2+|Vy|^2+|Vz|^2 exactly
// (Parseval/orthonormality) at every mode except k=0, where all three are
// zero by construction while |V(0)|^2 is generally nonzero -- the DC mode is
// simply not decomposed (see the file-level comment above).
struct ModeProjection {
  Kokkos::complex<Real> compressive, plus, minus;
};

KOKKOS_INLINE_FUNCTION
ModeProjection ProjectMode(const HelicalBasis &basis, Kokkos::complex<Real> Vx,
                           Kokkos::complex<Real> Vy, Kokkos::complex<Real> Vz) {
  ModeProjection p;
  p.compressive = Vx * basis.khat[0] + Vy * basis.khat[1] + Vz * basis.khat[2];
  p.plus = Kokkos::conj(basis.h_plus[0]) * Vx + Kokkos::conj(basis.h_plus[1]) * Vy +
          Kokkos::conj(basis.h_plus[2]) * Vz;
  p.minus = Kokkos::conj(basis.h_minus[0]) * Vx + Kokkos::conj(basis.h_minus[1]) * Vy +
           Kokkos::conj(basis.h_minus[2]) * Vz;
  return p;
}

// Decomposes an already-Fourier-transformed 3-component vector field into
// three per-mode complex scalar arrays: the projection onto khat
// (compressive), h_plus (plus), and h_minus (minus). FT_field_3comp is
// 3 * size_fourier_space_box() (the caller's own Forward() output, in the
// same layout FFTManager::Forward produces); each output array is
// size_fourier_space_box(). See the file-level comment above for the DC
// mode (k=0) and Hermitian-symmetry properties of these outputs.
struct DecomposedFourierField {
  parthenon::ParArray1D<Kokkos::complex<Real>> compressive, plus, minus;
};

DecomposedFourierField
DecomposeFourierField(parthenon::Mesh *pm,
                      const parthenon::ParArray1D<Kokkos::complex<Real>> &FT_field_3comp);

} // namespace energy_transfer

#endif // ENERGY_TRANSFER_DECOMPOSITION_HPP_
