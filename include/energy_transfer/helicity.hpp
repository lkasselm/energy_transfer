#ifndef ENERGY_TRANSFER_HELICITY_HPP_
#define ENERGY_TRANSFER_HELICITY_HPP_

#include <basic_types.hpp>
#include <kokkos_types.hpp>
#include <mesh/mesh.hpp>

namespace energy_transfer {

using parthenon::Real;

// Magnetic helicity: H(x) = A(x).B(x), where A is the Coulomb-gauge vector
// potential reconstructed spectrally from B (A_hat = i(k x B_hat)/|k|^2,
// A_hat(0)=0 since direction is undefined at k=0). Ported directly from
// athenapk/src/pgen/decaying_turbulence.cpp's UserWorkBeforeOutput (same
// formula, same DC handling), generalized to this library's usual flat
// FlatFields-style real-space array instead of pulling straight from a
// Parthenon MeshData variable -- a caller (in-situ or offline) hands in
// whichever 3-component B array it already has.

// The Coulomb-gauge vector potential of a real 3-component field B, kept in
// Fourier space -- the shared spectral step behind both CalcHelicity below
// and a magnetic helicity *spectrum* (a co-spectrum of A_hat and B_hat, see
// spectral_kernels.hpp's BinFourierCospectrum), which needs A_hat and B_hat
// directly rather than the real-space H(x) CalcHelicity produces. FT_B is
// returned alongside FT_A so a caller needing both doesn't have to
// forward-transform B a second time.
struct VectorPotentialFourier {
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_A; // 3 * size_fourier_space_box()
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_B; // 3 * size_fourier_space_box()
};

VectorPotentialFourier ComputeVectorPotentialFourier(parthenon::Mesh *pm,
                                                     const parthenon::ParArray1D<Real> &B);

// Magnetic helicity density H(x) = A(x).B(x) in real space. B is
// 3 * size_real_space_box(); the returned array is size_real_space_box()
// (one scalar per cell) -- a downstream caller can e.g. ScatterField it
// back into its own mesh the same way decaying_turbulence.cpp does today.
parthenon::ParArray1D<Real> CalcHelicity(parthenon::Mesh *pm, const parthenon::ParArray1D<Real> &B);

} // namespace energy_transfer

#endif // ENERGY_TRANSFER_HELICITY_HPP_
