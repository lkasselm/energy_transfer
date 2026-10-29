#ifndef ENERGY_TRANSFER_HELMHOLTZ_HPP_
#define ENERGY_TRANSFER_HELMHOLTZ_HPP_

#include <basic_types.hpp>
#include <kokkos_types.hpp>
#include <mesh/mesh.hpp>

namespace energy_transfer {

using parthenon::Real;

// Helmholtz decomposition of a periodic 3-component field F into its mean, a
// gradient (compressive, curl-free) part and a curl (solenoidal,
// divergence-free) part:
//   F = <F> + grad(phi) + curl(A),
// with A in Coulomb gauge (div A = 0). In Fourier space, for k != 0:
//   phi_hat = -i (k . F_hat) / |k|^2,     A_hat = i (k x F_hat) / |k|^2,
// and phi_hat = A_hat = 0 at k = 0 -- the mean of F is neither a gradient nor
// a curl of a periodic potential, so it's returned in neither (and both
// potentials have zero mean). Wavenumbers are physical (2*pi/L scaled), so
// the potentials carry the right units on any box size.
//
// F is packed 3 * size_real_space_box() (see flat_fields.hpp).

// Fourier-space result, for callers that never need to leave Fourier space
// (e.g. spec_helicity, a co-spectrum of A_hat and B_hat). FT_F is the input's
// own forward transform, kept since such callers need it too.
struct HelmholtzFourier {
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_F;   // 3 * size_fourier_space_box()
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_phi; // size_fourier_space_box()
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_A;   // 3 * size_fourier_space_box()
};
HelmholtzFourier ComputeHelmholtzFourier(parthenon::Mesh *pm,
                                         const parthenon::ParArray1D<Real> &F);

// Real-space potentials. E.g. for a force density F balanced by pressure,
// c_s^2 rho0 grad(delta) = F_compressive gives the overdensity
// delta = phi / (c_s^2 rho0).
struct HelmholtzPotentials {
  parthenon::ParArray1D<Real> phi; // size_real_space_box()
  parthenon::ParArray1D<Real> A;   // 3 * size_real_space_box()
};
HelmholtzPotentials CalcHelmholtzPotentials(parthenon::Mesh *pm,
                                            const parthenon::ParArray1D<Real> &F);

} // namespace energy_transfer

#endif // ENERGY_TRANSFER_HELMHOLTZ_HPP_
