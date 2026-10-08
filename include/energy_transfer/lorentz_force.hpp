#ifndef ENERGY_TRANSFER_LORENTZ_FORCE_HPP_
#define ENERGY_TRANSFER_LORENTZ_FORCE_HPP_

#include <basic_types.hpp>
#include <kokkos_types.hpp>
#include <mesh/mesh.hpp>

namespace energy_transfer {

using parthenon::Real;

// Lorentz force density F(x) = J(x) x B(x), where J = curl(B) is the
// current density (Ampere's law curl(B) = J with mu0 = 1 -- the same
// units-free convention this library's magnetic-pressure/tension terms
// already use, e.g. BUT/BUPbb/UBPbb in src/registry.cpp). J is
// reconstructed spectrally from B (J_hat = i k x B_hat, the curl in Fourier
// space) -- unlike helicity.hpp's vector potential, this needs no DC
// special-case: curl has no 1/|k|^2 to blow up at k=0, it's simply zero
// there like every other mode's cross product with a zero k. B is
// 3 * size_real_space_box(); the returned array is 3 * size_real_space_box()
// too -- a downstream caller can e.g. ScatterField it back into its own
// mesh, the same way decaying_turbulence.cpp does for helicity today.
parthenon::ParArray1D<Real> CalcLorentzForce(parthenon::Mesh *pm,
                                             const parthenon::ParArray1D<Real> &B);

} // namespace energy_transfer

#endif // ENERGY_TRANSFER_LORENTZ_FORCE_HPP_
