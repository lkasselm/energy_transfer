#ifndef ENERGY_TRANSFER_REGISTRY_HPP_
#define ENERGY_TRANSFER_REGISTRY_HPP_

#include <map>
#include <string>
#include <vector>

#include <basic_types.hpp>
#include <kokkos_types.hpp>
#include <mesh/mesh.hpp>
#include <utils/fft_manager.hpp>

#include "energy_transfer/flat_fields.hpp"

namespace energy_transfer {

using parthenon::Real;
using TransferReal = double;

// Base Fourier-transformed fields. FT_W is always populated; the others are
// only populated when some requested quantity's required_base_fields (or
// mediator_only_base_fields, when mediator decomposition was requested)
// names them ("U"/"B"/"P"/"Acc"/"b") -- see ComputeFieldFftClosure() in
// shell_transfer.cpp.
struct FourierFields {
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_W;
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_U;
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_B;
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_P;
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_Acc;
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_b; // FT of b = mag / sqrt(rho)
};

// Global (shell-independent) auxiliary real-space arrays. W_flat is always
// populated (FT_W is too); the rest are lazily computed once per
// ComputeEnergyTransfer call when some requested quantity's
// required_base_fields names them ("b"/"DivU"/"Divb").
struct GlobalAux {
  parthenon::ParArray1D<Real> W_flat; // W = sqrt(rho) * velocity
  parthenon::ParArray1D<Real> b_flat; // b = mag / sqrt(rho)
  parthenon::ParArray1D<Real> DivU;   // div(velocity)
  parthenon::ParArray1D<Real> Divb;   // div(b)
};

// One axis's shell restriction: which modes pass a ShellFilter/
// ShellFilterDerivative call. When active, that's the half-open band
// (low, high]; when not, low/high still hold the exact no-restriction
// sentinel (kNoRestrictionKLow, NoRestrictionKHigh(Nx,Ny,Nz) -- see
// spectral_kernels.hpp) so any code path that just passes them through
// stays correct, while a provider that can skip the filtering entirely
// checks `active` and returns the raw field (see FilterVector/FilterScalar
// in registry.cpp), paying zero extra FFT cost.
struct ShellRestriction {
  bool active = false;
  Real low = 0.0, high = 0.0;
};

// Everything a derived-quantity provider needs to compute one shell's worth
// of a real-space quantity. fields->mom_or_vel/pres_or_energy are already in
// primitive form (ConvertConservedToPrimitive has already run).
//
// The two ShellRestrictions below are deliberately the same type with the
// same contract: donor, receiver, and mediator are all on equal footing, and
// "this axis isn't being decomposed" means the same exact, DC-inclusive,
// zero-extra-cost thing on each of them.
struct ShellWorkspace {
  parthenon::FFTManager *fft_mgr;
  const FlatFields *fields;
  const FourierFields *ft;
  const GlobalAux *aux;
  std::size_t fft_size_inbox;
  std::size_t fft_size_outbox;
  int Nx, Ny, Nz; // global mesh dims
  Real two_pi_over_L;

  ShellRestriction axis;     // this quantity's own donor-or-receiver shell
                            // (whichever of the two this call is evaluating)
  ShellRestriction mediator; // the shell of the mediating field it reads
};

// Always returns a 3*fft_size_inbox vector (scalar quantities are broadcast
// into vector form so every term reduces via the same DotProductReduce).
using DerivedQuantityFn = parthenon::ParArray1D<Real> (*)(const ShellWorkspace &);

// required_base_fields: subset of {"U","B","P","Acc","b","DivU","Divb"},
// always needed regardless of DecompositionMode. mediator_only_base_fields
// is the (usually empty) subset of the same tags needed only when this
// quantity's mediator is actually being shell-decomposed (see
// DecompositionMode::mediator_resolved) -- e.g. "U_dot_grad_W" doesn't
// otherwise need FT_U (its donor field is FT_W), but does once its U
// mediator is being shell-filtered. has_decomposable_mediator marks
// quantities whose mediator is a real (vector) field that CAN be
// shell-restricted, as opposed to a scalar normalization (rho) that can't --
// see shell_transfer.cpp's mediator_resolved validation.
struct DerivedQuantity {
  std::vector<std::string> required_base_fields;
  std::vector<std::string> mediator_only_base_fields;
  bool has_decomposable_mediator = false;
  DerivedQuantityFn fn;
};

struct TransferTerm {
  std::string q_side_quantity;
  std::string k_side_quantity;
  Real prefactor = 1.0;
};

const std::map<std::string, DerivedQuantity> &BuiltinQuantities();
const std::map<std::string, TransferTerm> &BuiltinTerms();

// sum(a[idx] * b[idx]) over idx in [0, n), reduced across MPI ranks.
TransferReal DotProductReduce(const parthenon::ParArray1D<Real> &a,
                              const parthenon::ParArray1D<Real> &b, std::size_t n);

} // namespace energy_transfer

#endif // ENERGY_TRANSFER_REGISTRY_HPP_
