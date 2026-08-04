#ifndef ENERGY_TRANSFER_FLAT_FIELDS_HPP_
#define ENERGY_TRANSFER_FLAT_FIELDS_HPP_

#include <cstddef>

#include <basic_types.hpp>
#include <kokkos_types.hpp>

namespace energy_transfer {

using parthenon::Real;

// Shared flat-array representation both ingestion paths (live MeshData gather
// and offline ADIOS2 read) produce. rho/mom_or_vel are always populated;
// mag/pres_or_energy/acc are only allocated when some requested quantity
// needs them (empty ParArray1D otherwise).
struct FlatFields {
  std::size_t fft_size_inbox = 0;
  bool is_conserved = false; // true: mom_or_vel holds momentum, pres_or_energy holds total energy
  Real gamma = 5.0 / 3.0;    // only consulted if is_conserved and pres_or_energy is populated

  parthenon::ParArray1D<Real> rho;             // fft_size_inbox
  parthenon::ParArray1D<Real> mom_or_vel;      // 3 * fft_size_inbox
  parthenon::ParArray1D<Real> mag;             // 3 * fft_size_inbox, or empty
  parthenon::ParArray1D<Real> pres_or_energy;  // fft_size_inbox, or empty
  parthenon::ParArray1D<Real> acc;             // 3 * fft_size_inbox, or empty
};

} // namespace energy_transfer

#endif // ENERGY_TRANSFER_FLAT_FIELDS_HPP_
