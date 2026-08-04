#ifndef ENERGY_TRANSFER_CONVERT_HPP_
#define ENERGY_TRANSFER_CONVERT_HPP_

#include "energy_transfer/flat_fields.hpp"

namespace energy_transfer {

// Converts FlatFields in place from conserved (momentum, total energy) to
// primitive (velocity, pressure) form. No-op if !fields.is_conserved.
// Requires fields.mag to be populated if fields.pres_or_energy is populated
// (total energy includes the magnetic contribution, which must be
// subtracted to recover pressure).
void ConvertConservedToPrimitive(FlatFields &fields);

} // namespace energy_transfer

#endif // ENERGY_TRANSFER_CONVERT_HPP_
