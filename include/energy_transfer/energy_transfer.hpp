#ifndef ENERGY_TRANSFER_ENERGY_TRANSFER_HPP_
#define ENERGY_TRANSFER_ENERGY_TRANSFER_HPP_

// Umbrella header: the complete public API of energy_transfer, for
// applications that just want "the library". Code inside the library (and its
// tests) should keep including only the individual headers it uses.
// Deliberately not included here (implementation detail): spectral_kernels.hpp,
// decomposition.hpp, registry.hpp.

#include "energy_transfer/field_spec.hpp"     // LiveFieldSpec, Make*LiveSpec
#include "energy_transfer/flat_fields.hpp"    // FlatFields, PackVector/UnpackVector
#include "energy_transfer/ingest.hpp"         // GatherLiveFields, GatherLiveVector, Read*Fields
#include "energy_transfer/convert.hpp"        // ConvertConservedToPrimitive
#include "energy_transfer/shell_transfer.hpp" // ComputeEnergyTransfer
#include "energy_transfer/spectra.hpp"        // ComputeSpectra
#include "energy_transfer/helicity.hpp"       // CalcHelicity
#include "energy_transfer/lorentz_force.hpp"  // CalcLorentzForce
#include "energy_transfer/io_openpmd.hpp"     // WriteResult

#endif // ENERGY_TRANSFER_ENERGY_TRANSFER_HPP_
