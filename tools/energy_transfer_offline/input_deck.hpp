#ifndef ENERGY_TRANSFER_OFFLINE_INPUT_DECK_HPP_
#define ENERGY_TRANSFER_OFFLINE_INPUT_DECK_HPP_

#include <string>
#include <vector>

#include <parameter_input.hpp>

#include "energy_transfer/ingest.hpp"
#include "energy_transfer/shell_transfer.hpp"

// Deck-parsing for this standalone tool only -- turns a Parthenon
// ParameterInput's <energy_transfer> block into the plain, programmatically
// constructed config structs the library itself takes. The library core has
// no notion of an input deck: an in-situ caller (e.g. AthenaPK) is always
// expected to build ShellTransferConfig/FileFieldNaming directly in C++
// (see docs/README.md's in-situ example), so this parsing lives here, next
// to the one executable that actually needs it, rather than in the library.

// Reads binning=/num_shells=/shell_edges=, donor_binning=/mediator_binning=/
// receiver_binning= (each with its own _num_shells=/_shell_edges=), terms=,
// and mode= -- see docs/README.md for the full key reference.
energy_transfer::ShellTransferConfig ParseShellTransferConfig(parthenon::ParameterInput *pin);

// Reads energy_transfer/spectra= (comma-separated names, default "spec_U").
std::vector<std::string> ParseSpectrumNames(parthenon::ParameterInput *pin);

// Picks the ADIOS2 or Parthenon HDF5 naming convention based on input_file's
// extension (via energy_transfer::DetectInputFileFormat), then reads the
// corresponding <energy_transfer>/input_*_field parameters -- see
// docs/README.md.
energy_transfer::FileFieldNaming ParseFileFieldNaming(parthenon::ParameterInput *pin,
                                                       const std::string &input_file,
                                                       bool need_mag, bool need_pres_or_energy,
                                                       bool need_acc);

#endif // ENERGY_TRANSFER_OFFLINE_INPUT_DECK_HPP_
