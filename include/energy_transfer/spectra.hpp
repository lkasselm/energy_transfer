#ifndef ENERGY_TRANSFER_SPECTRA_HPP_
#define ENERGY_TRANSFER_SPECTRA_HPP_

#include <map>
#include <string>
#include <vector>

#include <mesh/mesh.hpp>
#include <parameter_input.hpp>

#include "energy_transfer/flat_fields.hpp"
#include "energy_transfer/registry.hpp" // for the shared TransferReal alias only

namespace energy_transfer {

// Reads energy_transfer/spectra= (comma-separated names, default "spec_U").
std::vector<std::string> ParseSpectrumNames(parthenon::ParameterInput *pin);

// Whether computing this spectrum requires the magnetic field to be loaded.
// Throws for an unknown name. Used by ComputeFieldRequirements to decide
// whether ingestion must load FlatFields::mag.
bool SpectrumNeedsMag(const std::string &name);

// Computes every requested spectrum independently -- no shell-transfer
// state, no ShellTransferConfig, no dependency on any term/quantity being
// requested. fields must already be in primitive form. "spec_U"/"spec_rho"/
// "spec_W"/"spec_B" are plain power spectra; "spec_U_decomp"/"spec_W_decomp"/
// "spec_B_decomp" each expand into four entries in the returned map (the
// plain spectrum plus its compressive/plus/minus directional decomposition,
// computed together in one shared Forward() FFT -- see decomposition.hpp
// and spectral_kernels.hpp's BinFourierSpectrum).
std::map<std::string, parthenon::HostArray2D<TransferReal>>
ComputeSpectra(parthenon::Mesh *pm, const FlatFields &fields,
               const std::vector<std::string> &spectrum_names);

} // namespace energy_transfer

#endif // ENERGY_TRANSFER_SPECTRA_HPP_
