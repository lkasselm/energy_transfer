#ifndef ENERGY_TRANSFER_IO_OPENPMD_HPP_
#define ENERGY_TRANSFER_IO_OPENPMD_HPP_

#include <map>
#include <string>

#include <mpi.h>

#include "energy_transfer/shell_transfer.hpp"

namespace energy_transfer {

using SpectraResult = std::map<std::string, parthenon::HostArray2D<TransferReal>>;

// Writes whichever of this library's independent result kinds are passed
// (each is optional, nullable, and independent of the others -- pass
// nullptr for anything you didn't compute this call) into one openPMD
// iteration: shell_transfer non-null writes every entry of its matrices as
// a 3D mesh record named after its term (shape per its requested
// DecompositionMode -- the mediator axis has extent 1 unless
// mediator_resolved was set) plus donor/mediator/receiver's own
// edges/n_shells/binning as iteration attributes, prefixed accordingly;
// spectra non-null writes every entry as three records "<name>_pow_sum",
// "<name>_k_sum", "<name>_count_sum" (matching
// parthenon::utils::fft::CalcSpectrum's [num_bins, 3] layout -- "_" rather
// than "/" since ADIOS2 rejects "/" in dataset names outright). Passing
// neither still creates a valid (empty) iteration.
void WriteResult(const std::string &output_file, int output_number,
                 const ShellTransferResult *shell_transfer = nullptr,
                 const SpectraResult *spectra = nullptr, MPI_Comm comm = MPI_COMM_WORLD);

} // namespace energy_transfer

#endif // ENERGY_TRANSFER_IO_OPENPMD_HPP_
