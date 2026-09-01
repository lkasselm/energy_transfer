#ifndef ENERGY_TRANSFER_IO_OPENPMD_HPP_
#define ENERGY_TRANSFER_IO_OPENPMD_HPP_

#include <string>

#include <mpi.h>

#include "energy_transfer/shell_transfer.hpp"

namespace energy_transfer {

// Writes every entry of result.matrices as a 3D openPMD mesh record named
// after its term (shape per its requested DecompositionMode -- the mediator
// axis has extent 1 unless mediator_resolved was set), and every
// entry of result.spectra as three records "<name>_pow_sum", "<name>_k_sum",
// "<name>_count_sum" (matching parthenon::utils::fft::CalcSpectrum's
// [num_bins, 3] layout -- "_" rather than "/" since ADIOS2 rejects "/" in
// dataset names outright). Each of donor/mediator/receiver's own edges/
// n_shells/binning (independent per axis, see ShellTransferConfig) are
// written as iteration attributes, prefixed accordingly.
void WriteResult(const TransferResult &result, const std::string &output_file,
                 int output_number, MPI_Comm comm = MPI_COMM_WORLD);

} // namespace energy_transfer

#endif // ENERGY_TRANSFER_IO_OPENPMD_HPP_
