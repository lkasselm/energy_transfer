#ifndef ENERGY_TRANSFER_IO_OPENPMD_HPP_
#define ENERGY_TRANSFER_IO_OPENPMD_HPP_

#include <string>

#include <mpi.h>

#include "energy_transfer/shell_transfer.hpp"

namespace energy_transfer {

// Writes every entry of result.matrices as an openPMD mesh record named
// after its term (shape per its requested DecompositionMode), and every
// entry of result.spectra as three records "<name>_pow_sum", "<name>_k_sum",
// "<name>_count_sum" (matching parthenon::utils::fft::CalcSpectrum's
// [num_bins, 3] layout -- "_" rather than "/" since ADIOS2 rejects "/" in
// dataset names outright). shell_edges/n_shells/binning are written as
// iteration attributes.
void WriteResult(const TransferResult &result, const std::string &output_file,
                 int output_number, MPI_Comm comm = MPI_COMM_WORLD);

} // namespace energy_transfer

#endif // ENERGY_TRANSFER_IO_OPENPMD_HPP_
