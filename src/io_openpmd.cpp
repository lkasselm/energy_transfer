#include "energy_transfer/io_openpmd.hpp"

#include <iostream>

#include <openPMD/openPMD.hpp>

#include <utils/error_checking.hpp>

namespace energy_transfer {

namespace {

std::string BinningTypeName(BinningSpec::Type type) {
  switch (type) {
  case BinningSpec::Type::Linear: return "lin";
  case BinningSpec::Type::Log: return "log";
  case BinningSpec::Type::Custom: return "custom";
  }
  return "unknown";
}

std::string ResolveIterationFilename(const std::string &output_file) {
  std::string fname = output_file;
  const auto has_iteration_pattern = [](const std::string &name) {
    for (std::size_t pos = name.find('%'); pos != std::string::npos;
        pos = name.find('%', pos + 1)) {
      auto digit_pos = pos + 1;
      while (digit_pos < name.size() && name[digit_pos] >= '0' && name[digit_pos] <= '9') {
        digit_pos++;
      }
      if (digit_pos < name.size() && name[digit_pos] == 'T') return true;
    }
    return false;
  }(fname);
  const auto has_bp_suffix = fname.size() >= 3 && fname.substr(fname.size() - 3) == ".bp";
  if (!has_iteration_pattern) {
    if (has_bp_suffix) {
      fname.insert(fname.size() - 3, ".%05T");
    } else {
      fname += ".%05T";
    }
  }
  if (fname.size() < 3 || fname.substr(fname.size() - 3) != ".bp") {
    fname += ".bp";
  }
  return fname;
}

} // namespace

void WriteResult(const TransferResult &result, const std::string &output_file,
                 int output_number, MPI_Comm comm) {
  PARTHENON_REQUIRE_THROWS(output_number >= 0, "energy_transfer: output_number must be non-negative");

  const auto fname = ResolveIterationFilename(output_file);
  openPMD::Series series(fname, openPMD::Access::CREATE, comm, "{}");
  series.setIterationEncoding(openPMD::IterationEncoding::fileBased);

  auto it = series.iterations[static_cast<uint64_t>(output_number)];
  it.open();
  it.setAttribute("shell_edges", result.shell_edges);
  it.setAttribute("n_shells", result.n_shells);
  it.setAttribute("binning", BinningTypeName(result.binning.type));

  int my_rank = 0;
  MPI_Comm_rank(comm, &my_rank);

  auto write_matrix = [&](const std::string &name,
                          const parthenon::HostArray2D<TransferReal> &matrix) {
    auto mesh = it.meshes[name];
    auto comp = mesh[openPMD::MeshRecordComponent::SCALAR];
    const auto n_q = matrix.extent(0);
    const auto n_k = matrix.extent(1);
    // TransferResult::matrices is stored in-memory as (Q, K) -- dim0=donor
    // shell, dim1=receiver shell (see shell_transfer.hpp) -- but the original
    // driver's on-disk convention (and any tooling built against it) is
    // matrix(kk, q): row=K, col=Q. Transpose here at the I/O boundary so
    // output *files* match that established layout exactly, without
    // disturbing this library's own (Q,K) C++ API.
    std::vector<TransferReal> transposed(n_q * n_k);
    for (std::size_t qi = 0; qi < n_q; qi++) {
      for (std::size_t ki = 0; ki < n_k; ki++) {
        transposed[ki * n_q + qi] = matrix(qi, ki);
      }
    }
    openPMD::Extent extent = {static_cast<uint64_t>(n_k), static_cast<uint64_t>(n_q)};
    comp.resetDataset(openPMD::Dataset(openPMD::determineDatatype<TransferReal>(), extent));
    comp.storeChunkRaw(transposed.data(), {0, 0}, extent);
    it.seriesFlush();
  };
  for (const auto &[name, matrix] : result.matrices) {
    write_matrix(name, matrix);
  }

  auto write_vector_from_matrix = [&](const std::string &name,
                                      const parthenon::HostArray2D<TransferReal> &matrix,
                                      int col) {
    auto mesh = it.meshes[name];
    auto comp = mesh[openPMD::MeshRecordComponent::SCALAR];
    const auto num_bins = matrix.extent(0);
    std::vector<TransferReal> outdata(num_bins);
    for (int i = 0; i < static_cast<int>(num_bins); i++) outdata.at(i) = matrix(i, col);
    openPMD::Extent extent = {static_cast<uint64_t>(num_bins)};
    // The spectrum is already MPI-reduced to rank 0, so only rank 0 writes.
    if (my_rank == 0) {
      comp.resetDataset(openPMD::Dataset(openPMD::determineDatatype<TransferReal>(), extent));
      comp.storeChunkRaw(outdata.data(), {0}, extent);
    }
    it.seriesFlush();
  };
  for (const auto &[name, spectrum] : result.spectra) {
    write_vector_from_matrix(name + "/pow_sum", spectrum, 0);
    write_vector_from_matrix(name + "/k_sum", spectrum, 1);
    write_vector_from_matrix(name + "/count_sum", spectrum, 2);
  }

  series.close();
  if (my_rank == 0) {
    std::cout << "energy_transfer: wrote " << fname << " with iteration " << output_number
              << std::endl;
  }
}

} // namespace energy_transfer
