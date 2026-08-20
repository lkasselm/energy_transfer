#ifndef ENERGY_TRANSFER_SHELL_TRANSFER_HPP_
#define ENERGY_TRANSFER_SHELL_TRANSFER_HPP_

#include <map>
#include <string>
#include <vector>

#include <interface/mesh_data.hpp>
#include <mesh/mesh.hpp>
#include <parameter_input.hpp>

#include "energy_transfer/field_spec.hpp"
#include "energy_transfer/flat_fields.hpp"
#include "energy_transfer/ingest.hpp"
#include "energy_transfer/registry.hpp"

namespace energy_transfer {

// How shells are defined. Custom lets a caller supply arbitrary bin edges
// directly, with no library change needed.
struct BinningSpec {
  enum class Type { Linear, Log, Custom } type = Type::Linear;
  int num_shells = 20;            // ignored if type == Custom
  std::vector<Real> custom_edges; // used only if type == Custom

  static BinningSpec Linear(int n) { return {Type::Linear, n, {}}; }
  static BinningSpec Log(int n) { return {Type::Log, n, {}}; }
  static BinningSpec Custom(std::vector<Real> edges) {
    return {Type::Custom, 0, std::move(edges)};
  }
};

// Resolution a requested term is computed at. "Q" is the donor/advecting
// shell (outer loop), "K" the receiving shell (inner loop). Collapsing a
// side evaluates that side's derived quantity once over the whole domain
// (k_low=0, k_high=unrestricted) instead of once per real shell bin --
// shells partition Fourier space, so this reconstructs the unfiltered field
// directly rather than summing per-shell results after the fact.
enum class DecompositionMode {
  Full,       // [n_shells, n_shells]
  BySender,   // [n_shells, 1]
  ByReceiver, // [1, n_shells]
  Total       // [1, 1]
};

struct TermRequest {
  std::string name;
  DecompositionMode mode = DecompositionMode::Full;
  TermRequest(std::string n, DecompositionMode m = DecompositionMode::Full)
      : name(std::move(n)), mode(m) {}
  TermRequest(const char *n, DecompositionMode m = DecompositionMode::Full)
      : name(n), mode(m) {}
};

struct ShellTransferConfig {
  BinningSpec binning = BinningSpec::Linear(20);
  std::vector<TermRequest> terms;          // names from the library's fixed built-in set (BuiltinTerms())
  std::vector<std::string> spectrum_names; // names from BuiltinSpectra()

  // Reads an <energy_transfer> input block: binning=lin|log, num_shells=,
  // terms=UUA,BBA:total,BUT:by_receiver,..., spectra=spec_U,spec_rho,...
  static ShellTransferConfig FromInput(parthenon::ParameterInput *pin);
};

struct TransferResult {
  int n_shells = 0;
  BinningSpec binning;
  std::vector<Real> shell_edges;
  std::map<std::string, parthenon::HostArray2D<TransferReal>> matrices; // keyed by term name
  std::map<std::string, parthenon::HostArray2D<TransferReal>> spectra;  // keyed by spectrum name
};

// Which real-space fields ingestion must load for the requested terms/
// spectra to be computable. Used to build a minimal LiveFieldSpec/
// FileFieldNaming before ingestion runs.
struct FieldRequirements {
  bool mag = false;
  bool pres_or_energy = false;
  bool acc = false;
};
FieldRequirements ComputeFieldRequirements(const ShellTransferConfig &cfg);

// Core computation: fields must already be in primitive form (see
// ConvertConservedToPrimitive) and populated per ComputeFieldRequirements(cfg).
TransferResult ComputeShellTransfer(parthenon::Mesh *pmesh, FlatFields &fields,
                                    const ShellTransferConfig &cfg);

// Live/on-the-fly entry point -- a plain function, no StateDescriptor/package
// registration. Call from an app's own UserWorkBeforeOutput or similar hook,
// which already has a Mesh*/MeshData<Real>*.
TransferResult ComputeShellTransferLive(parthenon::Mesh *pmesh, parthenon::MeshData<Real> *md,
                                        const LiveFieldSpec &spec,
                                        const ShellTransferConfig &cfg);

// Offline entry point: reads an ADIOS2/bp5 or Parthenon HDF5 (.phdf/.h5/
// .hdf5) snapshot directly, dispatching on input_file's extension (see
// DetectInputFileFormat in ingest.hpp) -- naming must have been built with
// the matching FileFieldNaming::FromInputADIOS2/FromInputPHDF/FromInput.
TransferResult ComputeShellTransferFromFile(parthenon::Mesh *pmesh,
                                            const std::string &input_file,
                                            const FileFieldNaming &naming,
                                            const ShellTransferConfig &cfg);

} // namespace energy_transfer

#endif // ENERGY_TRANSFER_SHELL_TRANSFER_HPP_
