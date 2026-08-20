#ifndef ENERGY_TRANSFER_INGEST_HPP_
#define ENERGY_TRANSFER_INGEST_HPP_

#include <array>
#include <optional>
#include <string>

#include <interface/mesh_data.hpp>
#include <mesh/mesh.hpp>
#include <parameter_input.hpp>

#include "energy_transfer/field_spec.hpp"
#include "energy_transfer/flat_fields.hpp"

namespace energy_transfer {

enum class InputFileFormat { ADIOS2, ParthenonHDF5 };

// Detects the offline snapshot format from input_file's extension (.bp ->
// ADIOS2; .phdf/.h5/.hdf5 -> Parthenon HDF5). Throws for any other suffix.
InputFileFormat DetectInputFileFormat(const std::string &input_file);

// Resolved variable/component names to read from an offline snapshot file.
// Each vector-valued field is three separate names (x/y/z), matching how
// both supported file formats store components.
struct FileFieldNaming {
  std::string rho;
  bool input_conserved = false;
  std::array<std::string, 3> mom_or_vel;      // momentum_{x,y,z} or velocity_{x,y,z}
  std::optional<std::array<std::string, 3>> mag;
  std::optional<std::string> pres_or_energy;  // pressure or total_energy
  std::optional<std::array<std::string, 3>> acc;
  Real gamma = 5.0 / 3.0;

  // Convenience: reads the same <energy_transfer>/input_*_field parameters
  // the original driver used (mesh/field name pairs with defaults, only
  // meaningful for ADIOS2 -- ignored for Parthenon HDF5, which has no mesh/
  // prefix concept), for callers driving configuration from a parthenon
  // input deck.
  static FileFieldNaming FromInputADIOS2(parthenon::ParameterInput *pin, bool need_mag,
                                         bool need_pres_or_energy, bool need_acc);

  // Same idea, but defaulting to AthenaPK's native prim/cons component names
  // (see athenapk/src/hydro/hydro.cpp) instead of ADIOS2's flat/mesh naming,
  // since a Parthenon HDF5 dump stores AthenaPK's own field layout directly.
  static FileFieldNaming FromInputPHDF(parthenon::ParameterInput *pin, bool need_mag,
                                       bool need_pres_or_energy, bool need_acc);

  // Picks FromInputADIOS2 or FromInputPHDF based on input_file's extension.
  static FileFieldNaming FromInput(parthenon::ParameterInput *pin,
                                   const std::string &input_file, bool need_mag,
                                   bool need_pres_or_energy, bool need_acc);
};

// Gathers fields from a live MeshData container according to spec, into the
// shared flat-array representation. Deduplicates PackVariables calls by
// variable name, so e.g. AthenaPK's packed "prim" spec (where density/
// velocity/magnetic/pressure all share one variable name) packs it once.
FlatFields GatherLiveFields(parthenon::Mesh *pmesh, parthenon::MeshData<Real> *md,
                            const LiveFieldSpec &spec);

// Reads fields directly from an ADIOS2/bp5 snapshot file into the shared
// flat-array representation, selecting each rank's local box via the mesh's
// UniformGridHelper.
FlatFields ReadADIOS2Fields(parthenon::Mesh *pmesh, const std::string &input_file,
                            const FileFieldNaming &naming);

// Reads fields directly from a Parthenon HDF5 (.phdf/.h5/.hdf5) output file
// -- the format AthenaPK itself writes -- via raw HDF5 hyperslab reads of
// only the region overlapping this rank's local box. Requires Parthenon to
// have been built with HDF5 support (PARTHENON_DISABLE_HDF5 not set);
// throws a clear error otherwise. Requires a single-level (unrefined) input
// mesh, consistent with energy_transfer's general uniform-grid requirement.
FlatFields ReadPHDFFields(parthenon::Mesh *pmesh, const std::string &input_file,
                          const FileFieldNaming &naming);

} // namespace energy_transfer

#endif // ENERGY_TRANSFER_INGEST_HPP_
