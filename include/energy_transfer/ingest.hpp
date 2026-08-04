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

// Resolved ADIOS2/bp5 variable names to read. Each vector-valued field is
// three separate variable names (x/y/z), matching how the offline snapshot
// format stores components.
struct ADIOS2FieldNaming {
  std::string rho;
  bool input_conserved = false;
  std::array<std::string, 3> mom_or_vel;      // momentum_{x,y,z} or velocity_{x,y,z}
  std::optional<std::array<std::string, 3>> mag;
  std::optional<std::string> pres_or_energy;  // pressure or total_energy
  std::optional<std::array<std::string, 3>> acc;
  Real gamma = 5.0 / 3.0;

  // Convenience: reads the same <energy_transfer>/input_*_field parameters
  // the original driver used (mesh/field name pairs with defaults), for
  // callers driving configuration from a parthenon input deck.
  static ADIOS2FieldNaming FromInput(parthenon::ParameterInput *pin, bool need_mag,
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
                            const ADIOS2FieldNaming &naming);

} // namespace energy_transfer

#endif // ENERGY_TRANSFER_INGEST_HPP_
