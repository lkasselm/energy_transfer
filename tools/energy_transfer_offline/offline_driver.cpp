#include <parthenon_manager.hpp>
#include <utils/error_checking.hpp>

#include "energy_transfer/ingest.hpp"
#include "energy_transfer/io_openpmd.hpp"
#include "energy_transfer/shell_transfer.hpp"

// Thin standalone driver: builds just enough Parthenon Mesh infrastructure
// (for FFTManager/UniformGridHelper) to read an ADIOS2/bp5 snapshot and run
// the same energy_transfer library any live Parthenon app links against.
int main(int argc, char *argv[]) {
  parthenon::ParthenonManager pman;
  pman.app_input->ProblemGenerator = [](parthenon::MeshBlock *, parthenon::ParameterInput *) {};

  auto status = pman.ParthenonInitEnv(argc, argv);
  if (status == parthenon::ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (status == parthenon::ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  pman.ParthenonInitPackagesAndMesh();

  {
    auto *pin = pman.pinput.get();
    auto *pmesh = pman.pmesh.get();

    PARTHENON_REQUIRE_THROWS(pin->DoesParameterExist("energy_transfer", "input_file"),
                             "energy-transfer-offline requires <energy_transfer>/input_file "
                             "to be set to an ADIOS2/bp5 snapshot.");
    const auto input_file = pin->GetString("energy_transfer", "input_file");
    const auto output_file = pin->GetOrAddString("energy_transfer", "output_file", "transfer");
    const auto output_number = pin->GetOrAddInteger("energy_transfer", "output_number", 0);

    auto cfg = energy_transfer::ShellTransferConfig::FromInput(pin);
    const auto req = energy_transfer::ComputeFieldRequirements(cfg);
    const auto naming = energy_transfer::ADIOS2FieldNaming::FromInput(pin, req.mag,
                                                                       req.pres_or_energy,
                                                                       req.acc);

    auto result = energy_transfer::ComputeShellTransferFromFile(pmesh, input_file, naming, cfg);
    energy_transfer::WriteResult(result, output_file, output_number);
  }

  pman.ParthenonFinalize();
  return 0;
}
