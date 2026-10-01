#include "input_deck.hpp"

#include <array>
#include <sstream>
#include <utility>

#include <utils/error_checking.hpp>

namespace {

// Trims ASCII whitespace from both ends, so "UBT, UBP" and "UBT,UBP" parse
// identically -- a token that's all whitespace comes back empty and is
// dropped by SplitCommaList's caller.
std::string Trim(const std::string &s) {
  const auto first = s.find_first_not_of(" \t\n\r");
  if (first == std::string::npos) return "";
  const auto last = s.find_last_not_of(" \t\n\r");
  return s.substr(first, last - first + 1);
}

std::vector<std::string> SplitCommaList(const std::string &s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token = Trim(token);
    if (!token.empty()) out.push_back(token);
  }
  return out;
}

// energy_transfer/mode= is just the set of axes that are decomposed: any
// subset of donor/mediator/receiver, in any order. An empty list decomposes
// nothing (one global number per term); "donor,receiver" is the default.
energy_transfer::DecompositionMode ParseDecompositionMode(const std::string &s) {
  energy_transfer::DecompositionMode mode{false, false, false};
  for (const auto &axis : SplitCommaList(s)) {
    if (axis == "donor") {
      mode.donor_resolved = true;
    } else if (axis == "mediator") {
      mode.mediator_resolved = true;
    } else if (axis == "receiver") {
      mode.receiver_resolved = true;
    } else {
      PARTHENON_FAIL(("energy_transfer: unknown mode axis '" + axis +
                      "' -- energy_transfer/mode= takes any comma-separated subset of "
                      "donor, mediator, receiver")
                         .c_str());
    }
  }
  return mode;
}

energy_transfer::BinningSpec ParseBinningSpecFromKeys(parthenon::ParameterInput *pin,
                                                      const std::string &binning_key,
                                                      const std::string &num_shells_key,
                                                      const std::string &shell_edges_key) {
  const auto binning_str = pin->GetOrAddString("energy_transfer", binning_key, "lin");
  const auto num_shells = pin->GetOrAddInteger("energy_transfer", num_shells_key, 20);
  if (binning_str == "lin") return energy_transfer::BinningSpec::Linear(num_shells);
  if (binning_str == "log") return energy_transfer::BinningSpec::Log(num_shells);
  if (binning_str == "custom") {
    const auto edges_str = pin->GetOrAddString("energy_transfer", shell_edges_key, "");
    std::vector<energy_transfer::Real> edges;
    for (const auto &token : SplitCommaList(edges_str)) {
      edges.push_back(static_cast<energy_transfer::Real>(std::stod(token)));
    }
    PARTHENON_REQUIRE_THROWS(edges.size() >= 2,
                             "energy_transfer/" + binning_key +
                                 "=custom requires energy_transfer/" + shell_edges_key +
                                 " to list at least 2 comma-separated bin-edge values, e.g. " +
                                 shell_edges_key + " = 0.5,1.5,2.5,16.0,26.5,28.5,32.0");
    return energy_transfer::BinningSpec::Custom(std::move(edges));
  }
  PARTHENON_FAIL(("energy_transfer/" + binning_key + " must be 'lin', 'log', or 'custom'").c_str());
  return energy_transfer::BinningSpec::Linear(num_shells);
}

std::string JoinInputName(const std::string &prefix, const std::string &mesh,
                          const std::string &field) {
  std::string name = prefix;
  auto append = [&](const std::string &part) {
    if (part.empty()) return;
    if (!name.empty() && name.back() != '/') name += "/";
    name += part;
  };
  append(mesh);
  append(field);
  return name;
}

energy_transfer::FileFieldNaming ParseFileFieldNamingADIOS2(parthenon::ParameterInput *pin,
                                                            bool need_mag,
                                                            bool need_pres_or_energy,
                                                            bool need_acc) {
  energy_transfer::FileFieldNaming naming;

  const auto input_quantity_type =
      pin->GetOrAddString("energy_transfer", "input_quantity_type", "primitive");
  PARTHENON_REQUIRE_THROWS(
      input_quantity_type == "primitive" || input_quantity_type == "conserved",
      "energy_transfer/input_quantity_type must be 'primitive' or 'conserved'");
  naming.input_conserved = input_quantity_type == "conserved";
  naming.gamma = pin->GetOrAddReal("energy_transfer", "gamma", 5.0 / 3.0);

  const auto prefix = pin->GetOrAddString("energy_transfer", "input_variable_prefix", "");
  auto input_name = [&](const std::string &mesh_param, const std::string &field_param,
                        const std::string &flat_default,
                        const std::string &component_default) -> std::string {
    const auto mesh = pin->GetOrAddString("energy_transfer", mesh_param, std::string(""));
    const auto field_default = mesh.empty() ? flat_default : component_default;
    const auto field = pin->GetOrAddString("energy_transfer", field_param, field_default);
    return JoinInputName(prefix, mesh, field);
  };

  naming.rho = input_name("input_rho_mesh", "input_rho_field", "rho", "SCALAR");

  if (naming.input_conserved) {
    naming.mom_or_vel = {
        input_name("input_momentum_mesh", "input_momentum_x_field", "mom_x", "x"),
        input_name("input_momentum_mesh", "input_momentum_y_field", "mom_y", "y"),
        input_name("input_momentum_mesh", "input_momentum_z_field", "mom_z", "z")};
  } else {
    naming.mom_or_vel = {
        input_name("input_velocity_mesh", "input_velocity_x_field", "vel_x", "x"),
        input_name("input_velocity_mesh", "input_velocity_y_field", "vel_y", "y"),
        input_name("input_velocity_mesh", "input_velocity_z_field", "vel_z", "z")};
  }

  // Total energy includes the magnetic contribution, so converting conserved
  // energy to pressure always requires the magnetic field, even if no
  // requested term otherwise needs it -- mirrors driver.cpp:310-311.
  if (need_mag || (naming.input_conserved && need_pres_or_energy)) {
    naming.mag = std::array<std::string, 3>{
        input_name("input_magnetic_mesh", "input_magnetic_x_field", "mag_x", "x"),
        input_name("input_magnetic_mesh", "input_magnetic_y_field", "mag_y", "y"),
        input_name("input_magnetic_mesh", "input_magnetic_z_field", "mag_z", "z")};
  }

  if (need_pres_or_energy) {
    naming.pres_or_energy =
        naming.input_conserved
            ? input_name("input_total_energy_mesh", "input_total_energy_field",
                         "total_energy", "SCALAR")
            : input_name("input_pressure_mesh", "input_pressure_field", "pres", "SCALAR");
  }

  if (need_acc) {
    naming.acc = std::array<std::string, 3>{
        input_name("input_acceleration_mesh", "input_acceleration_x_field", "acc_x", "x"),
        input_name("input_acceleration_mesh", "input_acceleration_y_field", "acc_y", "y"),
        input_name("input_acceleration_mesh", "input_acceleration_z_field", "acc_z", "z")};
  }

  return naming;
}

// Same field-selection parameter names as the ADIOS2 path (e.g.
// input_rho_field), but defaulting to AthenaPK's native prim/cons component
// names (see athenapk/src/hydro/hydro.cpp) since a Parthenon HDF5 dump
// stores AthenaPK's own field layout directly -- there's no mesh/prefix
// concept to resolve here, unlike ADIOS2's flat/mesh naming.
energy_transfer::FileFieldNaming ParseFileFieldNamingPHDF(parthenon::ParameterInput *pin,
                                                          bool need_mag,
                                                          bool need_pres_or_energy,
                                                          bool need_acc) {
  energy_transfer::FileFieldNaming naming;

  const auto input_quantity_type =
      pin->GetOrAddString("energy_transfer", "input_quantity_type", "primitive");
  PARTHENON_REQUIRE_THROWS(
      input_quantity_type == "primitive" || input_quantity_type == "conserved",
      "energy_transfer/input_quantity_type must be 'primitive' or 'conserved'");
  naming.input_conserved = input_quantity_type == "conserved";
  naming.gamma = pin->GetOrAddReal("energy_transfer", "gamma", 5.0 / 3.0);

  auto field_name = [&](const std::string &field_param, const std::string &default_component) {
    return pin->GetOrAddString("energy_transfer", field_param, default_component);
  };

  naming.rho = field_name("input_rho_field", "prim_density");

  if (naming.input_conserved) {
    naming.mom_or_vel = {field_name("input_momentum_x_field", "cons_momentum_density_1"),
                        field_name("input_momentum_y_field", "cons_momentum_density_2"),
                        field_name("input_momentum_z_field", "cons_momentum_density_3")};
  } else {
    naming.mom_or_vel = {field_name("input_velocity_x_field", "prim_velocity_1"),
                        field_name("input_velocity_y_field", "prim_velocity_2"),
                        field_name("input_velocity_z_field", "prim_velocity_3")};
  }

  // Total energy includes the magnetic contribution, so converting conserved
  // energy to pressure always requires the magnetic field, even if no
  // requested term otherwise needs it -- mirrors the ADIOS2 path.
  if (need_mag || (naming.input_conserved && need_pres_or_energy)) {
    naming.mag = std::array<std::string, 3>{
        field_name("input_magnetic_x_field", "prim_magnetic_field_1"),
        field_name("input_magnetic_y_field", "prim_magnetic_field_2"),
        field_name("input_magnetic_z_field", "prim_magnetic_field_3")};
  }

  if (need_pres_or_energy) {
    naming.pres_or_energy = naming.input_conserved
                                ? field_name("input_total_energy_field",
                                            "cons_total_energy_density")
                                : field_name("input_pressure_field", "prim_pressure");
  }

  if (need_acc) {
    naming.acc = std::array<std::string, 3>{field_name("input_acceleration_x_field", "acc_1"),
                                            field_name("input_acceleration_y_field", "acc_2"),
                                            field_name("input_acceleration_z_field", "acc_3")};
  }

  return naming;
}

} // namespace

energy_transfer::ShellTransferConfig ParseShellTransferConfig(parthenon::ParameterInput *pin) {
  energy_transfer::ShellTransferConfig cfg;

  // Shared default: binning=/num_shells=/shell_edges= (no prefix), used by
  // any axis without its own explicit donor_/mediator_/receiver_ override --
  // an input deck that only sets these keeps all three axes sharing one
  // binning, exactly as before mediator decomposition existed.
  const auto default_binning = ParseBinningSpecFromKeys(pin, "binning", "num_shells", "shell_edges");

  cfg.donor_binning = pin->DoesParameterExist("energy_transfer", "donor_binning")
                          ? ParseBinningSpecFromKeys(pin, "donor_binning", "donor_num_shells",
                                                     "donor_shell_edges")
                          : default_binning;
  cfg.mediator_binning =
      pin->DoesParameterExist("energy_transfer", "mediator_binning")
          ? ParseBinningSpecFromKeys(pin, "mediator_binning", "mediator_num_shells",
                                     "mediator_shell_edges")
          : default_binning;
  cfg.receiver_binning =
      pin->DoesParameterExist("energy_transfer", "receiver_binning")
          ? ParseBinningSpecFromKeys(pin, "receiver_binning", "receiver_num_shells",
                                     "receiver_shell_edges")
          : default_binning;

  const auto terms_str = pin->GetOrAddString("energy_transfer", "terms", "UUA,UUC");
  cfg.terms = SplitCommaList(terms_str);

  const auto mode_str = pin->GetOrAddString("energy_transfer", "mode", "donor,receiver");
  cfg.mode = ParseDecompositionMode(mode_str);

  return cfg;
}

std::vector<std::string> ParseSpectrumNames(parthenon::ParameterInput *pin) {
  const auto spectra_str = pin->GetOrAddString("energy_transfer", "spectra", "spec_U");
  return SplitCommaList(spectra_str);
}

energy_transfer::FileFieldNaming ParseFileFieldNaming(parthenon::ParameterInput *pin,
                                                       const std::string &input_file,
                                                       bool need_mag, bool need_pres_or_energy,
                                                       bool need_acc) {
  switch (energy_transfer::DetectInputFileFormat(input_file)) {
  case energy_transfer::InputFileFormat::ADIOS2:
    return ParseFileFieldNamingADIOS2(pin, need_mag, need_pres_or_energy, need_acc);
  case energy_transfer::InputFileFormat::ParthenonHDF5:
    return ParseFileFieldNamingPHDF(pin, need_mag, need_pres_or_energy, need_acc);
  }
  PARTHENON_FAIL("energy_transfer: unreachable");
  return energy_transfer::FileFieldNaming{};
}
