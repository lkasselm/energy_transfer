#ifndef ENERGY_TRANSFER_FIELD_SPEC_HPP_
#define ENERGY_TRANSFER_FIELD_SPEC_HPP_

#include <array>
#include <optional>
#include <string>

#include <basic_types.hpp>

namespace energy_transfer {

using parthenon::Real;

// Describes where to find each physical quantity in a live MeshData container,
// as a variable name plus component index (or indices, for 3-vectors). Lets
// GatherLiveFields() pull from either separate named fields (density_var !=
// momentum_or_velocity_var != ...) or a single packed field addressed by
// component index (density_var == momentum_or_velocity_var == "cons"/"prim").
struct LiveFieldSpec {
  std::string density_var;
  int density_component = 0;

  std::string momentum_or_velocity_var;
  std::array<int, 3> momentum_or_velocity_components{0, 1, 2};
  bool is_conserved = false; // selects momentum(+energy) vs velocity(+pressure) semantics

  std::optional<std::string> magnetic_var;
  std::optional<std::array<int, 3>> magnetic_components;

  std::optional<std::string> pressure_or_energy_var;
  std::optional<int> pressure_or_energy_component;

  std::optional<std::string> acceleration_var;
  std::optional<std::array<int, 3>> acceleration_components;

  Real gamma = 5.0 / 3.0;
};

// Today's example layout: separate 1-comp/3-comp fields "rho"/"vel"/"mag"/"acc"/"pres".
LiveFieldSpec MakeSeparateFieldsLiveSpec();

// AthenaPK, reading already-derived primitives ("prim": IDN/IV1-3/IB1-3/IPR).
// Cheaper than the conserved variant since it reuses AthenaPK's own EOS.
LiveFieldSpec MakeAthenaPKPrimitiveLiveSpec(int idn, int iv1, int iv2, int iv3, int ipr,
                                           bool has_bfield, int ib1 = -1, int ib2 = -1,
                                           int ib3 = -1);

// AthenaPK, reading conserved state directly ("cons": IDN/IM1-3/IB1-3/IEN).
LiveFieldSpec MakeAthenaPKConservedLiveSpec(int idn, int im1, int im2, int im3, int ien,
                                            Real gamma, bool has_bfield, int ib1 = -1,
                                            int ib2 = -1, int ib3 = -1);

} // namespace energy_transfer

#endif // ENERGY_TRANSFER_FIELD_SPEC_HPP_
