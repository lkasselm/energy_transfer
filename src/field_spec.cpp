#include "energy_transfer/field_spec.hpp"

namespace energy_transfer {

LiveFieldSpec MakeSeparateFieldsLiveSpec() {
  LiveFieldSpec spec;
  spec.density_var = "rho";
  spec.density_component = 0;
  spec.momentum_or_velocity_var = "vel";
  spec.momentum_or_velocity_components = {0, 1, 2};
  spec.is_conserved = false;
  spec.magnetic_var = "mag";
  spec.magnetic_components = std::array<int, 3>{0, 1, 2};
  spec.pressure_or_energy_var = "pres";
  spec.pressure_or_energy_component = 0;
  spec.acceleration_var = "acc";
  spec.acceleration_components = std::array<int, 3>{0, 1, 2};
  return spec;
}

LiveFieldSpec MakeAthenaPKPrimitiveLiveSpec(int idn, int iv1, int iv2, int iv3, int ipr,
                                           bool has_bfield, int ib1, int ib2, int ib3) {
  LiveFieldSpec spec;
  spec.density_var = "prim";
  spec.density_component = idn;
  spec.momentum_or_velocity_var = "prim";
  spec.momentum_or_velocity_components = {iv1, iv2, iv3};
  spec.is_conserved = false;
  spec.pressure_or_energy_var = "prim";
  spec.pressure_or_energy_component = ipr;
  if (has_bfield) {
    spec.magnetic_var = "prim";
    spec.magnetic_components = std::array<int, 3>{ib1, ib2, ib3};
  }
  return spec;
}

LiveFieldSpec MakeAthenaPKConservedLiveSpec(int idn, int im1, int im2, int im3, int ien,
                                            Real gamma, bool has_bfield, int ib1, int ib2,
                                            int ib3) {
  LiveFieldSpec spec;
  spec.density_var = "cons";
  spec.density_component = idn;
  spec.momentum_or_velocity_var = "cons";
  spec.momentum_or_velocity_components = {im1, im2, im3};
  spec.is_conserved = true;
  spec.pressure_or_energy_var = "cons";
  spec.pressure_or_energy_component = ien;
  spec.gamma = gamma;
  if (has_bfield) {
    spec.magnetic_var = "cons";
    spec.magnetic_components = std::array<int, 3>{ib1, ib2, ib3};
  }
  return spec;
}

} // namespace energy_transfer
