#include "energy_transfer/ingest.hpp"

#include <kokkos_abstraction.hpp>
#include <utils/fft_manager.hpp>
#include <utils/uniform_grid_helper.hpp>

namespace energy_transfer {

FlatFields GatherLiveFields(parthenon::Mesh *pmesh, parthenon::MeshData<Real> *md,
                            const LiveFieldSpec &spec) {
  auto FFTMgr = pmesh->GetFFTManager();
  const auto fft_size_inbox = FFTMgr->size_real_space_box();
  auto UniformGridHelper = pmesh->GetUniformGridHelper();
  auto helper = UniformGridHelper->GetKernelHelper();

  const bool has_mag = spec.magnetic_var.has_value();
  const bool has_pres = spec.pressure_or_energy_var.has_value();
  const bool has_acc = spec.acceleration_var.has_value();

  FlatFields fields;
  fields.fft_size_inbox = fft_size_inbox;
  fields.is_conserved = spec.is_conserved;
  fields.gamma = spec.gamma;
  fields.rho = parthenon::ParArray1D<Real>("rho_flat", fft_size_inbox);
  fields.mom_or_vel = parthenon::ParArray1D<Real>("mom_or_vel_flat", 3 * fft_size_inbox);
  if (has_mag) fields.mag = parthenon::ParArray1D<Real>("mag_flat", 3 * fft_size_inbox);
  if (has_pres)
    fields.pres_or_energy = parthenon::ParArray1D<Real>("pres_or_energy_flat", fft_size_inbox);
  if (has_acc) fields.acc = parthenon::ParArray1D<Real>("acc_flat", 3 * fft_size_inbox);

  // Pack each distinct variable name exactly once (a caller like AthenaPK
  // typically names the same packed "prim"/"cons" field for all of these).
  auto density_pack = md->PackVariables(std::vector<std::string>{spec.density_var});

  auto mom_pack = (spec.momentum_or_velocity_var == spec.density_var)
                      ? density_pack
                      : md->PackVariables(std::vector<std::string>{spec.momentum_or_velocity_var});

  auto mag_pack = mom_pack;
  if (has_mag) {
    if (*spec.magnetic_var == spec.density_var) {
      mag_pack = density_pack;
    } else if (*spec.magnetic_var == spec.momentum_or_velocity_var) {
      mag_pack = mom_pack;
    } else {
      mag_pack = md->PackVariables(std::vector<std::string>{*spec.magnetic_var});
    }
  }

  auto pres_pack = mom_pack;
  if (has_pres) {
    if (*spec.pressure_or_energy_var == spec.density_var) {
      pres_pack = density_pack;
    } else if (*spec.pressure_or_energy_var == spec.momentum_or_velocity_var) {
      pres_pack = mom_pack;
    } else if (has_mag && *spec.pressure_or_energy_var == *spec.magnetic_var) {
      pres_pack = mag_pack;
    } else {
      pres_pack = md->PackVariables(std::vector<std::string>{*spec.pressure_or_energy_var});
    }
  }

  auto acc_pack = mom_pack;
  if (has_acc) {
    if (*spec.acceleration_var == spec.density_var) {
      acc_pack = density_pack;
    } else if (*spec.acceleration_var == spec.momentum_or_velocity_var) {
      acc_pack = mom_pack;
    } else if (has_mag && *spec.acceleration_var == *spec.magnetic_var) {
      acc_pack = mag_pack;
    } else if (has_pres && *spec.acceleration_var == *spec.pressure_or_energy_var) {
      acc_pack = pres_pack;
    } else {
      acc_pack = md->PackVariables(std::vector<std::string>{*spec.acceleration_var});
    }
  }

  auto ib = md->GetBlockData(0)->GetBoundsI(parthenon::IndexDomain::interior);
  auto jb = md->GetBlockData(0)->GetBoundsJ(parthenon::IndexDomain::interior);
  auto kb = md->GetBlockData(0)->GetBoundsK(parthenon::IndexDomain::interior);
  const int num_blocks = pmesh->GetNumMeshBlocksThisRank();

  const int density_component = spec.density_component;
  const auto mom_components = spec.momentum_or_velocity_components;
  const auto mag_components = has_mag ? *spec.magnetic_components : std::array<int, 3>{};
  const int pres_component = has_pres ? *spec.pressure_or_energy_component : 0;
  const auto acc_components = has_acc ? *spec.acceleration_components : std::array<int, 3>{};

  auto rho = fields.rho;
  auto mom_or_vel = fields.mom_or_vel;
  auto mag = fields.mag;
  auto pres_or_energy = fields.pres_or_energy;
  auto acc = fields.acc;

  parthenon::par_for(
      "GatherLiveFields", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto idx = helper.FlatIndex(b, k, j, i);
        rho(idx) = density_pack(b, density_component, k, j, i);
        for (int n = 0; n < 3; n++) {
          mom_or_vel(n * fft_size_inbox + idx) = mom_pack(b, mom_components[n], k, j, i);
        }
        if (has_mag) {
          for (int n = 0; n < 3; n++) {
            mag(n * fft_size_inbox + idx) = mag_pack(b, mag_components[n], k, j, i);
          }
        }
        if (has_pres) {
          pres_or_energy(idx) = pres_pack(b, pres_component, k, j, i);
        }
        if (has_acc) {
          for (int n = 0; n < 3; n++) {
            acc(n * fft_size_inbox + idx) = acc_pack(b, acc_components[n], k, j, i);
          }
        }
      });
  Kokkos::fence();

  return fields;
}

} // namespace energy_transfer
