#include "energy_transfer/ingest.hpp"

#include "config.hpp"
#ifdef ENABLE_HDF5
#include <hdf5.h>

#include "outputs/parthenon_hdf5.hpp"
#endif

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>
#include <utils/fft_manager.hpp>
#include <utils/uniform_grid_helper.hpp>

namespace energy_transfer {

// Same field-selection parameter names as the ADIOS2 path (e.g.
// input_rho_field), but defaulting to AthenaPK's native prim/cons component
// names (see athenapk/src/hydro/hydro.cpp) since a Parthenon HDF5 dump
// stores AthenaPK's own field layout directly -- there's no mesh/prefix
// concept to resolve here, unlike ADIOS2's flat/mesh naming.
FileFieldNaming FileFieldNaming::FromInputPHDF(parthenon::ParameterInput *pin, bool need_mag,
                                               bool need_pres_or_energy, bool need_acc) {
  FileFieldNaming naming;

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

#ifndef ENABLE_HDF5

FlatFields ReadPHDFFields(parthenon::Mesh *, const std::string &, const FileFieldNaming &) {
  PARTHENON_FAIL("Reading a Parthenon HDF5 (.phdf/.h5/.hdf5) input file requires Parthenon "
                 "to be built with HDF5 support (ENABLE_HDF5).");
  return FlatFields{};
}

#else // ENABLE_HDF5

FlatFields ReadPHDFFields(parthenon::Mesh *pmesh, const std::string &input_file,
                          const FileFieldNaming &naming) {
  using namespace parthenon::HDF5;

  auto mesh_size = pmesh->mesh_size;
  const int Nx = mesh_size.nx(parthenon::X1DIR);
  const int Ny = mesh_size.nx(parthenon::X2DIR);
  const int Nz = mesh_size.nx(parthenon::X3DIR);

  auto FFTMgr = pmesh->GetFFTManager();
  const auto fft_size_inbox = FFTMgr->size_real_space_box();
  auto UniformGridHelper = pmesh->GetUniformGridHelper();
  const auto &local_box = UniformGridHelper->local_mesh_box;

  const H5F file = H5F::FromHIDCheck(H5Fopen(input_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
  const H5O info_obj = H5O::FromHIDCheck(H5Oopen(file, "Info", H5P_DEFAULT));

  const int num_blocks = HDF5ReadAttributeVec<int>(info_obj, "NumMeshBlocks").at(0);
  const auto mesh_block_size = HDF5ReadAttributeVec<int>(info_obj, "MeshBlockSize");
  PARTHENON_REQUIRE_THROWS(mesh_block_size.size() == 3,
                           "Expected Info/MeshBlockSize to have 3 entries in " + input_file);

  // NumComponents is written as a vector<size_t> (see parthenon_hdf5.cpp), unlike the
  // other Info attributes read as <int> above.
  const auto num_components_vec = HDF5ReadAttributeVec<std::size_t>(info_obj, "NumComponents");
  const auto component_names = HDF5ReadAttributeVec<std::string>(info_obj, "ComponentNames");
  const auto dataset_names = HDF5ReadAttributeVec<std::string>(info_obj, "OutputDatasetNames");
  PARTHENON_REQUIRE_THROWS(num_components_vec.size() == dataset_names.size(),
                           "Info/NumComponents and Info/OutputDatasetNames size mismatch in " +
                               input_file);

  // Map from a component's full name (e.g. "prim_density") to the dataset that
  // contains it and its component index within that dataset.
  std::unordered_map<std::string, std::pair<std::string, int>> component_map;
  {
    std::size_t idx = 0;
    for (std::size_t d = 0; d < dataset_names.size(); d++) {
      const int ncomp = static_cast<int>(num_components_vec[d]);
      for (int c = 0; c < ncomp; c++) {
        PARTHENON_REQUIRE_THROWS(idx < component_names.size(),
                                 "Info/ComponentNames too short in " + input_file);
        component_map[component_names[idx]] = {dataset_names[d], c};
        idx++;
      }
    }
  }

  std::vector<std::int64_t> levels(num_blocks);
  {
    const H5D ds = H5D::FromHIDCheck(H5Dopen2(file, "Levels", H5P_DEFAULT));
    PARTHENON_HDF5_CHECK(
        H5Dread(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, levels.data()));
  }
  std::vector<std::int64_t> logical_locations(static_cast<std::size_t>(num_blocks) * 3);
  {
    const H5D ds = H5D::FromHIDCheck(H5Dopen2(file, "LogicalLocations", H5P_DEFAULT));
    PARTHENON_HDF5_CHECK(H5Dread(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                                 logical_locations.data()));
  }

  // energy_transfer requires a uniform (single-level) grid for its FFTs, so every
  // block in the input file must be at the root level; the root grid size implied by
  // the blocks must also match this run's mesh.
  int max_lx1 = -1, max_lx2 = -1, max_lx3 = -1;
  for (int b = 0; b < num_blocks; b++) {
    PARTHENON_REQUIRE_THROWS(
        levels[b] == 0,
        "energy_transfer requires a single-level (non-refined) mesh when reading a "
        "Parthenon HDF5 input file, but block " +
            std::to_string(b) + " is at level " + std::to_string(levels[b]) + " in " +
            input_file);
    max_lx1 = std::max(max_lx1, static_cast<int>(logical_locations[b * 3 + 0]));
    max_lx2 = std::max(max_lx2, static_cast<int>(logical_locations[b * 3 + 1]));
    max_lx3 = std::max(max_lx3, static_cast<int>(logical_locations[b * 3 + 2]));
  }
  PARTHENON_REQUIRE_THROWS(
      (max_lx1 + 1) * mesh_block_size[0] == Nx && (max_lx2 + 1) * mesh_block_size[1] == Ny &&
          (max_lx3 + 1) * mesh_block_size[2] == Nz,
      "Root grid size implied by the blocks in '" + input_file +
          "' does not match the mesh dimensions [" + std::to_string(Nz) + ", " +
          std::to_string(Ny) + ", " + std::to_string(Nx) + "]");

  enum class InputRealType { Float, Double };

  auto get_dataset_dims = [&](hid_t ds) {
    const H5S space = H5S::FromHIDCheck(H5Dget_space(ds));
    const int rank = PARTHENON_HDF5_CHECK(H5Sget_simple_extent_ndims(space));
    PARTHENON_REQUIRE_THROWS(rank == 5, "Expected a 5D dataset [nblocks, ncomp, nz, ny, nx] "
                                        "in " +
                                            input_file + ", got rank " + std::to_string(rank));
    std::vector<hsize_t> dims(rank);
    PARTHENON_HDF5_CHECK(H5Sget_simple_extent_dims(space, dims.data(), NULL));
    return dims;
  };
  auto get_dataset_real_type = [&](hid_t ds) {
    const H5T dset_type = H5T::FromHIDCheck(H5Dget_type(ds));
    const double *dp = nullptr;
    const float *fp = nullptr;
    if (PARTHENON_HDF5_CHECK(H5Tequal(dset_type, getHDF5Type(dp))) > 0) {
      return InputRealType::Double;
    }
    if (PARTHENON_HDF5_CHECK(H5Tequal(dset_type, getHDF5Type(fp))) > 0) {
      return InputRealType::Float;
    }
    PARTHENON_FAIL("Dataset in " + input_file + " is neither float nor double");
    return InputRealType::Double;
  };

  // Reads a single named component (e.g. "prim_density") for the blocks that overlap
  // this rank's local box, and scatters them into dest at [offset, offset+fft_size_inbox).
  auto read_field_hdf5 = [&](const std::string &component_name,
                             parthenon::ParArray1D<Real> &dest, const std::size_t offset) {
    const auto it = component_map.find(component_name);
    PARTHENON_REQUIRE_THROWS(it != component_map.end(), "Component '" + component_name +
                                                             "' not found in " + input_file);
    const std::string &dataset_name = it->second.first;
    const int comp_idx = it->second.second;

    const H5D ds = H5D::FromHIDCheck(H5Dopen2(file, dataset_name.c_str(), H5P_DEFAULT));
    const auto dims = get_dataset_dims(ds); // [nblocks, ncomp, nz, ny, nx]
    PARTHENON_REQUIRE_THROWS(static_cast<int>(dims[0]) == num_blocks,
                             "Dataset '" + dataset_name + "' block count mismatch in " +
                                 input_file);
    const int ghost_x = (static_cast<int>(dims[4]) - mesh_block_size[0]) / 2;
    const int ghost_y = (static_cast<int>(dims[3]) - mesh_block_size[1]) / 2;
    const int ghost_z = (static_cast<int>(dims[2]) - mesh_block_size[2]) / 2;
    const auto real_type = get_dataset_real_type(ds);
    const H5S filespace = H5S::FromHIDCheck(H5Dget_space(ds));

    std::vector<Real> host_dest(fft_size_inbox, Real(0));

    for (int b = 0; b < num_blocks; b++) {
      const int bx = static_cast<int>(logical_locations[b * 3 + 0]) * mesh_block_size[0];
      const int by = static_cast<int>(logical_locations[b * 3 + 1]) * mesh_block_size[1];
      const int bz = static_cast<int>(logical_locations[b * 3 + 2]) * mesh_block_size[2];
      const int overlap_low_x = std::max(bx, local_box.low[0]);
      const int overlap_low_y = std::max(by, local_box.low[1]);
      const int overlap_low_z = std::max(bz, local_box.low[2]);
      const int overlap_high_x = std::min(bx + mesh_block_size[0], local_box.high[0] + 1);
      const int overlap_high_y = std::min(by + mesh_block_size[1], local_box.high[1] + 1);
      const int overlap_high_z = std::min(bz + mesh_block_size[2], local_box.high[2] + 1);
      if (overlap_low_x >= overlap_high_x || overlap_low_y >= overlap_high_y ||
          overlap_low_z >= overlap_high_z) {
        continue; // this block does not overlap the local box
      }
      const int count_x = overlap_high_x - overlap_low_x;
      const int count_y = overlap_high_y - overlap_low_y;
      const int count_z = overlap_high_z - overlap_low_z;

      const hsize_t h5_offset[5] = {static_cast<hsize_t>(b), static_cast<hsize_t>(comp_idx),
                                    static_cast<hsize_t>(overlap_low_z - bz + ghost_z),
                                    static_cast<hsize_t>(overlap_low_y - by + ghost_y),
                                    static_cast<hsize_t>(overlap_low_x - bx + ghost_x)};
      const hsize_t h5_count[5] = {1, 1, static_cast<hsize_t>(count_z),
                                   static_cast<hsize_t>(count_y),
                                   static_cast<hsize_t>(count_x)};
      PARTHENON_HDF5_CHECK(
          H5Sselect_hyperslab(filespace, H5S_SELECT_SET, h5_offset, NULL, h5_count, NULL));
      const H5S memspace = H5S::FromHIDCheck(H5Screate_simple(5, h5_count, NULL));

      const std::size_t block_count = static_cast<std::size_t>(count_x) * count_y * count_z;
      std::vector<Real> block_buf(block_count);
      if (real_type == InputRealType::Double) {
        std::vector<double> raw(block_count);
        PARTHENON_HDF5_CHECK(
            H5Dread(ds, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, raw.data()));
        for (std::size_t n = 0; n < block_count; n++) block_buf[n] = static_cast<Real>(raw[n]);
      } else {
        std::vector<float> raw(block_count);
        PARTHENON_HDF5_CHECK(
            H5Dread(ds, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, raw.data()));
        for (std::size_t n = 0; n < block_count; n++) block_buf[n] = static_cast<Real>(raw[n]);
      }

      for (int kz = 0; kz < count_z; kz++) {
        const int gz = overlap_low_z - local_box.low[2] + kz;
        for (int jy = 0; jy < count_y; jy++) {
          const int gy = overlap_low_y - local_box.low[1] + jy;
          const std::size_t dest_row =
              (static_cast<std::size_t>(gz) * local_box.size[1] + gy) * local_box.size[0];
          const std::size_t src_row = (static_cast<std::size_t>(kz) * count_y + jy) * count_x;
          for (int ix = 0; ix < count_x; ix++) {
            const int gx = overlap_low_x - local_box.low[0] + ix;
            host_dest[dest_row + gx] = block_buf[src_row + ix];
          }
        }
      }
    }

    auto dest_sub = Kokkos::subview(dest, Kokkos::make_pair(offset, offset + fft_size_inbox));
    auto host_view = Kokkos::View<Real *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        host_dest.data(), fft_size_inbox);
    Kokkos::deep_copy(dest_sub, host_view);
  };

  FlatFields fields;
  fields.fft_size_inbox = fft_size_inbox;
  fields.is_conserved = naming.input_conserved;
  fields.gamma = naming.gamma;
  fields.rho = parthenon::ParArray1D<Real>("rho_flat", fft_size_inbox);
  fields.mom_or_vel = parthenon::ParArray1D<Real>("mom_or_vel_flat", 3 * fft_size_inbox);
  if (naming.mag) fields.mag = parthenon::ParArray1D<Real>("mag_flat", 3 * fft_size_inbox);
  if (naming.pres_or_energy)
    fields.pres_or_energy = parthenon::ParArray1D<Real>("pres_or_energy_flat", fft_size_inbox);
  if (naming.acc) fields.acc = parthenon::ParArray1D<Real>("acc_flat", 3 * fft_size_inbox);

  read_field_hdf5(naming.rho, fields.rho, 0);
  read_field_hdf5(naming.mom_or_vel[0], fields.mom_or_vel, 0);
  read_field_hdf5(naming.mom_or_vel[1], fields.mom_or_vel, fft_size_inbox);
  read_field_hdf5(naming.mom_or_vel[2], fields.mom_or_vel, 2 * fft_size_inbox);

  if (naming.mag) {
    read_field_hdf5((*naming.mag)[0], fields.mag, 0);
    read_field_hdf5((*naming.mag)[1], fields.mag, fft_size_inbox);
    read_field_hdf5((*naming.mag)[2], fields.mag, 2 * fft_size_inbox);
  }
  if (naming.pres_or_energy) {
    read_field_hdf5(*naming.pres_or_energy, fields.pres_or_energy, 0);
  }
  if (naming.acc) {
    read_field_hdf5((*naming.acc)[0], fields.acc, 0);
    read_field_hdf5((*naming.acc)[1], fields.acc, fft_size_inbox);
    read_field_hdf5((*naming.acc)[2], fields.acc, 2 * fft_size_inbox);
  }

  return fields;
}

#endif // ENABLE_HDF5

} // namespace energy_transfer
