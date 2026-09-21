#include "energy_transfer/ingest.hpp"

#include <adios2.h>

#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>
#include <utils/fft_manager.hpp>
#include <utils/uniform_grid_helper.hpp>

namespace energy_transfer {

namespace {

bool HasSuffix(const std::string &s, const std::string &suffix) {
  return s.size() >= suffix.size() &&
        s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

} // namespace

InputFileFormat DetectInputFileFormat(const std::string &input_file) {
  if (HasSuffix(input_file, ".bp")) return InputFileFormat::ADIOS2;
  if (HasSuffix(input_file, ".phdf") || HasSuffix(input_file, ".h5") ||
      HasSuffix(input_file, ".hdf5")) {
    return InputFileFormat::ParthenonHDF5;
  }
  PARTHENON_FAIL("energy_transfer/input_file must be an ADIOS2/bp5 file (.bp) or a "
                 "Parthenon HDF5 output file (.phdf, .h5, or .hdf5), got: " + input_file);
  return InputFileFormat::ADIOS2;
}

FlatFields ReadADIOS2Fields(parthenon::Mesh *pmesh, const std::string &input_file,
                            const FileFieldNaming &naming) {
  PARTHENON_REQUIRE_THROWS(
      input_file.size() >= 3 && input_file.substr(input_file.size() - 3) == ".bp",
      "input_file must be an ADIOS2/bp5 file (ending in .bp), got: " + input_file);

  auto mesh_size = pmesh->mesh_size;
  const int Nx = mesh_size.nx(parthenon::X1DIR);
  const int Ny = mesh_size.nx(parthenon::X2DIR);
  const int Nz = mesh_size.nx(parthenon::X3DIR);

  auto FFTMgr = pmesh->GetFFTManager();
  const auto fft_size_inbox = FFTMgr->size_real_space_box();
  auto UniformGridHelper = pmesh->GetUniformGridHelper();

  FlatFields fields;
  fields.fft_size_inbox = fft_size_inbox;
  fields.is_conserved = naming.input_conserved;
  fields.gamma = naming.gamma;
  fields.rho = parthenon::ParArray1D<Real>("rho_flat", fft_size_inbox);
  fields.mom_or_vel = parthenon::ParArray1D<Real>("mom_or_vel_flat", 3 * fft_size_inbox);
  if (naming.mag) fields.mag = parthenon::ParArray1D<Real>("mag_flat", 3 * fft_size_inbox);
  if (naming.pres_or_energy)
    fields.pres_or_energy =
        parthenon::ParArray1D<Real>("pres_or_energy_flat", fft_size_inbox);
  if (naming.acc) fields.acc = parthenon::ParArray1D<Real>("acc_flat", 3 * fft_size_inbox);

  const auto &local_box = UniformGridHelper->local_mesh_box;
  const adios2::Dims start = {static_cast<std::size_t>(local_box.low[2]),
                              static_cast<std::size_t>(local_box.low[1]),
                              static_cast<std::size_t>(local_box.low[0])};
  const adios2::Dims count = {static_cast<std::size_t>(local_box.size[2]),
                              static_cast<std::size_t>(local_box.size[1]),
                              static_cast<std::size_t>(local_box.size[0])};

  adios2::ADIOS adios(MPI_COMM_WORLD);
  adios2::IO io = adios.DeclareIO("EnergyTransferInputReader");
  adios2::Engine reader = io.Open(input_file, adios2::Mode::Read);
  reader.BeginStep();

  enum class InputRealType { Float, Double };

  auto get_input_type_and_shape = [&](const std::string &name,
                                      adios2::Dims &shape) -> InputRealType {
    auto var_double = io.InquireVariable<double>(name);
    if (var_double) {
      shape = var_double.Shape();
      return InputRealType::Double;
    }
    auto var_float = io.InquireVariable<float>(name);
    if (var_float) {
      shape = var_float.Shape();
      return InputRealType::Float;
    }
    PARTHENON_FAIL("Variable '" + name + "' not found as float or double in " + input_file);
    return InputRealType::Double;
  };

  auto validate_shape = [&](const std::string &name, const adios2::Dims &shape) {
    PARTHENON_REQUIRE_THROWS(
        shape.size() == 3 && static_cast<int>(shape[0]) == Nz &&
            static_cast<int>(shape[1]) == Ny && static_cast<int>(shape[2]) == Nx,
        "ADIOS2 variable '" + name + "' dimensions [" + std::to_string(shape[0]) + ", " +
            std::to_string(shape[1]) + ", " + std::to_string(shape[2]) +
            "] do not match mesh dimensions [" + std::to_string(Nz) + ", " +
            std::to_string(Ny) + ", " + std::to_string(Nx) + "]");
  };

  auto read_field = [&](const std::string &name, parthenon::ParArray1D<Real> &dest,
                        const std::size_t offset) {
    adios2::Dims shape;
    const auto type = get_input_type_and_shape(name, shape);
    validate_shape(name, shape);
    auto dest_sub = Kokkos::subview(dest, Kokkos::make_pair(offset, offset + fft_size_inbox));
    auto dest_view = dest;
    const auto dest_offset = offset;
    if (type == InputRealType::Double) {
      auto var = io.InquireVariable<double>(name);
      PARTHENON_REQUIRE_THROWS(var, "Variable '" + name + "' not found in " + input_file);
      std::vector<double> buffer(fft_size_inbox);
      var.SetSelection({start, count});
      reader.Get(var, buffer.data(), adios2::Mode::Deferred);
      reader.PerformGets();
#if SINGLE_PRECISION_ENABLED
      parthenon::ParArray1D<double> input_double("input_double", fft_size_inbox);
      auto host_view = Kokkos::View<double *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
          buffer.data(), fft_size_inbox);
      Kokkos::deep_copy(input_double, host_view);
      parthenon::par_for(
          "ConvertInputDoubleToReal", std::size_t(0), fft_size_inbox - 1,
          KOKKOS_LAMBDA(const std::size_t idx) {
            dest_view(dest_offset + idx) = static_cast<Real>(input_double(idx));
          });
      Kokkos::fence();
#else
      auto host_view = Kokkos::View<Real *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
          buffer.data(), fft_size_inbox);
      Kokkos::deep_copy(dest_sub, host_view);
#endif
    } else {
      auto var = io.InquireVariable<float>(name);
      PARTHENON_REQUIRE_THROWS(var, "Variable '" + name + "' not found in " + input_file);
      std::vector<float> buffer(fft_size_inbox);
      var.SetSelection({start, count});
      reader.Get(var, buffer.data(), adios2::Mode::Deferred);
      reader.PerformGets();
#if SINGLE_PRECISION_ENABLED
      auto host_view = Kokkos::View<Real *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
          buffer.data(), fft_size_inbox);
      Kokkos::deep_copy(dest_sub, host_view);
#else
      parthenon::ParArray1D<float> input_float("input_float", fft_size_inbox);
      auto host_view = Kokkos::View<float *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
          buffer.data(), fft_size_inbox);
      Kokkos::deep_copy(input_float, host_view);
      parthenon::par_for(
          "ConvertInputFloatToReal", std::size_t(0), fft_size_inbox - 1,
          KOKKOS_LAMBDA(const std::size_t idx) {
            dest_view(dest_offset + idx) = static_cast<Real>(input_float(idx));
          });
      Kokkos::fence();
#endif
    }
  };

  read_field(naming.rho, fields.rho, 0);
  read_field(naming.mom_or_vel[0], fields.mom_or_vel, 0);
  read_field(naming.mom_or_vel[1], fields.mom_or_vel, fft_size_inbox);
  read_field(naming.mom_or_vel[2], fields.mom_or_vel, 2 * fft_size_inbox);

  if (naming.mag) {
    read_field((*naming.mag)[0], fields.mag, 0);
    read_field((*naming.mag)[1], fields.mag, fft_size_inbox);
    read_field((*naming.mag)[2], fields.mag, 2 * fft_size_inbox);
  }
  if (naming.pres_or_energy) {
    read_field(*naming.pres_or_energy, fields.pres_or_energy, 0);
  }
  if (naming.acc) {
    read_field((*naming.acc)[0], fields.acc, 0);
    read_field((*naming.acc)[1], fields.acc, fft_size_inbox);
    read_field((*naming.acc)[2], fields.acc, 2 * fft_size_inbox);
  }

  reader.EndStep();
  reader.Close();

  return fields;
}

} // namespace energy_transfer
