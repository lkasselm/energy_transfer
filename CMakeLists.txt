cmake_minimum_required(VERSION 3.16)
project(EnergyTransfer VERSION 0.1.0 LANGUAGES CXX)

option(ENERGY_TRANSFER_BUILD_TOOLS "Build the offline post-processing executable" ON)
option(ENERGY_TRANSFER_BUILD_TESTS "Build unit/smoke tests" ON)

set(ENERGY_TRANSFER_PARTHENON_SOURCE_DIR "" CACHE PATH
    "Path to a Parthenon source checkout to add_subdirectory() when no \
Parthenon::parthenon target already exists and no installed Parthenon is \
found via find_package. Leave empty to require one of those two instead.")

if(NOT TARGET Parthenon::parthenon)
  find_package(parthenon QUIET CONFIG)
  if(NOT parthenon_FOUND AND ENERGY_TRANSFER_PARTHENON_SOURCE_DIR)
    add_subdirectory(${ENERGY_TRANSFER_PARTHENON_SOURCE_DIR} parthenon)
  elseif(NOT parthenon_FOUND)
    message(FATAL_ERROR
      "energy_transfer needs a Parthenon target. Either: (a) add this "
      "project via add_subdirectory() from a build that already defines "
      "Parthenon::parthenon (e.g. AthenaPK's build), (b) install Parthenon "
      "and make it discoverable via find_package(parthenon), or (c) set "
      "-DENERGY_TRANSFER_PARTHENON_SOURCE_DIR=/path/to/parthenon to build "
      "it from source alongside this library.")
  endif()
endif()

if(NOT TARGET adios2::cxx_mpi)
  find_package(ADIOS2 REQUIRED COMPONENTS CXX MPI)
endif()
if(NOT TARGET openPMD::openPMD)
  find_package(openPMD REQUIRED CONFIG)
endif()
find_package(MPI REQUIRED COMPONENTS CXX)

add_library(energy_transfer
  src/field_spec.cpp
  src/spectral_kernels.cpp
  src/convert.cpp
  src/ingest_live.cpp
  src/ingest_adios2.cpp
  src/registry.cpp
  src/shell_transfer.cpp
  src/io_openpmd.cpp
)
add_library(EnergyTransfer::energy_transfer ALIAS energy_transfer)

target_include_directories(energy_transfer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(energy_transfer
  PUBLIC Parthenon::parthenon
  PRIVATE adios2::cxx_mpi openPMD::openPMD MPI::MPI_CXX
)
target_compile_features(energy_transfer PUBLIC cxx_std_17)

if(ENERGY_TRANSFER_BUILD_TOOLS)
  add_subdirectory(tools/energy_transfer_offline)
endif()

if(ENERGY_TRANSFER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

install(TARGETS energy_transfer EXPORT EnergyTransferTargets
        ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT EnergyTransferTargets NAMESPACE EnergyTransfer::
        DESTINATION lib/cmake/EnergyTransfer)
