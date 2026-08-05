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

# Mesh::GetFFTManager()/GetUniformGridHelper() (which energy_transfer calls
# unconditionally throughout) only exist at all when Parthenon was configured
# with -DPARTHENON_ENABLE_FFT=ON -- src/CMakeLists.txt guards their sources
# and PUBLIC-propagated compile definition behind that same option(). Check
# it here (best-effort: PARTHENON_ENABLE_FFT may not be visible when Parthenon
# was found via find_package(parthenon) rather than add_subdirectory) so a
# misconfigured Parthenon fails with a clear message instead of a wall of
# "no member named GetFFTManager in Mesh" compiler errors.
if(DEFINED PARTHENON_ENABLE_FFT AND NOT PARTHENON_ENABLE_FFT)
  message(FATAL_ERROR
    "energy_transfer requires Parthenon to be built with -DPARTHENON_ENABLE_FFT=ON "
    "(it exposes Mesh::GetFFTManager()/GetUniformGridHelper(), FFTManager, "
    "UniformGridHelper, and CalcSpectrum, all of which this library depends on "
    "unconditionally). Reconfigure Parthenon with that flag set.")
endif()

# ADIOS2: energy_transfer calls the raw ADIOS2 C++ API directly (ingest_adios2.cpp),
# and Parthenon's own CMakeLists never links an adios2:: target onto Parthenon::parthenon
# (only openPMD, HDF5, HeFFTe, etc.) -- so this find is not redundant, it mirrors what
# the original example driver's CMakeLists already had to do for the same reason.
if(NOT TARGET adios2::cxx_mpi)
  find_package(ADIOS2 REQUIRED COMPONENTS CXX MPI)
endif()

# openPMD: Parthenon's own CMakeLists links `target_link_libraries(parthenon PUBLIC
# openPMD::openPMD)`, so when Parthenon is pulled in via add_subdirectory in this same
# configure run (the common case: energy_transfer added as a subdirectory of an app that
# already builds Parthenon, e.g. AthenaPK), the openPMD::openPMD target already exists
# and this find_package call is skipped -- openPMD comes along for free via the PUBLIC
# link to Parthenon::parthenon below. It only actually runs as a fallback for a truly
# standalone build against an *installed* Parthenon, since parthenonConfig.cmake does not
# re-export openPMD as a dependency for find_package(parthenon) consumers.
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
