cmake_minimum_required(VERSION 3.16)
# LANGUAGES C CXX, not just CXX: find_package(MPI ... COMPONENTS C ...) below
# needs the C language itself enabled (ADIOS2's exported cxx11_mpi target
# links against MPI::MPI_C) -- declaring it here directly is more robust than
# relying on it propagating from Parthenon's own nested project(LANGUAGES C
# CXX) call, whether that runs via this repo's vendored submodule or an
# app's own Parthenon build.
project(EnergyTransfer VERSION 0.1.0 LANGUAGES C CXX)

option(ENERGY_TRANSFER_BUILD_TOOLS "Build the offline post-processing executable" ON)
option(ENERGY_TRANSFER_BUILD_TESTS "Build unit/smoke tests" ON)

# Parthenon is vendored as a submodule (external/parthenon), mirroring the
# same nested-submodule convention AthenaPK itself uses (Kokkos is a
# submodule of Parthenon, Parthenon+Kokkos are submodules of AthenaPK).
# The guard below means this vendored copy is only actually used for a
# genuinely standalone build of this repo: when energy_transfer is instead
# added via add_subdirectory() from an app that already builds its own
# Parthenon (e.g. AthenaPK, right after its own add_subdirectory(external/parthenon)),
# Parthenon::parthenon already exists by the time this file runs, so this
# block is skipped entirely and that app's Parthenon build is reused as-is --
# this vendored submodule is never even configured in that case.
if(NOT TARGET Parthenon::parthenon)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/parthenon/CMakeLists.txt)
    # Parthenon's own default is PARTHENON_DISABLE_OPENPMD=ON (openPMD off
    # unless explicitly requested) -- override it here the same way
    # AthenaPK's own top-level CMakeLists.txt does before its equivalent
    # add_subdirectory(external/parthenon) call, so Parthenon actually
    # builds/fetches openPMD and creates the openPMD::openPMD target this
    # library links against (io_openpmd.cpp) PUBLIC via Parthenon::parthenon.
    set(PARTHENON_DISABLE_OPENPMD OFF CACHE BOOL "" FORCE)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/parthenon parthenon)
  else()
    find_package(parthenon REQUIRED)
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

# MPI: found first (both C and CXX components) because ADIOS2's own exported
# targets file (adios2-cxx11-targets.cmake) sets a link interface referencing
# MPI::MPI_C on adios2::cxx11_mpi -- if only the CXX component were requested,
# or this ran after find_package(ADIOS2), that target wouldn't exist yet and
# find_package(ADIOS2) itself would fail deep inside ADIOS2's own cmake files.
find_package(MPI REQUIRED COMPONENTS C CXX)

# ADIOS2: energy_transfer calls the raw ADIOS2 C++ API directly (ingest_adios2.cpp),
# and Parthenon's own CMakeLists never links an adios2:: target onto Parthenon::parthenon
# (only openPMD, HDF5, HeFFTe, etc.) -- so this find is not redundant, it mirrors what
# the original example driver's CMakeLists already had to do for the same reason.
#
# The C++/MPI component target's name changed across ADIOS2 releases --
# "adios2::cxx11_mpi" historically, "adios2::cxx_mpi" in newer releases -- and
# not every install exports both, so probe for whichever exists instead of
# hardcoding one (the original example driver's CMakeLists hardcodes
# adios2::cxx_mpi, which is why it fails identically against an ADIOS2
# install -- like the one at hand -- that only exports adios2::cxx11_mpi).
if(NOT TARGET adios2::cxx_mpi AND NOT TARGET adios2::cxx11_mpi)
  find_package(ADIOS2 REQUIRED COMPONENTS CXX MPI)
endif()
if(TARGET adios2::cxx_mpi)
  set(_energy_transfer_adios2_target adios2::cxx_mpi)
elseif(TARGET adios2::cxx11_mpi)
  set(_energy_transfer_adios2_target adios2::cxx11_mpi)
else()
  message(FATAL_ERROR
    "energy_transfer: found ADIOS2 but neither adios2::cxx_mpi nor "
    "adios2::cxx11_mpi target exists -- check your ADIOS2 install was built "
    "with MPI + C++ bindings enabled.")
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

add_library(energy_transfer
  src/field_spec.cpp
  src/spectral_kernels.cpp
  src/convert.cpp
  src/ingest_live.cpp
  src/ingest_adios2.cpp
  src/ingest_phdf.cpp
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
  PRIVATE ${_energy_transfer_adios2_target} openPMD::openPMD MPI::MPI_CXX
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
