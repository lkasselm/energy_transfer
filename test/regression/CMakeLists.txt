# Gold-standard regression test against the upstream Python reference tool's own
# checked-in test fixture (github.com/pgrete/energy-transfer-analysis, testing/).
# Off by default: it needs a ~370MB external download (checksummed, cached after
# the first run) plus yt to convert it, neither of which a normal build/CI run
# should be forced to pull in.
option(ENERGY_TRANSFER_ENABLE_GOLD_STANDARD_TEST
  "Enable the DD0024 gold-standard regression test against pgrete/energy-transfer-analysis \
(requires a ~370MB external download, yt, and adios2; off by default)" OFF)

if(ENERGY_TRANSFER_ENABLE_GOLD_STANDARD_TEST)
  if(NOT ENERGY_TRANSFER_BUILD_TOOLS)
    message(FATAL_ERROR
      "ENERGY_TRANSFER_ENABLE_GOLD_STANDARD_TEST requires ENERGY_TRANSFER_BUILD_TOOLS=ON "
      "(the test drives the energy-transfer-offline executable).")
  endif()

  # parthenon_SOURCE_DIR is set by whichever Parthenon's project() call actually ran
  # this configure (the vendored external/parthenon submodule in the standalone
  # build -- the only case this test is reachable in, since AthenaPK's own build
  # sets ENERGY_TRANSFER_BUILD_TOOLS=OFF) -- more robust than a relative path, since
  # it resolves correctly regardless of where Parthenon actually came from.
  if(NOT DEFINED parthenon_SOURCE_DIR)
    message(FATAL_ERROR "ENERGY_TRANSFER_ENABLE_GOLD_STANDARD_TEST: parthenon_SOURCE_DIR is not set.")
  endif()
  include(${parthenon_SOURCE_DIR}/cmake/TestSetup.cmake)

  set(ENERGY_TRANSFER_GOLD_DATA_CACHE_DIR "${CMAKE_BINARY_DIR}/external_data/dd0024" CACHE PATH
    "Cache directory for the downloaded/converted DD0024 gold-standard dataset (outside git, persists across ctest re-runs)")

  setup_test_serial("gold_standard_dd0024"
    "--driver $<TARGET_FILE:energy-transfer-offline> \
--driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/gold_standard_dd0024/dd0024.in" "gold_standard")

  # SKIP_RETURN_CODE lets TestCase.Prepare() report "data unavailable" (no network,
  # missing yt, checksum mismatch, ...) as a clean ctest skip instead of a failure --
  # see gold_standard_dd0024.py / fetch_and_convert.py.
  set_tests_properties(regression_test:gold_standard_dd0024 PROPERTIES
    SKIP_RETURN_CODE 77
    ENVIRONMENT "ENERGY_TRANSFER_DD0024_CACHE=${ENERGY_TRANSFER_GOLD_DATA_CACHE_DIR}")
endif()
