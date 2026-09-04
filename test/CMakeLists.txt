add_executable(test_shell_filter_synthetic test_shell_filter_synthetic.cpp)
target_link_libraries(test_shell_filter_synthetic PRIVATE energy_transfer Parthenon::parthenon MPI::MPI_CXX)
add_test(NAME shell_filter_synthetic COMMAND test_shell_filter_synthetic -i ${CMAKE_CURRENT_SOURCE_DIR}/synthetic.in)

add_executable(test_live_ingest_smoke test_live_ingest_smoke.cpp)
target_link_libraries(test_live_ingest_smoke PRIVATE energy_transfer Parthenon::parthenon MPI::MPI_CXX)
add_test(NAME live_ingest_smoke COMMAND test_live_ingest_smoke -i ${CMAKE_CURRENT_SOURCE_DIR}/synthetic.in)

add_executable(test_mediator_decomposition_smoke test_mediator_decomposition_smoke.cpp)
target_link_libraries(test_mediator_decomposition_smoke PRIVATE energy_transfer Parthenon::parthenon MPI::MPI_CXX)
add_test(NAME mediator_decomposition_smoke COMMAND test_mediator_decomposition_smoke -i ${CMAKE_CURRENT_SOURCE_DIR}/synthetic.in)

add_executable(test_decomposed_spectrum_smoke test_decomposed_spectrum_smoke.cpp)
target_link_libraries(test_decomposed_spectrum_smoke PRIVATE energy_transfer Parthenon::parthenon MPI::MPI_CXX)
add_test(NAME decomposed_spectrum_smoke COMMAND test_decomposed_spectrum_smoke -i ${CMAKE_CURRENT_SOURCE_DIR}/synthetic.in)

add_subdirectory(regression)
