add_executable(energy-transfer-offline offline_driver.cpp)
target_link_libraries(energy-transfer-offline PRIVATE energy_transfer Parthenon::parthenon MPI::MPI_CXX)
