#include "energy_transfer/flat_fields.hpp"

#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

namespace energy_transfer {

parthenon::ParArray1D<Real> PackVector(const std::array<parthenon::ParArray1D<Real>, 3> &v) {
  const std::size_t n = v[0].size();
  PARTHENON_REQUIRE_THROWS(v[1].size() == n && v[2].size() == n,
                           "PackVector: all three components must have the same size.");
  parthenon::ParArray1D<Real> packed("packed_vector", 3 * n);
  auto vx = v[0];
  auto vy = v[1];
  auto vz = v[2];
  parthenon::par_for(
      "PackVector", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
        packed(0 * n + idx) = vx(idx);
        packed(1 * n + idx) = vy(idx);
        packed(2 * n + idx) = vz(idx);
      });
  Kokkos::fence();
  return packed;
}

std::array<parthenon::ParArray1D<Real>, 3> UnpackVector(const parthenon::ParArray1D<Real> &packed) {
  PARTHENON_REQUIRE_THROWS(packed.size() % 3 == 0,
                           "UnpackVector: packed size must be a multiple of 3.");
  const std::size_t n = packed.size() / 3;
  parthenon::ParArray1D<Real> vx("unpacked_x", n);
  parthenon::ParArray1D<Real> vy("unpacked_y", n);
  parthenon::ParArray1D<Real> vz("unpacked_z", n);
  parthenon::par_for(
      "UnpackVector", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
        vx(idx) = packed(0 * n + idx);
        vy(idx) = packed(1 * n + idx);
        vz(idx) = packed(2 * n + idx);
      });
  Kokkos::fence();
  return {vx, vy, vz};
}

} // namespace energy_transfer
