#include "energy_transfer/convert.hpp"

#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

namespace energy_transfer {

void ConvertConservedToPrimitive(FlatFields &fields) {
  if (!fields.is_conserved) return;

  const auto fft_size_inbox = fields.fft_size_inbox;
  auto rho = fields.rho;
  auto mom_or_vel = fields.mom_or_vel;

  parthenon::par_for(
      "ConvertMomentumToVelocity", std::size_t(0), fft_size_inbox - 1,
      KOKKOS_LAMBDA(const std::size_t idx) {
        const Real inv_rho = 1.0 / rho(idx);
        for (int n = 0; n < 3; n++) {
          mom_or_vel(n * fft_size_inbox + idx) *= inv_rho;
        }
      });
  Kokkos::fence();

  if (fields.pres_or_energy.size() > 0) {
    PARTHENON_REQUIRE_THROWS(
        fields.mag.size() > 0,
        "ConvertConservedToPrimitive: converting total energy to pressure requires "
        "the magnetic field to be loaded (total energy includes the magnetic "
        "contribution) -- populate FlatFields::mag before calling.");

    const Real gm1 = fields.gamma - 1.0;
    auto pres_or_energy = fields.pres_or_energy;
    auto mag = fields.mag;
    parthenon::par_for(
        "ConvertTotalEnergyToPressure", std::size_t(0), fft_size_inbox - 1,
        KOKKOS_LAMBDA(const std::size_t idx) {
          Real v2 = 0.0;
          Real b2 = 0.0;
          for (int n = 0; n < 3; n++) {
            const Real v = mom_or_vel(n * fft_size_inbox + idx);
            const Real b = mag(n * fft_size_inbox + idx);
            v2 += v * v;
            b2 += b * b;
          }
          pres_or_energy(idx) = gm1 * (pres_or_energy(idx) - 0.5 * rho(idx) * v2 - 0.5 * b2);
        });
    Kokkos::fence();
  }

  fields.is_conserved = false;
}

} // namespace energy_transfer
