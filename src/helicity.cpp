#include "energy_transfer/helicity.hpp"

#include <cmath>

#include <kokkos_abstraction.hpp>
#include <utils/error_checking.hpp>

#include "energy_transfer/helmholtz.hpp"

namespace energy_transfer {

VectorPotentialFourier ComputeVectorPotentialFourier(parthenon::Mesh *pm,
                                                     const parthenon::ParArray1D<Real> &B) {
  // B's Coulomb-gauge vector potential is exactly the solenoidal potential of
  // its Helmholtz decomposition (A_hat = i k x B_hat / |k|^2, zero at k=0) --
  // B is divergence-free, so the gradient part is (numerically) zero and is
  // simply dropped here.
  auto h = ComputeHelmholtzFourier(pm, B);
  VectorPotentialFourier vp;
  vp.FT_A = h.FT_A;
  vp.FT_B = h.FT_F;
  return vp;
}

parthenon::ParArray1D<Real> CalcHelicity(parthenon::Mesh *pm, const parthenon::ParArray1D<Real> &B) {
  auto vp = ComputeVectorPotentialFourier(pm, B);

  auto FFTMgr = pm->GetFFTManager();
  const auto fft_size_inbox = FFTMgr->size_real_space_box();
  const auto fft_size_outbox = FFTMgr->size_fourier_space_box();

  parthenon::ParArray1D<Real> A("A_helicity", 3 * fft_size_inbox);
  for (int n = 0; n < 3; n++) {
    FFTMgr->Backward(vp.FT_A.data() + n * fft_size_outbox, A.data() + n * fft_size_inbox);
  }

  parthenon::ParArray1D<Real> H("H", fft_size_inbox);
  parthenon::par_for(
      "ComputeHelicityDensity", std::size_t(0), fft_size_inbox - 1,
      KOKKOS_LAMBDA(const std::size_t idx) {
        Real h = 0.0;
        for (int n = 0; n < 3; n++) h += A(n * fft_size_inbox + idx) * B(n * fft_size_inbox + idx);
        H(idx) = h;
      });
  Kokkos::fence();

  return H;
}

} // namespace energy_transfer
