#include "energy_transfer/registry.hpp"

#include <kokkos_abstraction.hpp>
#include <utils/calc_spectrum.hpp>
#include <utils/error_checking.hpp>

#include "energy_transfer/spectral_kernels.hpp"

namespace energy_transfer {

namespace {

parthenon::ParArray1D<Real> AllocVec(const char *name, std::size_t n) {
  return parthenon::ParArray1D<Real>(name, 3 * n);
}

void ZeroVec(parthenon::ParArray1D<Real> &v, std::size_t n3) {
  Kokkos::deep_copy(
      Kokkos::View<Real *, Kokkos::DefaultExecutionSpace::memory_space>(v.data(), n3), 0.0);
}

Real UnrestrictedKHigh(const ShellWorkspace &ws) { return Real(ws.Nx + ws.Ny + ws.Nz); }

// ---- Level 1: derived-quantity providers -----------------------------

parthenon::ParArray1D<Real> WFilter(const ShellWorkspace &ws) {
  auto out = AllocVec("W_filter", ws.fft_size_inbox);
  parthenon::ParArray1D<Kokkos::complex<Real>> scratch("W_filter_scratch",
                                                        3 * ws.fft_size_outbox);
  ShellFilter(ws.fft_mgr, 3, ws.ft->FT_W, scratch, out, ws.k_low, ws.k_high);
  return out;
}

parthenon::ParArray1D<Real> BFilter(const ShellWorkspace &ws) {
  auto out = AllocVec("B_filter", ws.fft_size_inbox);
  parthenon::ParArray1D<Kokkos::complex<Real>> scratch("B_filter_scratch",
                                                        3 * ws.fft_size_outbox);
  ShellFilter(ws.fft_mgr, 3, ws.ft->FT_B, scratch, out, ws.k_low, ws.k_high);
  return out;
}

parthenon::ParArray1D<Real> AccFilterTimesSqrtRho(const ShellWorkspace &ws) {
  auto out = AllocVec("Acc_filter", ws.fft_size_inbox);
  parthenon::ParArray1D<Kokkos::complex<Real>> scratch("Acc_filter_scratch",
                                                        3 * ws.fft_size_outbox);
  ShellFilter(ws.fft_mgr, 3, ws.ft->FT_Acc, scratch, out, ws.k_low, ws.k_high);
  const auto n = ws.fft_size_inbox;
  auto rho = ws.fields->rho;
  parthenon::par_for(
      "ScaleAccBySqrtRho", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
        const Real s = Kokkos::sqrt(rho(idx));
        for (int c = 0; c < 3; c++) out(c * n + idx) *= s;
      });
  return out;
}

parthenon::ParArray1D<Real> UdotGradW(const ShellWorkspace &ws) {
  const auto n = ws.fft_size_inbox;
  auto out = AllocVec("UdotGradW", n);
  ZeroVec(out, 3 * n);
  parthenon::ParArray1D<Kokkos::complex<Real>> scratch("UdotGradW_scratch", ws.fft_size_outbox);
  parthenon::ParArray1D<Real> deriv("UdotGradW_deriv", n);
  auto vel = ws.fields->mom_or_vel;
  for (int comp_i = 0; comp_i < 3; comp_i++) {
    for (int dir_j = 0; dir_j < 3; dir_j++) {
      ShellFilterDerivative(ws.fft_mgr, ws.ft->FT_W, comp_i * ws.fft_size_outbox, scratch, 0,
                            deriv, 0, ws.k_low, ws.k_high, dir_j, ws.two_pi_over_L);
      const std::size_t vel_offset = dir_j * n;
      const std::size_t out_offset = comp_i * n;
      parthenon::par_for(
          "AccumUdotGradW", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
            out(out_offset + idx) += vel(vel_offset + idx) * deriv(idx);
          });
    }
  }
  return out;
}

parthenon::ParArray1D<Real> UdotGradB(const ShellWorkspace &ws) {
  const auto n = ws.fft_size_inbox;
  auto out = AllocVec("UdotGradB", n);
  ZeroVec(out, 3 * n);
  parthenon::ParArray1D<Kokkos::complex<Real>> scratch("UdotGradB_scratch", ws.fft_size_outbox);
  parthenon::ParArray1D<Real> deriv("UdotGradB_deriv", n);
  auto vel = ws.fields->mom_or_vel;
  for (int comp_i = 0; comp_i < 3; comp_i++) {
    for (int dir_j = 0; dir_j < 3; dir_j++) {
      ShellFilterDerivative(ws.fft_mgr, ws.ft->FT_B, comp_i * ws.fft_size_outbox, scratch, 0,
                            deriv, 0, ws.k_low, ws.k_high, dir_j, ws.two_pi_over_L);
      const std::size_t vel_offset = dir_j * n;
      const std::size_t out_offset = comp_i * n;
      parthenon::par_for(
          "AccumUdotGradB", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
            out(out_offset + idx) += vel(vel_offset + idx) * deriv(idx);
          });
    }
  }
  return out;
}

parthenon::ParArray1D<Real> BDotGradB(const ShellWorkspace &ws) {
  const auto n = ws.fft_size_inbox;
  auto out = AllocVec("bDotGradB", n);
  ZeroVec(out, 3 * n);
  parthenon::ParArray1D<Kokkos::complex<Real>> scratch("bDotGradB_scratch", ws.fft_size_outbox);
  parthenon::ParArray1D<Real> deriv("bDotGradB_deriv", n);
  auto b = ws.aux->b_flat;
  for (int comp_i = 0; comp_i < 3; comp_i++) {
    for (int dir_j = 0; dir_j < 3; dir_j++) {
      ShellFilterDerivative(ws.fft_mgr, ws.ft->FT_B, comp_i * ws.fft_size_outbox, scratch, 0,
                            deriv, 0, ws.k_low, ws.k_high, dir_j, ws.two_pi_over_L);
      const std::size_t b_offset = dir_j * n;
      const std::size_t out_offset = comp_i * n;
      parthenon::par_for(
          "AccumBDotGradB", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
            out(out_offset + idx) += b(b_offset + idx) * deriv(idx);
          });
    }
  }
  return out;
}

parthenon::ParArray1D<Real> BDotGradW(const ShellWorkspace &ws) {
  const auto n = ws.fft_size_inbox;
  auto out = AllocVec("bDotGradW", n);
  ZeroVec(out, 3 * n);
  parthenon::ParArray1D<Kokkos::complex<Real>> scratch("bDotGradW_scratch", ws.fft_size_outbox);
  parthenon::ParArray1D<Real> deriv("bDotGradW_deriv", n);
  auto b = ws.aux->b_flat;
  for (int comp_i = 0; comp_i < 3; comp_i++) {
    for (int dir_j = 0; dir_j < 3; dir_j++) {
      ShellFilterDerivative(ws.fft_mgr, ws.ft->FT_W, comp_i * ws.fft_size_outbox, scratch, 0,
                            deriv, 0, ws.k_low, ws.k_high, dir_j, ws.two_pi_over_L);
      const std::size_t b_offset = dir_j * n;
      const std::size_t out_offset = comp_i * n;
      parthenon::par_for(
          "AccumBDotGradW", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
            out(out_offset + idx) += b(b_offset + idx) * deriv(idx);
          });
    }
  }
  return out;
}

parthenon::ParArray1D<Real> WTimesDivU(const ShellWorkspace &ws) {
  auto w = WFilter(ws);
  const auto n = ws.fft_size_inbox;
  auto divu = ws.aux->DivU;
  auto out = AllocVec("W_times_DivU", n);
  parthenon::par_for(
      "WTimesDivU", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
        const Real d = divu(idx);
        for (int c = 0; c < 3; c++) out(c * n + idx) = w(c * n + idx) * d;
      });
  return out;
}

parthenon::ParArray1D<Real> BTimesDivU(const ShellWorkspace &ws) {
  auto b = BFilter(ws);
  const auto n = ws.fft_size_inbox;
  auto divu = ws.aux->DivU;
  auto out = AllocVec("B_times_DivU", n);
  parthenon::par_for(
      "BTimesDivU", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
        const Real d = divu(idx);
        for (int c = 0; c < 3; c++) out(c * n + idx) = b(c * n + idx) * d;
      });
  return out;
}

parthenon::ParArray1D<Real> WTimesDivb(const ShellWorkspace &ws) {
  auto w = WFilter(ws);
  const auto n = ws.fft_size_inbox;
  auto divb = ws.aux->Divb;
  auto out = AllocVec("W_times_Divb", n);
  parthenon::par_for(
      "WTimesDivb", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
        const Real d = divb(idx);
        for (int c = 0; c < 3; c++) out(c * n + idx) = w(c * n + idx) * d;
      });
  return out;
}

// div(b (x) W_Q): component i = sum_j d/dx_j (b_j * W_Q_i), no shell
// restriction on the derivative itself (the shell restriction already
// happened when W_Q was formed). Equals bDotGradW + W_times_Divb by the
// product rule -- used as a cross-check against those two terms summed.
parthenon::ParArray1D<Real> DivbW(const ShellWorkspace &ws) {
  const auto n = ws.fft_size_inbox;
  const auto nout = ws.fft_size_outbox;
  auto w = WFilter(ws);
  auto b = ws.aux->b_flat;
  auto out = AllocVec("DivbW", n);
  ZeroVec(out, 3 * n);
  parthenon::ParArray1D<Real> scalar_scratch("DivbW_scalar_scratch", n);
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_scalar_scratch("DivbW_FT_scalar_scratch",
                                                                 nout);
  parthenon::ParArray1D<Kokkos::complex<Real>> deriv_scratch("DivbW_deriv_scratch", nout);
  parthenon::ParArray1D<Real> deriv("DivbW_deriv", n);
  const Real huge_k = UnrestrictedKHigh(ws);
  for (int comp_i = 0; comp_i < 3; comp_i++) {
    for (int dir_j = 0; dir_j < 3; dir_j++) {
      const std::size_t b_offset = dir_j * n;
      const std::size_t w_offset = comp_i * n;
      parthenon::par_for(
          "DivbW_product", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
            scalar_scratch(idx) = b(b_offset + idx) * w(w_offset + idx);
          });
      ws.fft_mgr->Forward(scalar_scratch.data(), FT_scalar_scratch.data());
      ShellFilterDerivative(ws.fft_mgr, FT_scalar_scratch, 0, deriv_scratch, 0, deriv, 0, -1.0,
                            huge_k, dir_j, ws.two_pi_over_L);
      const std::size_t out_offset = comp_i * n;
      parthenon::par_for(
          "DivbW_accum", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
            out(out_offset + idx) += deriv(idx);
          });
    }
  }
  return out;
}

parthenon::ParArray1D<Real> GradBdotBQScaled(const ShellWorkspace &ws) {
  const auto n = ws.fft_size_inbox;
  const auto nout = ws.fft_size_outbox;
  auto bq = BFilter(ws);
  auto mag = ws.fields->mag;
  auto rho = ws.fields->rho;
  parthenon::ParArray1D<Real> scalar_scratch("GradBdotBQ_scalar", n);
  parthenon::par_for(
      "BdotBQ", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
        Real s = 0.0;
        for (int c = 0; c < 3; c++) s += mag(c * n + idx) * bq(c * n + idx);
        scalar_scratch(idx) = s;
      });
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_scalar("GradBdotBQ_FT", nout);
  ws.fft_mgr->Forward(scalar_scratch.data(), FT_scalar.data());
  auto out = AllocVec("GradBdotBQ", n);
  parthenon::ParArray1D<Kokkos::complex<Real>> deriv_scratch("GradBdotBQ_deriv_scratch", nout);
  const Real huge_k = UnrestrictedKHigh(ws);
  for (int dir_j = 0; dir_j < 3; dir_j++) {
    ShellFilterDerivative(ws.fft_mgr, FT_scalar, 0, deriv_scratch, 0, out, dir_j * n, -1.0,
                          huge_k, dir_j, ws.two_pi_over_L);
  }
  parthenon::par_for(
      "ScaleGradBdotBQ", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
        const Real scale = 0.5 / Kokkos::sqrt(rho(idx));
        for (int c = 0; c < 3; c++) out(c * n + idx) *= scale;
      });
  return out;
}

parthenon::ParArray1D<Real> BTimesMag(const ShellWorkspace &ws) {
  auto b = BFilter(ws);
  const auto n = ws.fft_size_inbox;
  auto mag = ws.fields->mag;
  auto out = AllocVec("B_times_mag", n);
  parthenon::par_for(
      "BTimesMag", std::size_t(0), 3 * n - 1,
      KOKKOS_LAMBDA(const std::size_t idx) { out(idx) = b(idx) * mag(idx); });
  return out;
}

parthenon::ParArray1D<Real> DivWOverSqrtRhoBroadcast(const ShellWorkspace &ws) {
  auto w = WFilter(ws);
  const auto n = ws.fft_size_inbox;
  const auto nout = ws.fft_size_outbox;
  auto rho = ws.fields->rho;
  parthenon::ParArray1D<Real> w_over_sqrt_rho("WOverSqrtRho", 3 * n);
  parthenon::par_for(
      "ScaleWBySqrtRho", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
        const Real scale = 0.5 / Kokkos::sqrt(rho(idx));
        for (int c = 0; c < 3; c++) w_over_sqrt_rho(c * n + idx) = scale * w(c * n + idx);
      });
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_vec("WOverSqrtRho_FT", 3 * nout);
  for (int c = 0; c < 3; c++) {
    ws.fft_mgr->Forward(w_over_sqrt_rho.data() + c * n, FT_vec.data() + c * nout);
  }
  parthenon::ParArray1D<Real> div_scalar("DivWOverSqrtRho", n);
  parthenon::ParArray1D<Kokkos::complex<Real>> scratch("DivWOverSqrtRho_scratch", nout);
  SpectralDivergence(ws.fft_mgr, FT_vec, scratch, div_scalar, ws.two_pi_over_L);
  auto out = AllocVec("DivWOverSqrtRho_bcast", n);
  parthenon::par_for(
      "BroadcastDivWOverSqrtRho", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
        const Real v = div_scalar(idx);
        for (int c = 0; c < 3; c++) out(c * n + idx) = v;
      });
  return out;
}

parthenon::ParArray1D<Real> GradPOverSqrtRho(const ShellWorkspace &ws) {
  const auto n = ws.fft_size_inbox;
  const auto nout = ws.fft_size_outbox;
  auto out = AllocVec("gradP", n);
  parthenon::ParArray1D<Kokkos::complex<Real>> scratch("gradP_scratch", nout);
  for (int dir_j = 0; dir_j < 3; dir_j++) {
    ShellFilterDerivative(ws.fft_mgr, ws.ft->FT_P, 0, scratch, 0, out, dir_j * n, ws.k_low,
                          ws.k_high, dir_j, ws.two_pi_over_L);
  }
  auto rho = ws.fields->rho;
  parthenon::par_for(
      "ScaleGradP", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
        const Real inv_sqrt_rho = 1.0 / Kokkos::sqrt(rho(idx));
        for (int c = 0; c < 3; c++) out(c * n + idx) *= inv_sqrt_rho;
      });
  return out;
}

// ---- Spectra -----------------------------------------------------------

parthenon::HostArray2D<TransferReal> SpecU(parthenon::Mesh *pm, const FlatFields &f,
                                           const parthenon::ParArray1D<Real> & /*W_flat*/) {
  auto spectra = parthenon::utils::fft::CalcSpectrum(pm, f.mom_or_vel, 3);
  return spectra.GetHostMirrorAndCopy();
}

parthenon::HostArray2D<TransferReal> SpecRho(parthenon::Mesh *pm, const FlatFields &f,
                                             const parthenon::ParArray1D<Real> & /*W_flat*/) {
  auto spectra = parthenon::utils::fft::CalcSpectrum(pm, f.rho, 1);
  return spectra.GetHostMirrorAndCopy();
}

parthenon::HostArray2D<TransferReal> SpecW(parthenon::Mesh *pm, const FlatFields & /*f*/,
                                           const parthenon::ParArray1D<Real> &W_flat) {
  auto spectra = parthenon::utils::fft::CalcSpectrum(pm, W_flat, 3);
  return spectra.GetHostMirrorAndCopy();
}

parthenon::HostArray2D<TransferReal> SpecB(parthenon::Mesh *pm, const FlatFields &f,
                                           const parthenon::ParArray1D<Real> & /*W_flat*/) {
  auto spectra = parthenon::utils::fft::CalcSpectrum(pm, f.mag, 3);
  return spectra.GetHostMirrorAndCopy();
}

} // namespace

const std::map<std::string, DerivedQuantity> &BuiltinQuantities() {
  static const std::map<std::string, DerivedQuantity> table = {
      {"W_filter", {{}, &WFilter}},
      {"B_filter", {{"B"}, &BFilter}},
      {"Acc_filter_times_sqrt_rho", {{"Acc"}, &AccFilterTimesSqrtRho}},
      {"U_dot_grad_W", {{}, &UdotGradW}},
      {"U_dot_grad_B", {{"B"}, &UdotGradB}},
      {"b_dot_grad_B", {{"B", "b"}, &BDotGradB}},
      {"b_dot_grad_W", {{"b"}, &BDotGradW}},
      {"W_times_DivU", {{"U", "DivU"}, &WTimesDivU}},
      {"B_times_DivU", {{"B", "U", "DivU"}, &BTimesDivU}},
      {"W_times_Divb", {{"b", "Divb"}, &WTimesDivb}},
      {"Div_bW", {{"b"}, &DivbW}},
      {"grad_BdotBQ_scaled", {{"B"}, &GradBdotBQScaled}},
      {"B_times_mag", {{"B"}, &BTimesMag}},
      {"Div_WOverSqrtRho_broadcast", {{}, &DivWOverSqrtRhoBroadcast}},
      {"grad_P_over_sqrt_rho", {{"P"}, &GradPOverSqrtRho}},
  };
  return table;
}

const std::map<std::string, TransferTerm> &BuiltinTerms() {
  static const std::map<std::string, TransferTerm> table = {
      {"UUA", {"U_dot_grad_W", "W_filter", -1.0}},
      {"UUC", {"W_times_DivU", "W_filter", -0.5}},
      {"BBA", {"U_dot_grad_B", "B_filter", -1.0}},
      {"BBC", {"B_times_DivU", "B_filter", -0.5}},
      {"BUT", {"b_dot_grad_B", "W_filter", 1.0}},
      {"UBTb", {"Div_bW", "B_filter", 1.0}},
      {"UBTbA", {"b_dot_grad_W", "B_filter", 1.0}},
      {"UBTbC", {"W_times_Divb", "B_filter", 1.0}},
      {"BUPbb", {"grad_BdotBQ_scaled", "W_filter", -1.0}},
      {"UBPbb", {"Div_WOverSqrtRho_broadcast", "B_times_mag", -1.0}},
      {"PU", {"grad_P_over_sqrt_rho", "W_filter", -1.0}},
      {"FU", {"Acc_filter_times_sqrt_rho", "W_filter", 1.0}},
  };
  return table;
}

const std::map<std::string, Spectrum> &BuiltinSpectra() {
  static const std::map<std::string, Spectrum> table = {
      {"spec_U", {&SpecU}},
      {"spec_rho", {&SpecRho}},
      {"spec_W", {&SpecW}},
      {"spec_B", {&SpecB}},
  };
  return table;
}

TransferReal DotProductReduce(const parthenon::ParArray1D<Real> &a,
                              const parthenon::ParArray1D<Real> &b, std::size_t n) {
  TransferReal local = 0.0;
  Kokkos::parallel_reduce(
      "DotProductReduce", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(const std::size_t idx, TransferReal &sum) {
        sum += static_cast<TransferReal>(a(idx)) * static_cast<TransferReal>(b(idx));
      },
      Kokkos::Sum<TransferReal>(local));

  TransferReal global = local;
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
#endif
  return global;
}

} // namespace energy_transfer
