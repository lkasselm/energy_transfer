#include "energy_transfer/registry.hpp"

#include <kokkos_abstraction.hpp>
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

// ---- Shell-filtering helpers ------------------------------------------
//
// Both take whichever ShellRestriction the caller means -- ws.axis (this
// quantity's own donor/receiver shell) or ws.mediator (the shell of the
// mediating field it reads). Donor, receiver, and mediator are on equal
// footing: whenever a restriction is inactive, field_full comes back
// completely unchanged (no extra FFT/filter work, DC mode included), and
// the caller never has to know which axis it was.

// Vector (3-component) field with an already-computed full-domain FT
// (FT_W/FT_U/FT_B/FT_b/FT_Acc) -- shell-filters it to (low, high].
parthenon::ParArray1D<Real>
FilterVector(const ShellWorkspace &ws, const ShellRestriction &restriction, const char *name,
             const parthenon::ParArray1D<Kokkos::complex<Real>> &FT_field_full,
             const parthenon::ParArray1D<Real> &field_full) {
  if (!restriction.active) return field_full;
  auto out = AllocVec(name, ws.fft_size_inbox);
  parthenon::ParArray1D<Kokkos::complex<Real>> scratch(std::string(name) + "_scratch",
                                                        3 * ws.fft_size_outbox);
  ShellFilter(ws.fft_mgr, 3, FT_field_full, scratch, out, restriction.low, restriction.high);
  return out;
}

// Scalar field with no persistent FT array (div(U), div(b)) -- forward-
// transforms field_full once, then shell-filters, mirroring the inline
// Forward()-then-ShellFilter idiom also used by DivbW/GradBdotBQScaled for
// their own on-the-fly spectral work.
parthenon::ParArray1D<Real> FilterScalar(const ShellWorkspace &ws,
                                         const ShellRestriction &restriction, const char *name,
                                         const parthenon::ParArray1D<Real> &field_full) {
  if (!restriction.active) return field_full;
  const auto n = ws.fft_size_inbox;
  const auto nout = ws.fft_size_outbox;
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_scratch(std::string(name) + "_FT", nout);
  ws.fft_mgr->Forward(field_full.data(), FT_scratch.data());
  parthenon::ParArray1D<Real> out(name, n);
  parthenon::ParArray1D<Kokkos::complex<Real>> filter_scratch(std::string(name) + "_scratch",
                                                               nout);
  ShellFilter(ws.fft_mgr, 1, FT_scratch, filter_scratch, out, restriction.low, restriction.high);
  return out;
}

// ---- Level 1: derived-quantity providers -----------------------------

parthenon::ParArray1D<Real> WFilter(const ShellWorkspace &ws) {
  return FilterVector(ws, ws.axis, "W_filter", ws.ft->FT_W, ws.aux->W_flat);
}

parthenon::ParArray1D<Real> BFilter(const ShellWorkspace &ws) {
  return FilterVector(ws, ws.axis, "B_filter", ws.ft->FT_B, ws.fields->mag);
}

parthenon::ParArray1D<Real> AccFilterTimesSqrtRho(const ShellWorkspace &ws) {
  // FilterVector aliases fields->acc when the axis is inactive, so the
  // sqrt(rho) scaling must go into a fresh buffer rather than in place.
  auto filtered = FilterVector(ws, ws.axis, "Acc_filter", ws.ft->FT_Acc, ws.fields->acc);
  const auto n = ws.fft_size_inbox;
  auto out = AllocVec("Acc_filter_scaled", n);
  auto rho = ws.fields->rho;
  parthenon::par_for(
      "ScaleAccBySqrtRho", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
        const Real s = Kokkos::sqrt(rho(idx));
        for (int c = 0; c < 3; c++) out(c * n + idx) = filtered(c * n + idx) * s;
      });
  return out;
}

parthenon::ParArray1D<Real> UdotGradW(const ShellWorkspace &ws) {
  const auto n = ws.fft_size_inbox;
  auto out = AllocVec("UdotGradW", n);
  ZeroVec(out, 3 * n);
  parthenon::ParArray1D<Kokkos::complex<Real>> scratch("UdotGradW_scratch", ws.fft_size_outbox);
  parthenon::ParArray1D<Real> deriv("UdotGradW_deriv", n);
  auto vel = FilterVector(ws, ws.mediator, "UdotGradW_U_med", ws.ft->FT_U, ws.fields->mom_or_vel);
  for (int comp_i = 0; comp_i < 3; comp_i++) {
    for (int dir_j = 0; dir_j < 3; dir_j++) {
      ShellFilterDerivative(ws.fft_mgr, ws.ft->FT_W, comp_i * ws.fft_size_outbox, scratch, 0,
                            deriv, 0, ws.axis.low, ws.axis.high, dir_j, ws.two_pi_over_L);
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
  auto vel = FilterVector(ws, ws.mediator, "UdotGradB_U_med", ws.ft->FT_U, ws.fields->mom_or_vel);
  for (int comp_i = 0; comp_i < 3; comp_i++) {
    for (int dir_j = 0; dir_j < 3; dir_j++) {
      ShellFilterDerivative(ws.fft_mgr, ws.ft->FT_B, comp_i * ws.fft_size_outbox, scratch, 0,
                            deriv, 0, ws.axis.low, ws.axis.high, dir_j, ws.two_pi_over_L);
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

// Magnetic helicity transfer mediator quantity: U x B_Q (own field is B,
// filtered to ws.axis; mediator is U, filtered to ws.mediator) -- see the
// "H" term below, e.g. doi:10.1017/jfm.2021.496 eq. (4.1). Linear in U, so
// (like U_dot_grad_W/U_dot_grad_B) shell-decomposing the mediator and
// summing over shells reconstructs the unfiltered-mediator result exactly.
parthenon::ParArray1D<Real> UCrossB(const ShellWorkspace &ws) {
  const auto n = ws.fft_size_inbox;
  auto out = AllocVec("UCrossB", n);
  auto B = FilterVector(ws, ws.axis, "UCrossB_B", ws.ft->FT_B, ws.fields->mag);
  auto U = FilterVector(ws, ws.mediator, "UCrossB_U_med", ws.ft->FT_U, ws.fields->mom_or_vel);
  parthenon::par_for(
      "UCrossB", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
        const Real Ux = U(0 * n + idx), Uy = U(1 * n + idx), Uz = U(2 * n + idx);
        const Real Bx = B(0 * n + idx), By = B(1 * n + idx), Bz = B(2 * n + idx);
        out(0 * n + idx) = Uy * Bz - Uz * By;
        out(1 * n + idx) = Uz * Bx - Ux * Bz;
        out(2 * n + idx) = Ux * By - Uy * Bx;
      });
  return out;
}

parthenon::ParArray1D<Real> BDotGradB(const ShellWorkspace &ws) {
  const auto n = ws.fft_size_inbox;
  auto out = AllocVec("bDotGradB", n);
  ZeroVec(out, 3 * n);
  parthenon::ParArray1D<Kokkos::complex<Real>> scratch("bDotGradB_scratch", ws.fft_size_outbox);
  parthenon::ParArray1D<Real> deriv("bDotGradB_deriv", n);
  auto b = FilterVector(ws, ws.mediator, "BDotGradB_b_med", ws.ft->FT_b, ws.aux->b_flat);
  for (int comp_i = 0; comp_i < 3; comp_i++) {
    for (int dir_j = 0; dir_j < 3; dir_j++) {
      ShellFilterDerivative(ws.fft_mgr, ws.ft->FT_B, comp_i * ws.fft_size_outbox, scratch, 0,
                            deriv, 0, ws.axis.low, ws.axis.high, dir_j, ws.two_pi_over_L);
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
  auto b = FilterVector(ws, ws.mediator, "BDotGradW_b_med", ws.ft->FT_b, ws.aux->b_flat);
  for (int comp_i = 0; comp_i < 3; comp_i++) {
    for (int dir_j = 0; dir_j < 3; dir_j++) {
      ShellFilterDerivative(ws.fft_mgr, ws.ft->FT_W, comp_i * ws.fft_size_outbox, scratch, 0,
                            deriv, 0, ws.axis.low, ws.axis.high, dir_j, ws.two_pi_over_L);
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
  auto divu = FilterScalar(ws, ws.mediator, "WTimesDivU_DivU_med", ws.aux->DivU);
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
  auto divu = FilterScalar(ws, ws.mediator, "BTimesDivU_DivU_med", ws.aux->DivU);
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
  auto divb = FilterScalar(ws, ws.mediator, "WTimesDivb_Divb_med", ws.aux->Divb);
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
  auto b = FilterVector(ws, ws.mediator, "DivbW_b_med", ws.ft->FT_b, ws.aux->b_flat);
  auto out = AllocVec("DivbW", n);
  ZeroVec(out, 3 * n);
  parthenon::ParArray1D<Real> scalar_scratch("DivbW_scalar_scratch", n);
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_scalar_scratch("DivbW_FT_scalar_scratch",
                                                                 nout);
  parthenon::ParArray1D<Kokkos::complex<Real>> deriv_scratch("DivbW_deriv_scratch", nout);
  parthenon::ParArray1D<Real> deriv("DivbW_deriv", n);
  const Real huge_k = NoRestrictionKHigh(ws.Nx, ws.Ny, ws.Nz);
  for (int comp_i = 0; comp_i < 3; comp_i++) {
    for (int dir_j = 0; dir_j < 3; dir_j++) {
      const std::size_t b_offset = dir_j * n;
      const std::size_t w_offset = comp_i * n;
      parthenon::par_for(
          "DivbW_product", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
            scalar_scratch(idx) = b(b_offset + idx) * w(w_offset + idx);
          });
      ws.fft_mgr->Forward(scalar_scratch.data(), FT_scalar_scratch.data());
      ShellFilterDerivative(ws.fft_mgr, FT_scalar_scratch, 0, deriv_scratch, 0, deriv, 0,
                            kNoRestrictionKLow, huge_k, dir_j, ws.two_pi_over_L);
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
  auto mag = FilterVector(ws, ws.mediator, "GradBdotBQ_B_med", ws.ft->FT_B, ws.fields->mag);
  auto rho = ws.fields->rho; // scalar normalization, not a decomposable mediator
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
  const Real huge_k = NoRestrictionKHigh(ws.Nx, ws.Ny, ws.Nz);
  for (int dir_j = 0; dir_j < 3; dir_j++) {
    ShellFilterDerivative(ws.fft_mgr, FT_scalar, 0, deriv_scratch, 0, out, dir_j * n,
                          kNoRestrictionKLow, huge_k, dir_j, ws.two_pi_over_L);
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
  auto mag = FilterVector(ws, ws.mediator, "BTimesMag_B_med", ws.ft->FT_B, ws.fields->mag);
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
    ShellFilterDerivative(ws.fft_mgr, ws.ft->FT_P, 0, scratch, 0, out, dir_j * n, ws.axis.low,
                          ws.axis.high, dir_j, ws.two_pi_over_L);
  }
  auto rho = ws.fields->rho;
  parthenon::par_for(
      "ScaleGradP", std::size_t(0), n - 1, KOKKOS_LAMBDA(const std::size_t idx) {
        const Real inv_sqrt_rho = 1.0 / Kokkos::sqrt(rho(idx));
        for (int c = 0; c < 3; c++) out(c * n + idx) *= inv_sqrt_rho;
      });
  return out;
}

} // namespace

const std::map<std::string, DerivedQuantity> &BuiltinQuantities() {
  static const std::map<std::string, DerivedQuantity> table = {
      // Pure shell filters -- no mediator at all.
      {"W_filter", {{}, {}, false, &WFilter}},
      {"B_filter", {{"B"}, {}, false, &BFilter}},
      // Mediator is rho (a scalar normalization, not decomposable).
      {"Acc_filter_times_sqrt_rho", {{"Acc"}, {}, false, &AccFilterTimesSqrtRho}},
      {"Div_WOverSqrtRho_broadcast", {{}, {}, false, &DivWOverSqrtRhoBroadcast}},
      {"grad_P_over_sqrt_rho", {{"P"}, {}, false, &GradPOverSqrtRho}},
      // Mediator is U -- doesn't otherwise need FT_U (donor field is FT_W),
      // so it's only pulled in via mediator_only_base_fields.
      {"U_dot_grad_W", {{}, {"U"}, true, &UdotGradW}},
      {"U_dot_grad_B", {{"B"}, {"U"}, true, &UdotGradB}},
      // Mediator is b -- FT_b is always computed alongside b_flat whenever
      // "b" is required, so no extra mediator_only_base_fields tag needed.
      {"b_dot_grad_B", {{"B", "b"}, {}, true, &BDotGradB}},
      {"b_dot_grad_W", {{"b"}, {}, true, &BDotGradW}},
      {"Div_bW", {{"b"}, {}, true, &DivbW}},
      // Mediator is div(U)/div(b) -- DivU/Divb are already required, and the
      // mediator filter forward-transforms them on the fly.
      {"W_times_DivU", {{"U", "DivU"}, {}, true, &WTimesDivU}},
      {"B_times_DivU", {{"B", "U", "DivU"}, {}, true, &BTimesDivU}},
      {"W_times_Divb", {{"b", "Divb"}, {}, true, &WTimesDivb}},
      // Mediator is B -- FT_B is already required, so no extra tag needed.
      {"grad_BdotBQ_scaled", {{"B"}, {}, true, &GradBdotBQScaled}},
      {"B_times_mag", {{"B"}, {}, true, &BTimesMag}},
      // Mediator is U -- own field is B, same shape as U_dot_grad_B.
      {"U_cross_B", {{"B"}, {"U"}, true, &UCrossB}},
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
      // T_H(Q,K) = 2 * <B_filter_K, U_cross_B_Q> -- magnetic helicity
      // transfer, e.g. doi:10.1017/jfm.2021.496 eq. (4.1).
      {"H", {"U_cross_B", "B_filter", 2.0}},
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
