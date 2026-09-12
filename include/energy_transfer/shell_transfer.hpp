#ifndef ENERGY_TRANSFER_SHELL_TRANSFER_HPP_
#define ENERGY_TRANSFER_SHELL_TRANSFER_HPP_

#include <map>
#include <string>
#include <vector>

#include <mesh/mesh.hpp>
#include <parameter_input.hpp>

#include "energy_transfer/flat_fields.hpp"
#include "energy_transfer/registry.hpp"

namespace energy_transfer {

// How shells are defined. Custom lets a caller supply arbitrary bin edges
// directly, with no library change needed.
struct BinningSpec {
  enum class Type { Linear, Log, Custom } type = Type::Linear;
  int num_shells = 20;            // ignored if type == Custom
  std::vector<Real> custom_edges; // used only if type == Custom

  static BinningSpec Linear(int n) { return {Type::Linear, n, {}}; }
  static BinningSpec Log(int n) { return {Type::Log, n, {}}; }
  static BinningSpec Custom(std::vector<Real> edges) {
    return {Type::Custom, 0, std::move(edges)};
  }
};

// Which shell axes are resolved per-shell vs collapsed to the whole domain,
// for EVERY term requested in a given ShellTransferConfig::terms (see
// ShellTransferConfig::mode below -- this is deliberately a single global
// setting, not per-term, so that ComputeEnergyTransfer can share one (Q,M,K)
// shell sweep across all requested terms instead of a separate sweep per
// term. That shared sweep is what keeps peak memory bounded to a small,
// fixed number of in-flight shell-filtered fields, independent of shell
// count or term count -- a per-term mode would let different terms want
// different sweep shapes in the same call, which is exactly what forced an
// earlier version of this code into an unbounded cache that grew for the
// whole call and caused real OOMs on long sweeps).
//
// "Q" is the donor/advecting shell (outer loop), "K" the receiving shell
// (inner loop), and "mediator" is the field each term's derived quantity
// reads directly (e.g. the advecting velocity U in UUA, or the tension
// field b in BUT) -- normally read full/unfiltered, but can optionally be
// shell-restricted too, independently of donor/receiver. Collapsing an axis
// evaluates that axis's derived quantity once over the whole domain
// (k_low=0, k_high=unrestricted) instead of once per real shell bin --
// shells partition Fourier space, so this reconstructs the unfiltered field
// directly rather than summing per-shell results after the fact.
//
// Not every term has a decomposable mediator: some terms' only mediator is
// a scalar normalization (density, via sqrt(rho) scaling) rather than a
// field being transported, and shell-restricting a scalar normalization
// isn't physically meaningful -- requesting mode.mediator_resolved=true
// while ShellTransferConfig::terms includes such a term (currently PU and
// FU) throws immediately, for the whole call, before any computation
// starts. See src/registry.cpp's DerivedQuantity::has_decomposable_mediator
// for the authoritative list.
struct DecompositionMode {
  bool donor_resolved = true;
  bool mediator_resolved = false;
  bool receiver_resolved = true;

  static DecompositionMode Full() { return {true, false, true}; }
  static DecompositionMode BySender() { return {true, false, false}; }
  static DecompositionMode ByReceiver() { return {false, false, true}; }
  static DecompositionMode Total() { return {false, false, false}; }
  static DecompositionMode FullWithMediator() { return {true, true, true}; }
  static DecompositionMode BySenderWithMediator() { return {true, true, false}; }
  static DecompositionMode ByReceiverWithMediator() { return {false, true, true}; }
  static DecompositionMode MediatorOnly() { return {false, true, false}; }
};

struct ShellTransferConfig {
  // Donor (Q), mediator, and receiver (K) each get their own independent
  // binning -- e.g. a wide, coarse custom band for Q and K to pin them at
  // two specific scales, while mediator uses fine Log() binning to sweep
  // finely across whichever scales actually mediate that Q->K transfer.
  // All three default to the same Linear(20) binning, matching this
  // library's original single-binning behavior.
  BinningSpec donor_binning = BinningSpec::Linear(20);
  BinningSpec mediator_binning = BinningSpec::Linear(20);
  BinningSpec receiver_binning = BinningSpec::Linear(20);
  std::vector<std::string> terms; // names from the library's fixed built-in set (BuiltinTerms())
  // ONE DecompositionMode shared by every term in `terms` above -- not
  // per-term. This is what lets ComputeEnergyTransfer share a single (Q,M,K)
  // shell sweep across all requested terms (see shell_transfer.cpp), which
  // in turn is what keeps peak memory bounded to a small, fixed number of
  // in-flight shell-filtered fields regardless of shell count or term count
  // -- see the "why global, not per-term" note on DecompositionMode above.
  // If mode.mediator_resolved is true and any requested term lacks a
  // decomposable mediator (its mediator, if any, is a scalar normalization
  // like rho -- currently PU and FU), the whole call throws immediately,
  // before any computation starts.
  DecompositionMode mode = DecompositionMode::Full();

  // Reads an <energy_transfer> input block: binning=lin|log|custom,
  // num_shells=, shell_edges= set a default binning shared by all three
  // axes; donor_binning=/mediator_binning=/receiver_binning= (each with its
  // own _num_shells=/_shell_edges=) override just that one axis when
  // present, falling back to the shared default otherwise. terms=UUA,BBA,BUT
  // is a plain comma-separated name list (no per-term mode suffix); mode=
  // (full/by_sender/by_receiver/total/full_mediator/by_sender_mediator/
  // by_receiver_mediator/mediator_only, default full) sets the one
  // DecompositionMode shared by all of them. Does NOT read spectra= --
  // that's a separate, independent concern, see spectra.hpp's
  // ParseSpectrumNames.
  static ShellTransferConfig FromInput(parthenon::ParameterInput *pin);
};

struct TransferResult {
  int n_donor_shells = 0;
  int n_mediator_shells = 0;
  int n_receiver_shells = 0;
  BinningSpec donor_binning;
  BinningSpec mediator_binning;
  BinningSpec receiver_binning;
  std::vector<Real> donor_edges;
  std::vector<Real> mediator_edges;
  std::vector<Real> receiver_edges;
  // keyed by term name; dims (n_q, n_m, n_k) -- shared by every term, since
  // ShellTransferConfig::mode is global (n_m is 1 unless mode.mediator_resolved
  // was set, matching how n_q/n_k collapse to 1 for an unresolved donor/receiver
  // side).
  std::map<std::string, parthenon::HostArray3D<TransferReal>> matrices;
  // Keyed by spectrum name -- left empty by ComputeEnergyTransfer (it knows
  // nothing about spectra); a caller wanting both fills this in itself from
  // ComputeSpectra's result (see spectra.hpp) before e.g. passing the result
  // to WriteResult.
  std::map<std::string, parthenon::HostArray2D<TransferReal>> spectra;
};

// Which real-space fields ingestion must load for the requested terms/
// spectra to be computable. Used to build a minimal LiveFieldSpec/
// FileFieldNaming before ingestion runs. spectrum_names is typically
// ParseSpectrumNames(pin) (see spectra.hpp) -- passed in explicitly rather
// than folded into ShellTransferConfig, since spectra are an independent
// concern from the shell-transfer terms.
struct FieldRequirements {
  bool mag = false;
  bool pres_or_energy = false;
  bool acc = false;
};
FieldRequirements ComputeFieldRequirements(const ShellTransferConfig &cfg,
                                           const std::vector<std::string> &spectrum_names);

// Core computation: fields must already be in primitive form (see
// ConvertConservedToPrimitive) and populated per ComputeFieldRequirements(cfg).
// Populates only TransferResult::matrices -- see spectra.hpp's ComputeSpectra
// for the independent, separately-callable spectra computation.
TransferResult ComputeEnergyTransfer(parthenon::Mesh *pmesh, FlatFields &fields,
                                     const ShellTransferConfig &cfg);

} // namespace energy_transfer

#endif // ENERGY_TRANSFER_SHELL_TRANSFER_HPP_
