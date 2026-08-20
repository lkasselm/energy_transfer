# Regression test: validates energy_transfer's C++ output against the upstream Python
# reference tool's own checked-in gold-standard fixture (github.com/pgrete/energy-
# transfer-analysis, testing/0024-All-Forc-Pres-testing-128_HDF-gold.pkl). See README.md
# in this directory for provenance.

import os
import pickle
import sys

import numpy as np
import utils.test_case

from . import fetch_and_convert

sys.dont_write_bytecode = True

# All 12 of this library's built-in terms match the gold pickle's term names exactly
# (same formulas, same signs -- verified by direct code comparison against the
# upstream tool's EnergyTransfer.py). Must match dd0024.in's `terms=` list.
TERMS = [
    "UUA", "UUC", "BBA", "BBC", "BUT", "UBTb",
    "UBTbA", "UBTbC", "BUPbb", "UBPbb", "PU", "FU",
]
# Reproduces the gold pickle's binning exactly (the upstream tool's "binning=test").
# Must match dd0024.in's `shell_edges=`.
SHELL_EDGES = [0.5, 1.5, 2.5, 16.0, 26.5, 28.5, 32.0]
# Same tolerance the upstream tool's own testing/runTest.py uses to compare two runs
# of itself. The C++ side uses exact spectral derivatives where the Python reference
# uses finite differences, so this may need revisiting once real numbers are seen --
# see this test's README.md.
RTOL = 1e-14
ATOL = 1e-7

GOLD_PKL_NAME = "gold_standard_0024.pkl"
OUTPUT_FILE = "gold_standard_dd0024_output.00000.bp"


def _shell_labels(edges):
    return [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(edges[:-1], edges[1:])]


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        cache_dir = os.environ.get(
            "ENERGY_TRANSFER_DD0024_CACHE",
            os.path.join(parameters.output_path, "dd0024_cache"),
        )
        try:
            bp_path = fetch_and_convert.ensure_dataset(cache_dir)
        except fetch_and_convert.DatasetUnavailable as e:
            # Not a test failure -- the dataset genuinely isn't available (no network,
            # missing yt, checksum mismatch, ...). SystemExit isn't caught by
            # run_test.py's `except Exception`, so this exits the whole process with
            # code 77, which CMake's SKIP_RETURN_CODE property turns into a clean
            # ctest skip rather than a failure.
            print(f"gold_standard_dd0024: skipping, dataset unavailable ({e})")
            sys.exit(77)

        parameters.driver_cmd_line_args = [f"energy_transfer/input_file={bp_path}"]
        return parameters

    def Analyse(self, parameters):
        try:
            import openpmd_api as io
        except ModuleNotFoundError:
            print("gold_standard_dd0024: openpmd_api not available, cannot analyse output.")
            return False

        output_path = os.path.join(parameters.output_path, OUTPUT_FILE)
        # ADIOS2's BP5 engine writes this as a directory (data.0, md.0, ...), not a
        # single file -- os.path.isfile() would always be False even on success.
        if not os.path.exists(output_path):
            print(f"gold_standard_dd0024: expected output file {output_path} not found "
                  "(did the driver crash? check the Run() output above).")
            return False

        gold_path = os.path.join(parameters.test_path, GOLD_PKL_NAME)
        with open(gold_path, "rb") as f:
            # The gold pickle was produced under Python 2.
            gold = pickle.load(f, encoding="latin1")

        labels = _shell_labels(SHELL_EDGES)

        series = io.Series(output_path, io.Access.read_only)
        it = series.iterations[0]
        cpp = {}
        for term in TERMS:
            cpp[term] = it.meshes[term][io.Mesh_Record_Component.SCALAR].load_chunk()
        series.flush()

        failures = []
        per_term_max_abs = {}
        per_term_max_rel = {}
        for term in TERMS:
            # On-disk shape is (K, Q) -- row=receiver shell, col=donor shell -- which
            # is deliberately the same convention as gold[...][KBin][QBin] (that's the
            # whole reason WriteResult transposes at the I/O boundary), so no reordering
            # is needed here.
            matrix = cpp[term]
            gold_term = gold["WW"][term]["AnyToAny"]
            max_abs = 0.0
            max_rel = 0.0
            for ki, kbin in enumerate(labels):
                for qi, qbin in enumerate(labels):
                    gold_val = gold_term[kbin][qbin]
                    cpp_val = matrix[ki, qi]
                    absdiff = abs(cpp_val - gold_val)
                    reldiff = absdiff / (abs(gold_val) if gold_val != 0 else 1.0)
                    max_abs = max(max_abs, absdiff)
                    max_rel = max(max_rel, reldiff)
                    try:
                        np.testing.assert_allclose(cpp_val, gold_val, rtol=RTOL, atol=ATOL)
                    except AssertionError:
                        failures.append((term, kbin, qbin, gold_val, cpp_val, absdiff, reldiff))
            per_term_max_abs[term] = max_abs
            per_term_max_rel[term] = max_rel

        # Bonus, non-gating cross-checks: the convenience sums UU=UUA+UUC, BB=BBA+BBC
        # also exist as their own keys in the gold pickle.
        for sum_name, parts in (("UU", ("UUA", "UUC")), ("BB", ("BBA", "BBC"))):
            summed = cpp[parts[0]] + cpp[parts[1]]
            gold_term = gold["WW"][sum_name]["AnyToAny"]
            bonus_max_abs = 0.0
            for ki, kbin in enumerate(labels):
                for qi, qbin in enumerate(labels):
                    bonus_max_abs = max(bonus_max_abs, abs(summed[ki, qi] - gold_term[kbin][qbin]))
            print(f"[bonus] {sum_name} = {'+'.join(parts)} vs gold: max abs diff = {bonus_max_abs:.3e}")

        print("\nPer-term max |diff| (informational, printed even on success -- watch this "
              "drift over time in CI history):")
        for term in TERMS:
            print(f"  {term:8s} max_abs={per_term_max_abs[term]:.3e}  max_rel={per_term_max_rel[term]:.3e}")

        if failures:
            failures.sort(key=lambda f: -f[5])
            total = len(TERMS) * len(labels) * len(labels)
            by_term = {}
            for f in failures:
                by_term[f[0]] = by_term.get(f[0], 0) + 1
            print(f"\ngold_standard_dd0024: {len(failures)}/{total} entries exceeded "
                  f"tolerance (rtol={RTOL}, atol={ATOL}) across {len(by_term)} terms: {by_term}")
            for term, kbin, qbin, gold_val, cpp_val, absdiff, reldiff in failures[:50]:
                print(f"  {term:8s} K={kbin:12s} Q={qbin:12s} gold={gold_val: .6e} "
                      f"cpp={cpp_val: .6e} absdiff={absdiff:.3e} reldiff={reldiff:.3e}")
            return False

        print("gold_standard_dd0024: all entries within tolerance. Good to go!")
        return True
