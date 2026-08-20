"""Assert that appending R7 to ``dgp.RULES`` changed no E0 golden value.

Run: ``uv run python scripts/export_e0_r7_invariance.py``

The instinct on a freeze-line change is to re-run the record and write down new golden
values. That would be wrong here, and this script is what makes "wrong" checkable rather
than asserted. ``evaluate_rules`` consumes no RNG and runs after all sampling;
``solve_binding_rates`` only indexes ``RULES`` by name, for the R3/R4 antecedents. A rule
*addition* therefore cannot perturb the sampler, so every pre-existing E0 quantity must
come back **bit-identical** -- the E0 table gains one row for R7 and no existing cell is
edited.

Comparison is exact ``==`` / ``DataFrame.equals`` / ``np.array_equal`` throughout, never
``np.isclose``. The claim is bit-identity; a tolerance here would hide precisely the
perturbation the script exists to rule out.

The last check is structural rather than numeric, and is stronger than any tolerance band
on either binding rate: R1 binds on non-smokers and R7 on smokers, so **every row binds
exactly one of the two**. ``n_R1_bound + n_R7_bound == n`` exactly, and the two masks are
disjoint.

A failure here is a finding, not something to tune away. It would mean either that the
mechanism argument above is wrong or that something reads ``RULES`` in a way this did not
anticipate, and either is worth more than a green check.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import dgp
from dgp import DGPConfig, generate, solve_binding_rates
from export_s0 import _provenance  # noqa: E402  -- the repo's one provenance convention

RESULTS = ROOT / "results"
MANIFESTS = ROOT / "manifests"

E0_TARGETS = {"R3_glucose_ceil": 0.08, "R4_bp_mandatory": 0.20}
N_PROBE = 40_000
N_SAMPLES = 50_000
SEED = 0

R7 = "R7_smoker_cigs_min"
R1 = "R1_smoker_cigs"
SHARED = ["R1_smoker_cigs", "R2_glucose_floor", "R3_glucose_ceil",
          "R4_bp_mandatory", "R5_anatomical", "R6_preg_age"]

SCALARS = ["glucose_base", "sbp_base", "solved_gamma", "solved_alpha",
           "realised_prevalence", "expected_bayes_auc"]

# Stamp provenance before anything is written -- _provenance() reads `git status`, so
# calling it after the CSV lands reports the tree this run itself dirtied.
PROV = _provenance()


@contextmanager
def _without_r7():
    """Run the body with R7 removed from ``dgp.RULES``, then put it back.

    The "before" arm has to be produced from canonical itself, not from the frozen
    reference: frozen draws the RNG in a different order, so its numbers transfer in
    distribution and never to the digit, and bit-identity is the whole claim.
    """
    entry = dgp.RULES.pop(R7)
    try:
        yield
    finally:
        dgp.RULES[R7] = entry


def _run() -> dict:
    """The E0 recipe: calibrate at n_probe, regenerate at n_samples, seed 0."""
    calibrated = solve_binding_rates(E0_TARGETS, n_probe=N_PROBE)
    cfg = DGPConfig(**{**vars(calibrated), "n_samples": N_SAMPLES, "seed": SEED})
    X, godview, manifest = generate(cfg)
    rates = {r["name"]: r["binding_rate"] for r in manifest["rules"]}
    violations = {r["name"]: r["violations"] for r in manifest["rules"]}
    return {
        "cfg": cfg,
        "X": X,
        "godview": godview,
        "manifest": manifest,
        "scalars": {
            "glucose_base": float(cfg.glucose_base),
            "sbp_base": float(cfg.sbp_base),
            "solved_gamma": float(manifest["solved_gamma"]),
            "solved_alpha": float(manifest["solved_alpha"]),
            "realised_prevalence": float(manifest["realised_prevalence"]),
            "expected_bayes_auc": float(manifest["expected_bayes_auc"]),
        },
        "rates": rates,
        "violations": violations,
        "total_violations": int(manifest["total_violations_on_real"]),
    }


def main() -> int:
    t0 = time.perf_counter()
    RESULTS.mkdir(exist_ok=True)
    MANIFESTS.mkdir(exist_ok=True)

    assert R7 in dgp.RULES, f"{R7} is not in dgp.RULES; nothing to test"

    with _without_r7():
        assert R7 not in dgp.RULES
        before = _run()
    after = _run()

    failures: list[str] = []

    def record(quantity: str, b, a) -> dict:
        same = b == a
        if not same:
            failures.append(f"{quantity}: {b!r} != {a!r}")
        return {"quantity": quantity, "before_R7": b, "after_R7": a, "equal": bool(same)}

    rows = [record(k, before["scalars"][k], after["scalars"][k]) for k in SCALARS]
    rows += [record(f"{n}_binding", before["rates"][n], after["rates"][n]) for n in SHARED]
    rows.append(record("total_violations_on_real",
                       before["total_violations"], after["total_violations"]))

    # R7 has no "before" value by construction: it is the row the table gains.
    rows.append({"quantity": f"{R7}_binding", "before_R7": None,
                 "after_R7": after["rates"][R7], "equal": None})
    rows.append({"quantity": "R7_violations_on_real", "before_R7": None,
                 "after_R7": after["violations"][R7], "equal": None})

    # ---- elementwise identity of the sampled frame and of p_true ----
    x_identical = bool(before["X"].equals(after["X"]))
    p_identical = bool(np.array_equal(before["godview"]["p_true"].to_numpy(),
                                      after["godview"]["p_true"].to_numpy()))
    if not x_identical:
        failures.append("X differs between the R1-R6 and R1-R7 runs")
    if not p_identical:
        failures.append("p_true differs between the R1-R6 and R1-R7 runs")

    # ---- zero violations on real data, under both rule sets ----
    if before["total_violations"] != 0:
        failures.append(f"total_violations_on_real (R1-R6) = {before['total_violations']}")
    if after["total_violations"] != 0:
        failures.append(f"total_violations_on_real (R1-R7) = {after['total_violations']}")
    if after["violations"][R7] != 0:
        failures.append(f"R7 violations on real data = {after['violations'][R7]}")

    # ---- R1/R7 complementarity: every row binds exactly one of the two ----
    g = after["godview"]
    r1_bound = g[f"{R1}__binding"].to_numpy()
    r7_bound = g[f"{R7}__binding"].to_numpy()
    n = int(len(after["X"]))
    n_r1, n_r7 = int(r1_bound.sum()), int(r7_bound.sum())
    n_overlap = int((r1_bound & r7_bound).sum())
    partition = (n_r1 + n_r7 == n)
    disjoint = (n_overlap == 0)
    if not partition:
        failures.append(f"R1 + R7 binding counts {n_r1} + {n_r7} != n = {n}")
    if not disjoint:
        failures.append(f"R1 and R7 both bind on {n_overlap} rows")

    # ---- no smoker below the exact rule edge ----
    smoker = after["X"]["Current_Smoker"].to_numpy() == 1
    cigs = after["X"]["Cigs_Per_Day"].to_numpy()
    min_cigs_smokers = float(cigs[smoker].min())
    n_below_edge = int((cigs[smoker] < 1.0).sum())
    if n_below_edge != 0:
        failures.append(f"{n_below_edge} smokers have Cigs_Per_Day < 1.0")

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "s0_e0_r7_invariance.csv", index=False)

    manifest = {
        "artifact": "s0_e0_r7_invariance",
        "stage": "S2",
        "script": "scripts/export_e0_r7_invariance.py",
        "purpose": ("Assert every E0 golden value is bit-identical with and without R7 "
                    "in dgp.RULES. The E0 record is NOT re-run: the DGP's outputs are "
                    "provably identical, so the table gains one row and edits no cell."),
        "comparison": "exact ==; DataFrame.equals; np.array_equal. No tolerance anywhere.",
        **PROV,
        "wall_time_seconds": round(time.perf_counter() - t0, 3),
        "recipe": {"n_probe": N_PROBE, "n_samples": N_SAMPLES, "seed": SEED,
                   "solve_binding_rates": E0_TARGETS},
        "n_rows": int(len(frame)),
        "X_identical": x_identical,
        "p_true_identical": p_identical,
        "r1_r7_partition": {"n_R1_bound": n_r1, "n_R7_bound": n_r7, "sum": n_r1 + n_r7,
                            "n": n, "sums_to_n": partition,
                            "n_both_bound": n_overlap, "disjoint": disjoint},
        "r7_support_edge": {"min_cigs_among_smokers": min_cigs_smokers,
                            "n_smokers_below_1.0": n_below_edge},
        "table": rows,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
        "output_paths": {"csv": "results/s0_e0_r7_invariance.csv",
                         "manifest": "manifests/s0_e0_r7_invariance.json"},
    }
    (MANIFESTS / "s0_e0_r7_invariance.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n")

    def fmt(v):
        return "--" if v is None else (f"{v:.6f}" if isinstance(v, float) else str(v))

    print(f"{'quantity':<30} | {'before_R7':>18} | {'after_R7':>18} | ==")
    for r in rows:
        eq = "" if r["equal"] is None else str(r["equal"])
        print(f"{r['quantity']:<30} | {fmt(r['before_R7']):>18} | "
              f"{fmt(r['after_R7']):>18} | {eq}")

    print()
    print(f"X_identical: {x_identical}")
    print(f"p_true_identical: {p_identical}")
    print(f"R1_bound + R7_bound == n: {n_r1} + {n_r7} = {n_r1 + n_r7} "
          f"vs n={n} -> {partition}")
    print(f"R1/R7 disjoint: {disjoint}")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS -- every pre-existing E0 quantity is bit-identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
