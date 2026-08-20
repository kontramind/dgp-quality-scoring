"""Run the occupancy harness over every registered arm and write the results table.

Run: ``uv run python scripts/export_occupancy.py``

Design is ``occ_cop_r7.py``'s at ``eb5c040``, reproduced rather than reinvented -- see
``occupancy.py``'s module docstring for the list and the reasons. The one thing that is
*not* inherited is that script's R1-R6 arm: at HEAD ``check_rules_tol`` reads R7 out of
``dgp.RULES``, so an "R1-R6" row computed through it would silently be an R1-R7 row. Only
the R1-R7 measurement is made here. The R1-R6 contrast stands as a historical figure at
``571c444`` and is not recomputable at HEAD.

``TrueRisk`` is frozen once per DGP seed on the *train* half and passed into every arm.
It must not be rebuilt inside the arm loop: it freezes its moments at construction, so a
different reference frame rescales p* by a small amount that leaves each arm internally
consistent while making arms incomparable with each other.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import dgp
from corrupt import TrueRisk
from dgp import generate, solve_binding_rates
from export_s0 import _provenance  # noqa: E402  -- the repo's one provenance convention
from occupancy import ARMS, RULE_NAMES, TOL_BAND, evaluate_frame

RESULTS = ROOT / "results"
MANIFESTS = ROOT / "manifests"

# --- design constants, all taken from occ_cop_r7.py at eb5c040 ---
E0_TARGETS = {"R3_glucose_ceil": 0.08, "R4_bp_mandatory": 0.20}
N_SAMPLES = 40_000              # real cohort per DGP seed
TEST_SIZE = 0.4                 # stratified on the label, random_state=0
SPLIT_STATE = 0
DGP_SEEDS = range(5)
GEN_SEEDS = range(5)            # independent of the DGP seed, as in occ_cop_r7.py

ARTIFACT = "occupancy_arms"

# Stamp provenance before anything is written -- _provenance() reads `git status`, so
# calling it after the CSV lands reports the tree this run itself dirtied.
PROV = _provenance()


def main() -> int:
    t0 = time.perf_counter()
    RESULTS.mkdir(exist_ok=True)
    MANIFESTS.mkdir(exist_ok=True)

    cfg = solve_binding_rates(E0_TARGETS)
    cfg.n_samples = N_SAMPLES   # solve_binding_rates returns n_samples = n_probe

    rows: list[dict] = []
    dgp_checks: dict[str, dict] = {}
    n_train = n_holdout = None

    for dgp_seed in DGP_SEEDS:
        X, _G, M = generate(cfg, seed=dgp_seed)
        feat = [c for c in X.columns if c != dgp.LABEL]
        dgp_checks[str(dgp_seed)] = {
            "realised_prevalence": M["realised_prevalence"],
            "expected_bayes_auc": M["expected_bayes_auc"],
            "solved_gamma": M["solved_gamma"],
            "total_violations_on_real": M["total_violations_on_real"],
        }

        Xtr, Xho = train_test_split(X, test_size=TEST_SIZE, random_state=SPLIT_STATE,
                                    stratify=X[dgp.LABEL])
        ytr = Xtr.pop(dgp.LABEL)
        yho = Xho.pop(dgp.LABEL)
        n_train, n_holdout = len(Xtr), len(Xho)

        # X_ref resolved ONCE per DGP seed, outside the arm loop, and passed in.
        true_risk = TrueRisk(Xtr, M)

        # The real holdout: the estimator's null. Independent of any generator, so it is
        # computed once per DGP seed rather than once per (arm, seed) -- doing the latter
        # would make its across-seed sd zero by construction.
        real = evaluate_frame(Xho.reset_index(drop=True), yho.to_numpy(), true_risk, cfg)
        rows.append({"arm": "real", "dgp_seed": dgp_seed, "gen_seed": np.nan, **real})

        train_with_label = Xtr.assign(**{dgp.LABEL: ytr.values})
        for arm_name, make_arm in ARMS.items():
            arm = make_arm()
            for gen_seed in GEN_SEEDS:
                synth = arm.fit_sample(train_with_label, n_train, gen_seed)
                ys = synth.pop(dgp.LABEL).astype(int).to_numpy()
                res = evaluate_frame(synth[feat], ys, true_risk, cfg)
                rows.append({"arm": arm_name, "dgp_seed": dgp_seed,
                             "gen_seed": gen_seed, **res})
        print(f"  dgp seed {dgp_seed} done", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / f"{ARTIFACT}.csv", index=False)

    # ---------------------------------------------------------------------------------
    # Aggregation, matching occ_cop_r7.py: mean over generator seeds within a DGP seed
    # first, so one DGP seed is one unit, then mean and sd across DGP seeds (ddof=1).
    #
    # The excess share is formed from the *averaged* mass and excess, exactly as
    # occ_cop_r7.py does (`per_seed["excess_mass"] = per_seed["mass"] * per_seed["excess"]`
    # applied after the groupby-mean). Averaging the per-row share column instead is a
    # mean of ratios rather than a ratio of means and does not reproduce the figure.
    # ---------------------------------------------------------------------------------
    summary: dict[str, dict] = {}
    for arm_name in [a for a in frame.arm.unique() if a != "real"]:
        sub = frame[frame.arm == arm_name]
        per_seed = sub.groupby("dgp_seed").mean(numeric_only=True)
        share = 100.0 * (
            per_seed["mass_on_manifold"] * per_seed["excess_on_manifold"]
        ) / (
            per_seed["mass_on_manifold"] * per_seed["excess_on_manifold"]
            + per_seed["mass_off_manifold"] * per_seed["excess_off_manifold"]
        )
        summary[arm_name] = {
            "on_manifold_excess_share_pct_by_dgp_seed": [float(v) for v in share],
            "on_manifold_excess_share_pct_mean": float(share.mean()),
            "on_manifold_excess_share_pct_sd": float(share.std(ddof=1)),
            "per_dgp_seed_means": {
                c: [float(v) for v in per_seed[c]]
                for c in ("on_manifold_row_share", "excess_all", "se_all",
                          "excess_on_manifold", "se_on_manifold",
                          "excess_off_manifold", "se_off_manifold",
                          "frac_density_zero_ruleclean", "frac_density_zero_violating",
                          "n_cigs_in_tol_band")
            },
            "mean": {c: float(per_seed[c].mean()) for c in per_seed.columns},
            "across_dgp_seed_sd": {c: float(per_seed[c].std(ddof=1))
                                   for c in per_seed.columns},
            "density_zero_violating_exact_all_rows": bool(
                sub["density_zero_violating_exact"].all()),
            "n_cigs_in_tol_band_total": int(sub["n_cigs_in_tol_band"].sum()),
        }

    real_rows = frame[frame.arm == "real"]
    real_null = {
        "excess_all_by_dgp_seed": [float(v) for v in real_rows["excess_all"]],
        "excess_all_mean": float(real_rows["excess_all"].mean()),
        "excess_all_sd": float(real_rows["excess_all"].std(ddof=1)),
        "se_all_mean": float(real_rows["se_all"].mean()),
        "on_manifold_row_share_mean": float(real_rows["on_manifold_row_share"].mean()),
        "expectation": "0 in expectation; y ~ Bernoulli(p*) on real data",
    }

    manifest = {
        "artifact": ARTIFACT,
        "stage": "S2",
        "script": "scripts/export_occupancy.py",
        "purpose": ("Generator occupancy and contradiction, R1-R7, at HEAD. Re-establishes "
                    "occ_cop_r7.py's post-R7 measurement through a harness whose generator "
                    "is a parameter."),
        **PROV,
        "wall_time_seconds": round(time.perf_counter() - t0, 3),
        "n_rows": int(len(frame)),
        "design_source": {
            "from": "occ_cop_r7.py at eb5c040 (results committed at 571c444)",
            "n_samples": N_SAMPLES,
            "split": {"test_size": TEST_SIZE, "random_state": SPLIT_STATE,
                      "stratify": dgp.LABEL, "n_train": n_train, "n_holdout": n_holdout},
            "X_ref_for_TrueRisk": ("the train half (features only, label popped), frozen "
                                   "once per DGP seed outside the arm loop"),
            "generator_sample_size": n_train,
            "seed_pairing": ("5 DGP seeds x 5 generator seeds; the generator seed is "
                             "np.random.default_rng(gen_seed) and does not depend on the "
                             "DGP seed. One RNG serves both fit and sample."),
            "aggregation": ("mean over generator seeds within a DGP seed first, then mean "
                            "and sd (ddof=1) across DGP seeds"),
            "not_inherited": ("occ_cop_r7.py's R1-R6 arm. check_rules_tol reads R7 out of "
                              "dgp.RULES at HEAD, so an R1-R6 row computed through it "
                              "would silently be an R1-R7 row. The R1-R6 contrast stands "
                              "as a historical figure at 571c444."),
        },
        "dgp_config": {
            "solve_binding_rates": E0_TARGETS, "n_probe": 40_000,
            "calibration": "solved once, reused across DGP seeds (E0 recipe)",
            "glucose_base": cfg.glucose_base, "sbp_base": cfg.sbp_base,
        },
        "dgp_manifest_checks_by_dgp_seed": dgp_checks,
        "arms": list(ARMS),
        "ruleset": "R1-R7 via rules_tol.check_rules_tol",
        "rules_tol": TOL_BAND,
        "rule_names": RULE_NAMES,
        "estimator": {
            "excess": "mean|y - p*| - 2*E[p*(1-p*)], computed within subset",
            "se": "sqrt(mean[p*(1-p*)(1-2p*)^2] / m), from each subset's own p*",
            "forbidden": ("excess is never computed inside a label-defined subset; the "
                          "on/off-manifold split is feature-defined and therefore safe"),
        },
        "summary_by_arm": summary,
        "real_null": real_null,
        "output_paths": {"csv": f"results/{ARTIFACT}.csv",
                         "manifest": f"manifests/{ARTIFACT}.json"},
    }
    (MANIFESTS / f"{ARTIFACT}.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n")

    # ---- report ----
    pd.set_option("display.width", 220)
    for arm_name, s in summary.items():
        m, sd = s["on_manifold_excess_share_pct_mean"], s["on_manifold_excess_share_pct_sd"]
        print(f"\n[{arm_name}] on-manifold share of excess contradiction: "
              f"{m:.4f}% [sd {sd:.4f}]")
        print(f"  per DGP seed: " +
              "  ".join(f"{v:.3f}" for v in s["on_manifold_excess_share_pct_by_dgp_seed"]))
        print(f"  on_manifold_row_share      {s['mean']['on_manifold_row_share']:.6f} "
              f"[sd {s['across_dgp_seed_sd']['on_manifold_row_share']:.6f}]")
        for c in ("excess_all", "se_all", "excess_on_manifold", "se_on_manifold",
                  "excess_off_manifold", "se_off_manifold"):
            print(f"  {c:<26} {s['mean'][c]:.6f} [sd {s['across_dgp_seed_sd'][c]:.6f}]")
        print(f"  frac_density_zero_ruleclean  {s['mean']['frac_density_zero_ruleclean']:.6f}")
        print(f"  frac_density_zero_violating  {s['mean']['frac_density_zero_violating']:.6f}"
              f"  (exact array equality on every row: "
              f"{s['density_zero_violating_exact_all_rows']})")
        print(f"  n_cigs_in_tol_band (total)   {s['n_cigs_in_tol_band_total']}")
        print("  per-rule violation rates:")
        for n in RULE_NAMES:
            print(f"    {n:<22} {s['mean'][f'{n}__violation_rate']:.6f} "
                  f"[sd {s['across_dgp_seed_sd'][f'{n}__violation_rate']:.6f}]")

    print(f"\nreal null: excess_all = {real_null['excess_all_mean']:+.6f} "
          f"[sd {real_null['excess_all_sd']:.6f}]  se = {real_null['se_all_mean']:.6f}")
    print(f"  per DGP seed: " +
          "  ".join(f"{v:+.6f}" for v in real_null["excess_all_by_dgp_seed"]))
    print(f"\nwrote results/{ARTIFACT}.csv ({len(frame)} rows) + manifest "
          f"(git_dirty={PROV['git_dirty']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
