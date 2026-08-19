"""A4 rerun: where does copula label error live, under R1-R6 vs R1-R6 + hypothetical R7?

Varies the DGP seed as well as the copula seed: 5 DGP seeds x 5 copula seeds, 25 fits.
The earlier single-DGP-seed run measured the spread of the generator conditional on one
dataset, not the spread of the measurement, and single seeds run high on the on-manifold
share. Calibration constants are solved once and reused across DGP seeds, matching
``scripts/export_s0.py::export_e0_golden``.

Two spreads are reported. ``across_copula_seed_sd`` is the within-DGP-seed spread over
copula draws, averaged over DGP seeds. ``across_dgp_seed_sd`` is the spread of the five
per-DGP-seed means -- that is the spread of the measurement.

REAL rows do not depend on the copula draw and are computed once per DGP seed. Computing
them inside the copula loop made their across-seed sd 0 by construction.

The R7 arm changes ONLY the on/off-manifold partition; the excess estimator, the local
Bayes floor and TrueRisk are untouched. R7 (`Current_Smoker == 1 -> Cigs_Per_Day >= 1`)
is a FILTER ONLY. dgp.py is not edited: dgp.RULES and every binding rate are unchanged.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
RESULTS = ROOT / "results"
MANIFESTS = ROOT / "manifests"

from dgp import generate, solve_binding_rates
from export_s0 import _provenance  # noqa: E402  -- the repo's one provenance convention
from copula_fix import GaussianCopulaDT
from corrupt import TrueRisk
from rules_tol import check_rules_tol

N_SAMPLES = 40_000          # matches occ_cop.py
DGP_SEEDS = range(5)
SEEDS = range(5)            # copula seeds, within each DGP seed

# Stamp provenance before anything is written -- _provenance() reads `git status`,
# so calling it after the CSV lands reports the tree this run itself dirtied.
PROV = _provenance()

cfg = solve_binding_rates({"R3_glucose_ceil": 0.08, "R4_bp_mandatory": 0.20})
cfg.n_samples = N_SAMPLES

CELLS = ["mass", "prev", "mean_pstar", "abs_res", "bayes_floor", "excess", "AUC_pstar_y"]
KEY = ["gen", "ruleset", "subset"]


def viol_r16(d):
    v = check_rules_tol(d.assign(Heart_Disease_Risk=0))
    return v[[c for c in v.columns if c.endswith("__violated")]].any(axis=1).to_numpy()


def viol_r7(d):
    """Hypothetical R7, filter only."""
    return (d.Current_Smoker.to_numpy() == 1) & (d.Cigs_Per_Day.to_numpy() < 1.0)


def describe(tag, ruleset, Xs, ys, vm, rows, tr, dgp_seed, copula_seed):
    p = tr.p(Xs)
    for nm, k in [("ALL", np.ones(len(p), bool)),
                  ("on-manifold", ~vm), ("off-manifold", vm)]:
        if k.sum() < 50:
            continue
        pk, yk = p[k], ys[k]
        floor = float(np.mean(2 * pk * (1 - pk)))
        rows.append({"gen": tag, "ruleset": ruleset, "subset": nm,
                     "dgp_seed": dgp_seed, "copula_seed": copula_seed,
                     "mass": k.mean(), "prev": yk.mean(),
                     "mean_pstar": pk.mean(), "abs_res": np.abs(yk - pk).mean(),
                     "bayes_floor": floor,
                     "excess": np.abs(yk - pk).mean() - floor,
                     "AUC_pstar_y": roc_auc_score(yk, pk)
                     if len(np.unique(yk)) > 1 else np.nan})


rows = []
n_train = n_holdout = None
bayes_aucs = {}
for dgp_seed in DGP_SEEDS:
    X, G, M = generate(cfg, seed=dgp_seed)
    bayes_aucs[dgp_seed] = M["expected_bayes_auc"]
    FEAT = [c for c in X.columns if c != "Heart_Disease_Risk"]
    Xtr, Xho = train_test_split(X, test_size=0.4, random_state=0,
                                stratify=X.Heart_Disease_Risk)
    ytr, yho = Xtr.pop("Heart_Disease_Risk"), Xho.pop("Heart_Disease_Risk")
    tr = TrueRisk(Xtr, M)
    n_train, n_holdout = len(Xtr), len(Xho)

    # REAL: independent of the copula draw, so computed once per DGP seed.
    Xh = Xho.reset_index(drop=True)
    yh = yho.to_numpy()
    describe("REAL", "R1-R6", Xh, yh, viol_r16(Xh), rows, tr, dgp_seed, np.nan)
    describe("REAL", "R1-R6+R7", Xh, yh, viol_r16(Xh) | viol_r7(Xh), rows, tr, dgp_seed, np.nan)

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        cop = GaussianCopulaDT().fit(Xtr.assign(Heart_Disease_Risk=ytr.values), rng)\
                                .sample(len(Xtr), rng)
        yc = cop.pop("Heart_Disease_Risk").astype(int).to_numpy()
        Xc = cop[FEAT]
        describe("copula", "R1-R6", Xc, yc, viol_r16(Xc), rows, tr, dgp_seed, seed)
        describe("copula", "R1-R6+R7", Xc, yc, viol_r16(Xc) | viol_r7(Xc), rows, tr, dgp_seed, seed)
    print("dgp seed", dgp_seed, "done", flush=True)

raw = pd.DataFrame(rows)

# Aggregate to a per-DGP-seed mean over copula seeds FIRST, then across DGP seeds.
per_seed = raw.groupby(KEY + ["dgp_seed"])[CELLS].mean().reset_index()
per_seed["excess_mass"] = per_seed["mass"] * per_seed["excess"]

d = per_seed.groupby(KEY)[CELLS + ["excess_mass"]].mean()
across_dgp_sd = per_seed.groupby(KEY)[CELLS + ["excess_mass"]].std()
# Within a DGP seed, over copula draws; averaged over DGP seeds. REAL has one row per
# DGP seed, so it has no within-seed spread and is absent here rather than reported as 0.
across_cop_sd = (raw[raw.gen == "copula"].groupby(KEY + ["dgp_seed"])[CELLS].std()
                 .groupby(KEY).mean())

pd.set_option("display.width", 220)
print("\n==== WHERE THE LABEL ERROR LIVES (%d DGP seeds x %d copula seeds, n_synth=%d) ===="
      % (len(list(DGP_SEEDS)), len(list(SEEDS)), n_train))
print(d.round(4).to_string())
print("\nacross-DGP-seed sd (of the five per-seed means):")
print(across_dgp_sd.round(5).to_string())
print("\nacross-copula-seed sd (within a DGP seed, averaged over DGP seeds):")
print(across_cop_sd.round(5).to_string())

# ---- the REAL ALL null: E|y - p*| = 2 E[p*(1-p*)] exactly, so this must be ~0 ----
null = per_seed[(per_seed.gen == "REAL") & (per_seed.ruleset == "R1-R6")
                & (per_seed.subset == "ALL")].sort_values("dgp_seed")
null_by_seed = {int(r.dgp_seed): float(r.excess) for r in null.itertuples()}
null_mean, null_sd = float(null.excess.mean()), float(null.excess.std())
print("\nREAL ALL excess (the null; must be ~0 in expectation):")
print("  per DGP seed: " + "  ".join(f"{s}={v:+.5f}" for s, v in null_by_seed.items()))
print(f"  mean={null_mean:+.5f}  sd={null_sd:.5f}  n_holdout={n_holdout}")

# ---- shares, per DGP seed then across ----
shares = {}
for rs in ["R1-R6", "R1-R6+R7"]:
    vals = []
    for s in DGP_SEEDS:
        sub = per_seed[(per_seed.gen == "copula") & (per_seed.ruleset == rs)
                       & (per_seed.dgp_seed == s)].set_index("subset")
        on, off = sub.loc["on-manifold", "excess_mass"], sub.loc["off-manifold", "excess_mass"]
        vals.append(100.0 * on / (on + off))
    g = d.loc[("copula", rs)]
    shares[rs] = {
        "on_manifold_share_pct_by_dgp_seed": [float(v) for v in vals],
        "on_manifold_share_pct_mean": float(np.mean(vals)),
        "on_manifold_share_pct_sd": float(np.std(vals, ddof=1)),
        "off_manifold_share_pct_mean": float(100 - np.mean(vals)),
        "off_manifold_mass_pct_mean": float(100 * g.loc["off-manifold", "mass"]),
        "on_manifold_excess_mean": float(g.loc["on-manifold", "excess"]),
        "off_manifold_excess_mean": float(g.loc["off-manifold", "excess"]),
    }
    print(f"\n[{rs}] copula share of excess contradiction: on-manifold "
          f"{np.mean(vals):.1f}% +/- {np.std(vals, ddof=1):.2f}  "
          f"off-manifold {100-np.mean(vals):.1f}%  "
          f"(off-manifold is {100*g.loc['off-manifold','mass']:.1f}% of rows)")

# ---- sign conditions, on each DGP seed individually ----
cop_ps = per_seed[per_seed.gen == "copula"].set_index(["ruleset", "subset", "dgp_seed"])
signs = {}
for s in DGP_SEEDS:
    on16 = cop_ps.loc[("R1-R6", "on-manifold", s), "excess"]
    on167 = cop_ps.loc[("R1-R6+R7", "on-manifold", s), "excess"]
    off16 = cop_ps.loc[("R1-R6", "off-manifold", s), "excess"]
    off167 = cop_ps.loc[("R1-R6+R7", "off-manifold", s), "excess"]
    signs[int(s)] = {"on_manifold_excess_rises": bool(on167 > on16),
                     "off_manifold_excess_falls": bool(off167 < off16),
                     "on_16": float(on16), "on_16_r7": float(on167),
                     "off_16": float(off16), "off_16_r7": float(off167)}
print("\nsign conditions per DGP seed (excess under R1-R6 -> R1-R6+R7):")
for s, v in signs.items():
    print(f"  seed {s}: on {v['on_16']:.4f}->{v['on_16_r7']:.4f} "
          f"{'RISES ok' if v['on_manifold_excess_rises'] else 'INVERTED'} | "
          f"off {v['off_16']:.4f}->{v['off_16_r7']:.4f} "
          f"{'FALLS ok' if v['off_manifold_excess_falls'] else 'INVERTED'}")

raw.to_csv(RESULTS / "occupancy_copula_r7.csv", index=False)
manifest = {
    "script": "occ_cop_r7.py",
    "purpose": "A4 rerun with R7 filter-only, across DGP seeds as well as copula seeds.",
    "design": ("5 DGP seeds x 5 copula seeds = 25 fits. Aggregated to a per-DGP-seed "
               "mean over copula seeds first, then across DGP seeds. REAL rows do not "
               "depend on the copula draw and are computed once per DGP seed."),
    "dgp_config": {"n_samples": N_SAMPLES, "dgp_seeds": list(DGP_SEEDS),
                   "solve_binding_rates": {"R3_glucose_ceil": 0.08,
                                           "R4_bp_mandatory": 0.20},
                   "calibration": "solved once, reused across DGP seeds (E0 recipe)",
                   "glucose_base": cfg.glucose_base, "sbp_base": cfg.sbp_base},
    "expected_bayes_auc_by_dgp_seed": {str(k): float(v) for k, v in bayes_aucs.items()},
    "split": {"test_size": 0.4, "random_state": 0, "stratify": "Heart_Disease_Risk",
              "n_train": n_train, "n_holdout": n_holdout},
    "copula_seeds": list(SEEDS),
    "R7_definition": "Current_Smoker == 1 -> Cigs_Per_Day >= 1 (filter only, not in dgp.RULES)",
    "rules_tol": 0.5,
    "estimator": "excess = mean|y-p*| - 2*E[p*(1-p*)], computed within subset",
    "real_all_excess_null": {"by_dgp_seed": null_by_seed, "mean": null_mean,
                             "sd": null_sd, "n_holdout": n_holdout,
                             "expectation": "exactly 0; y ~ Bernoulli(p*) on real data"},
    "shares": shares,
    "sign_conditions_by_dgp_seed": signs,
    "cell_means": {f"{a}|{b}|{c}": {k: float(v) for k, v in r.items()}
                   for (a, b, c), r in d.iterrows()},
    "across_dgp_seed_sd": {f"{a}|{b}|{c}": {k: (None if pd.isna(v) else float(v))
                                            for k, v in r.items()}
                           for (a, b, c), r in across_dgp_sd.iterrows()},
    "across_copula_seed_sd": {f"{a}|{b}|{c}": {k: (None if pd.isna(v) else float(v))
                                               for k, v in r.items()}
                              for (a, b, c), r in across_cop_sd.iterrows()},
    **PROV,
    "outputs": ["results/occupancy_copula_r7.csv"],
}
with open(MANIFESTS / "occupancy_copula_r7.json", "w") as f:
    json.dump(manifest, f, indent=2)
