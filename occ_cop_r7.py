"""A4 rerun: where does copula label error live, under R1-R6 vs R1-R6 + hypothetical R7?

Baseline arm reproduces occ_cop.py exactly (same config, same n, same split, same
seeds) so the record's 77.5% is checkable. The R7 arm changes ONLY the on/off-manifold
partition; the excess estimator, the local Bayes floor and TrueRisk are untouched.

R7 (`Current_Smoker == 1 -> Cigs_Per_Day >= 1`) is a FILTER ONLY. dgp.py is not
edited: dgp.RULES, check_rules and every binding rate in the record are unchanged.
"""
import json
import subprocess
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from dgp import generate, solve_binding_rates
from copula_fix import GaussianCopulaDT
from corrupt import TrueRisk
from rules_tol import check_rules_tol

N_SAMPLES = 40_000          # matches occ_cop.py
SEEDS = range(5)

cfg = solve_binding_rates({"R3_glucose_ceil": 0.08, "R4_bp_mandatory": 0.20})
cfg.n_samples = N_SAMPLES
X, G, M = generate(cfg)
FEAT = [c for c in X.columns if c != "Heart_Disease_Risk"]
Xtr, Xho = train_test_split(X, test_size=0.4, random_state=0,
                            stratify=X.Heart_Disease_Risk)
ytr, yho = Xtr.pop("Heart_Disease_Risk"), Xho.pop("Heart_Disease_Risk")
tr = TrueRisk(Xtr, M)
print("Bayes AUC of the DGP:", round(M["expected_bayes_auc"], 4), flush=True)


def viol_r16(d):
    v = check_rules_tol(d.assign(Heart_Disease_Risk=0))
    return v[[c for c in v.columns if c.endswith("__violated")]].any(axis=1).to_numpy()


def viol_r7(d):
    """Hypothetical R7, filter only."""
    return (d.Current_Smoker.to_numpy() == 1) & (d.Cigs_Per_Day.to_numpy() < 1.0)


def describe(tag, ruleset, Xs, ys, vm, rows):
    p = tr.p(Xs)
    for nm, k in [("ALL", np.ones(len(p), bool)),
                  ("on-manifold", ~vm), ("off-manifold", vm)]:
        if k.sum() < 50:
            continue
        pk, yk = p[k], ys[k]
        floor = float(np.mean(2 * pk * (1 - pk)))
        rows.append({"gen": tag, "ruleset": ruleset, "subset": nm,
                     "mass": k.mean(), "prev": yk.mean(),
                     "mean_pstar": pk.mean(), "abs_res": np.abs(yk - pk).mean(),
                     "bayes_floor": floor,
                     "excess": np.abs(yk - pk).mean() - floor,
                     "AUC_pstar_y": roc_auc_score(yk, pk)
                     if len(np.unique(yk)) > 1 else np.nan})


rows = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    Xh = Xho.reset_index(drop=True)
    yh = yho.to_numpy()
    describe("REAL", "R1-R6", Xh, yh, viol_r16(Xh), rows)
    describe("REAL", "R1-R6+R7", Xh, yh, viol_r16(Xh) | viol_r7(Xh), rows)

    cop = GaussianCopulaDT().fit(Xtr.assign(Heart_Disease_Risk=ytr.values), rng)\
                            .sample(len(Xtr), rng)
    yc = cop.pop("Heart_Disease_Risk").astype(int).to_numpy()
    Xc = cop[FEAT]
    describe("copula", "R1-R6", Xc, yc, viol_r16(Xc), rows)
    describe("copula", "R1-R6+R7", Xc, yc, viol_r16(Xc) | viol_r7(Xc), rows)
    print("seed", seed, "done", flush=True)

raw = pd.DataFrame(rows)
d = raw.groupby(["gen", "ruleset", "subset"]).mean(numeric_only=True)
d["excess_mass"] = d["mass"] * d["excess"]
sd = raw.groupby(["gen", "ruleset", "subset"])["excess"].std()

pd.set_option("display.width", 220)
print("\n==== WHERE THE LABEL ERROR LIVES (5 seeds, n_synth=%d) ====" % len(Xtr))
print(d.round(4).to_string())

shares = {}
for rs in ["R1-R6", "R1-R6+R7"]:
    s = d.loc[("copula", rs)]
    tot = s.loc["on-manifold", "excess_mass"] + s.loc["off-manifold", "excess_mass"]
    on_share = 100 * s.loc["on-manifold", "excess_mass"] / tot
    shares[rs] = {"on_manifold_share_pct": float(on_share),
                  "off_manifold_share_pct": float(100 - on_share),
                  "off_manifold_mass_pct": float(100 * s.loc["off-manifold", "mass"]),
                  "on_manifold_excess": float(s.loc["on-manifold", "excess"]),
                  "off_manifold_excess": float(s.loc["off-manifold", "excess"])}
    print(f"\n[{rs}] copula share of excess contradiction: "
          f"on-manifold {on_share:.1f}%  off-manifold {100-on_share:.1f}%  "
          f"(off-manifold is {100*s.loc['off-manifold','mass']:.1f}% of rows)")

print("\nacross-seed sd of `excess`:")
print(sd.round(5).to_string())

raw.to_csv("occupancy_copula_r7.csv", index=False)
try:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip() or None
except Exception:
    sha = None
manifest = {
    "script": "occ_cop_r7.py",
    "purpose": "A4 rerun with R7 filter-only; does the 77.5% on-manifold share survive?",
    "baseline_arm": "reproduces occ_cop.py (config, n, split, seeds) for checkability",
    "dgp_config": {"n_samples": N_SAMPLES, "seed": cfg.seed,
                   "solve_binding_rates": {"R3_glucose_ceil": 0.08,
                                           "R4_bp_mandatory": 0.20},
                   "glucose_base": cfg.glucose_base, "sbp_base": cfg.sbp_base},
    "dgp_manifest_checks": {
        "realised_prevalence": M["realised_prevalence"],
        "expected_bayes_auc": M["expected_bayes_auc"],
        "solved_gamma": M["solved_gamma"],
        "total_violations_on_real": M["total_violations_on_real"]},
    "split": {"test_size": 0.4, "random_state": 0, "stratify": "Heart_Disease_Risk",
              "n_train": len(Xtr), "n_holdout": len(Xho)},
    "seeds": list(SEEDS),
    "R7_definition": "Current_Smoker == 1 -> Cigs_Per_Day >= 1 (filter only, not in dgp.RULES)",
    "rules_tol": 0.5,
    "estimator": "excess = mean|y-p*| - 2*E[p*(1-p*)], computed within subset",
    "shares": shares,
    "across_seed_sd_excess": {f"{a}|{b}|{c}": float(v)
                              for (a, b, c), v in sd.items()},
    "git_sha": sha,
    "outputs": ["occupancy_copula_r7.csv"],
}
with open("occupancy_copula_r7_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
