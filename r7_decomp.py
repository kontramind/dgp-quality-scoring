"""Decompose rule-vs-density disagreement on copula output.

R7 (`Current_Smoker == 1 -> Cigs_Per_Day >= 1`) is applied as a FILTER ONLY here.
dgp.py is NOT edited: RULES, check_rules and every binding rate in the record are
untouched. R7 exists only inside this script.

Real-row log-density percentiles are taken from the real TRAIN half (the reference
distribution the copula was fitted to). Both the real holdout and the copula draws are
scored out-of-sample against that fixed floor, so the real-data row is a genuine null
with sampling noise rather than 0.0100/0.0500/0.1000 by construction.
"""
import json
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dgp import generate, solve_binding_rates
from copula_fix import GaussianCopulaDT
from rules_tol import check_rules_tol
from density import log_density_X

DGP_SEED = 0
N_SAMPLES = 50_000
N_PROBE = 40_000
COPULA_SEEDS = range(5)

# ---- E0 config: calibrate at n_probe, then set n_samples (deliberate, per record) ----
cfg = solve_binding_rates({"R3_glucose_ceil": 0.08, "R4_bp_mandatory": 0.20},
                          n_probe=N_PROBE)
cfg.n_samples = N_SAMPLES
cfg.seed = DGP_SEED
X, G, M = generate(cfg)
FEAT = [c for c in X.columns if c != "Heart_Disease_Risk"]

Xtr, Xho = train_test_split(X, test_size=0.4, random_state=0,
                            stratify=X.Heart_Disease_Risk)
ytr = Xtr.pop("Heart_Disease_Risk")
yho = Xho.pop("Heart_Disease_Risk")
Xtr = Xtr.reset_index(drop=True)
Xho = Xho.reset_index(drop=True)


def r1_r6_ok(d):
    v = check_rules_tol(d)
    return ~v[[c for c in v.columns if c.endswith("__violated")]].any(axis=1).to_numpy()


def r7_ok(d):
    """Hypothetical R7, filter only: Current_Smoker == 1 -> Cigs_Per_Day >= 1."""
    return ~((d.Current_Smoker.to_numpy() == 1) & (d.Cigs_Per_Day.to_numpy() < 1.0))


# ---- floor from the real train half ----
logp_tr = log_density_X(Xtr, cfg)
assert np.isfinite(logp_tr).all()
q1, q5, q10 = np.percentile(logp_tr, [1, 5, 10])


def profile(logp):
    zero = ~np.isfinite(logp)
    return {
        "n": len(logp),
        "frac_density_zero": float(zero.mean()),
        "frac_below_p1": float((logp < q1).mean()),
        "frac_below_p5": float((logp < q5).mean()),
        "frac_below_p10": float((logp < q10).mean()),
    }


# ---- real holdout: the null ----
logp_ho = log_density_X(Xho, cfg)
real_row = {"row_set": "real data (the null)", **profile(logp_ho)}

# sanity: R7 must never fire on real data
assert r7_ok(Xho).all(), "R7 fired on real data"
assert (Xho.Current_Smoker.to_numpy() * (Xho.Cigs_Per_Day.to_numpy() == 0)).sum() == 0

acc = {k: [] for k in ["r16", "r16_r7", "r16_r7_pos"]}
diag = []
for s in COPULA_SEEDS:
    rng = np.random.default_rng(s)
    cop = GaussianCopulaDT().fit(Xtr.assign(Heart_Disease_Risk=ytr.values), rng)\
                            .sample(len(Xtr), rng)
    cop.pop("Heart_Disease_Risk")
    cop = cop[FEAT]

    ok16 = r1_r6_ok(cop)
    ok7 = r7_ok(cop)
    logp = log_density_X(cop, cfg)

    m16 = ok16
    m167 = ok16 & ok7
    lp16, lp167 = logp[m16], logp[m167]
    lp167_pos = lp167[np.isfinite(lp167)]

    acc["r16"].append(profile(lp16))
    acc["r16_r7"].append(profile(lp167))
    acc["r16_r7_pos"].append(profile(lp167_pos))
    diag.append({
        "seed": s,
        "on_manifold_share_r16": float(m16.mean()),
        "smoker1_cigs0_share_of_r16": float(((cop.Current_Smoker.to_numpy() == 1)
                                             & (cop.Cigs_Per_Day.to_numpy() == 0))[m16].mean()),
        "r1_violations_share_all": float(((cop.Current_Smoker.to_numpy() == 0)
                                          & (cop.Cigs_Per_Day.to_numpy() >= 0.5)).mean()),
        "on_manifold_share_r16_r7": float(m167.mean()),
    })


def agg(key, label):
    fr = pd.DataFrame(acc[key])
    out = {"row_set": label, "n": int(round(fr["n"].mean()))}
    for c in ["frac_density_zero", "frac_below_p1", "frac_below_p5", "frac_below_p10"]:
        out[c] = float(fr[c].mean())
    return out, fr


rows = [real_row]
sds = {}
for key, label in [("r16", "copula, on-manifold under R1-R6"),
                   ("r16_r7", "copula, on-manifold under R1-R6 + R7"),
                   ("r16_r7_pos", "copula, on-manifold under R1-R6 + R7, positive density only")]:
    r, fr = agg(key, label)
    rows.append(r)
    sds[label] = fr[["frac_density_zero", "frac_below_p1",
                     "frac_below_p5", "frac_below_p10"]].std().to_dict()

tab = pd.DataFrame(rows)[["row_set", "n", "frac_density_zero",
                          "frac_below_p1", "frac_below_p5", "frac_below_p10"]]
tab.to_csv("r7_decomp_results.csv", index=False)
pd.DataFrame(diag).to_csv("r7_decomp_diagnostics.csv", index=False)

try:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip() or None
except Exception:
    sha = None

manifest = {
    "script": "r7_decomp.py",
    "purpose": "Decompose rules-vs-density disagreement; R7 filter-only, dgp.py unedited.",
    "dgp_config": {"n_probe": N_PROBE, "n_samples": N_SAMPLES, "seed": DGP_SEED,
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
    "density_floor": {"source": "real train half", "n": len(Xtr),
                      "p1": float(q1), "p5": float(q5), "p10": float(q10)},
    "copula_seeds": list(COPULA_SEEDS),
    "R7_definition": "Current_Smoker == 1 -> Cigs_Per_Day >= 1 (filter only, not in dgp.RULES)",
    "rules_tol": 0.5,
    "across_seed_sd": sds,
    "git_sha": sha,
    "outputs": ["r7_decomp_results.csv", "r7_decomp_diagnostics.csv"],
}
with open("r7_decomp_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(tab.round(4).to_string(index=False))
print()
print("across-seed sd, max over all copula cells:",
      round(max(max(v.values()) for v in sds.values()), 5))
print()
print(pd.DataFrame(diag).round(4).to_string(index=False))
