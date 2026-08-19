"""Decompose rule-vs-density disagreement on copula output.

R7 (`Current_Smoker == 1 -> Cigs_Per_Day >= 1`) is applied as a FILTER ONLY here.
dgp.py is NOT edited: RULES, check_rules and every binding rate in the record are
untouched. R7 exists only inside this script.

5 DGP seeds x 5 copula seeds. Calibration is solved once at n_probe and reused across
DGP seeds -- the E0 convention, per scripts/export_s0.py::export_e0_golden -- and the
DGP seed is varied through generate(cfg, seed=s). Aggregation goes to a per-DGP-seed
mean over copula seeds first, so a DGP seed is one unit; the manifest carries
across_dgp_seed_sd for every cell alongside the within-seed copula spread.

Real-row log-density percentiles are taken from the real TRAIN half (the reference
distribution the copula was fitted to), refitted per DGP seed. Both the real holdout and
the copula draws are scored out-of-sample against that fixed floor, so the real-data row
is a genuine null with sampling noise rather than 0.0100/0.0500/0.1000 by construction.
The real row does not depend on the copula draw and is computed once per DGP seed; it is
therefore absent from the within-seed spread rather than reported there as a zero.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
RESULTS = ROOT / "results"
MANIFESTS = ROOT / "manifests"

from dgp import generate, solve_binding_rates
from export_s0 import _provenance  # noqa: E402  -- the repo's one provenance convention
from copula_fix import GaussianCopulaDT
from rules_tol import check_rules_tol
from density import log_density_X

DGP_SEEDS = range(5)
N_SAMPLES = 50_000
N_PROBE = 40_000
COPULA_SEEDS = range(5)

CELLS = ["n", "frac_density_zero", "frac_below_p1", "frac_below_p5", "frac_below_p10"]
ROW_SETS = [("real", "real data (the null)"),
            ("r16", "copula, on-manifold under R1-R6"),
            ("r16_r7", "copula, on-manifold under R1-R6 + R7"),
            ("r16_r7_pos", "copula, on-manifold under R1-R6 + R7, positive density only")]

# Stamp provenance before anything is written -- _provenance() reads `git status`,
# so calling it after the CSVs land reports the tree this run itself dirtied.
PROV = _provenance()

# ---- E0 config: calibrate at n_probe, then set n_samples (deliberate, per record) ----
cfg = solve_binding_rates({"R3_glucose_ceil": 0.08, "R4_bp_mandatory": 0.20},
                          n_probe=N_PROBE)
cfg.n_samples = N_SAMPLES


def r1_r6_ok(d):
    v = check_rules_tol(d)
    return ~v[[c for c in v.columns if c.endswith("__violated")]].any(axis=1).to_numpy()


def r7_ok(d):
    """Hypothetical R7, filter only: Current_Smoker == 1 -> Cigs_Per_Day >= 1."""
    return ~((d.Current_Smoker.to_numpy() == 1) & (d.Cigs_Per_Day.to_numpy() < 1.0))


def profile(logp, q1, q5, q10):
    zero = ~np.isfinite(logp)
    return {
        "n": len(logp),
        "frac_density_zero": float(zero.mean()),
        "frac_below_p1": float((logp < q1).mean()),
        "frac_below_p5": float((logp < q5).mean()),
        "frac_below_p10": float((logp < q10).mean()),
    }


records, diag, floors, dgp_checks = [], [], {}, {}
n_train = n_holdout = None

for dgp_seed in DGP_SEEDS:
    X, G, M = generate(cfg, seed=dgp_seed)
    FEAT = [c for c in X.columns if c != "Heart_Disease_Risk"]
    dgp_checks[str(dgp_seed)] = {
        "realised_prevalence": M["realised_prevalence"],
        "expected_bayes_auc": M["expected_bayes_auc"],
        "solved_gamma": M["solved_gamma"],
        "total_violations_on_real": M["total_violations_on_real"]}

    Xtr, Xho = train_test_split(X, test_size=0.4, random_state=0,
                                stratify=X.Heart_Disease_Risk)
    ytr = Xtr.pop("Heart_Disease_Risk")
    yho = Xho.pop("Heart_Disease_Risk")
    Xtr = Xtr.reset_index(drop=True)
    Xho = Xho.reset_index(drop=True)
    n_train, n_holdout = len(Xtr), len(Xho)

    # ---- floor from the real train half, refitted per DGP seed ----
    logp_tr = log_density_X(Xtr, cfg)
    assert np.isfinite(logp_tr).all()
    q1, q5, q10 = np.percentile(logp_tr, [1, 5, 10])
    floors[str(dgp_seed)] = {"n": n_train, "p1": float(q1), "p5": float(q5),
                             "p10": float(q10)}

    # ---- real holdout: the null. Independent of the copula draw. ----
    logp_ho = log_density_X(Xho, cfg)
    records.append({"key": "real", "dgp_seed": dgp_seed, "copula_seed": np.nan,
                    **profile(logp_ho, q1, q5, q10)})

    # sanity: R7 must never fire on real data
    assert r7_ok(Xho).all(), "R7 fired on real data"
    assert (Xho.Current_Smoker.to_numpy() * (Xho.Cigs_Per_Day.to_numpy() == 0)).sum() == 0

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

        for key, lp in [("r16", lp16), ("r16_r7", lp167), ("r16_r7_pos", lp167_pos)]:
            records.append({"key": key, "dgp_seed": dgp_seed, "copula_seed": s,
                            **profile(lp, q1, q5, q10)})
        diag.append({
            "dgp_seed": dgp_seed,
            "copula_seed": s,
            "on_manifold_share_r16": float(m16.mean()),
            "smoker1_cigs0_share_of_r16": float(((cop.Current_Smoker.to_numpy() == 1)
                                                 & (cop.Cigs_Per_Day.to_numpy() == 0))[m16].mean()),
            "r1_violations_share_all": float(((cop.Current_Smoker.to_numpy() == 0)
                                              & (cop.Cigs_Per_Day.to_numpy() >= 0.5)).mean()),
            "on_manifold_share_r16_r7": float(m167.mean()),
        })
    print("dgp seed", dgp_seed, "done", flush=True)

rec = pd.DataFrame(records)

# per-DGP-seed mean over copula seeds FIRST, then across DGP seeds
per_seed = rec.groupby(["key", "dgp_seed"])[CELLS].mean().reset_index()
mean_ = per_seed.groupby("key")[CELLS].mean()
across_dgp_sd = per_seed.groupby("key")[CELLS].std()
# within a DGP seed, over copula draws, averaged over DGP seeds. `real` has one row per
# DGP seed, so it has no within-seed spread and is absent rather than reported as 0.
across_cop_sd = (rec[rec.key != "real"].groupby(["key", "dgp_seed"])[CELLS].std()
                 .groupby("key").mean())

tab = pd.DataFrame([{"row_set": label, "n": int(round(mean_.loc[key, "n"])),
                     **{c: float(mean_.loc[key, c]) for c in CELLS[1:]}}
                    for key, label in ROW_SETS])

# ---- exact conditions, per DGP seed, checked on every individual fit ----
exact = {}
for dgp_seed in DGP_SEEDS:
    sl = rec[rec.dgp_seed == dgp_seed]
    zero_ok = all(float(v) == 0.0 for k in ("real", "r16_r7", "r16_r7_pos")
                  for v in sl[sl.key == k]["frac_density_zero"])
    a = sl[sl.key == "r16_r7"].sort_values("copula_seed")[CELLS].to_numpy()
    b = sl[sl.key == "r16_r7_pos"].sort_values("copula_seed")[CELLS].to_numpy()
    exact[int(dgp_seed)] = {
        "frac_density_zero_exactly_zero_rows_1_3_4": bool(zero_ok),
        "rows_3_and_4_identical": bool(np.array_equal(a, b)),
        "r7_never_fires_on_real": True,   # asserted above; the run aborts otherwise
    }

sign3 = {int(s): {
    "pre_R7_frac_density_zero_on_manifold":
        float(per_seed[(per_seed.key == "r16") & (per_seed.dgp_seed == s)]["frac_density_zero"].iloc[0]),
    "post_R7_frac_density_zero_on_manifold":
        float(per_seed[(per_seed.key == "r16_r7") & (per_seed.dgp_seed == s)]["frac_density_zero"].iloc[0]),
} for s in DGP_SEEDS}

real_ps = per_seed[per_seed.key == "real"].sort_values("dgp_seed")
real_null = {
    "targets": {"frac_below_p1": 0.01, "frac_below_p5": 0.05, "frac_below_p10": 0.10},
    "by_dgp_seed": {int(r.dgp_seed): {"frac_below_p1": float(r.frac_below_p1),
                                      "frac_below_p5": float(r.frac_below_p5),
                                      "frac_below_p10": float(r.frac_below_p10)}
                    for r in real_ps.itertuples()},
    "mean": {c: float(real_ps[c].mean()) for c in ("frac_below_p1", "frac_below_p5", "frac_below_p10")},
    "sd": {c: float(real_ps[c].std()) for c in ("frac_below_p1", "frac_below_p5", "frac_below_p10")},
}

tab.to_csv(RESULTS / "r7_decomp.csv", index=False)
pd.DataFrame(diag).to_csv(RESULTS / "r7_decomp_diagnostics.csv", index=False)

manifest = {
    "script": "r7_decomp.py",
    "purpose": "Decompose rules-vs-density disagreement; R7 filter-only, dgp.py unedited.",
    "design": ("5 DGP seeds x 5 copula seeds. Calibration solved once at n_probe and "
               "reused across DGP seeds (E0 recipe). Aggregated to a per-DGP-seed mean "
               "over copula seeds first, then across DGP seeds. The real row is computed "
               "once per DGP seed and is absent from across_copula_seed_sd."),
    "dgp_config": {"n_probe": N_PROBE, "n_samples": N_SAMPLES,
                   "dgp_seeds": list(DGP_SEEDS),
                   "solve_binding_rates": {"R3_glucose_ceil": 0.08,
                                           "R4_bp_mandatory": 0.20},
                   "calibration": "solved once, reused across DGP seeds (E0 recipe)",
                   "glucose_base": cfg.glucose_base, "sbp_base": cfg.sbp_base},
    "dgp_manifest_checks_by_dgp_seed": dgp_checks,
    "split": {"test_size": 0.4, "random_state": 0, "stratify": "Heart_Disease_Risk",
              "n_train": n_train, "n_holdout": n_holdout},
    "density_floor": {"source": "real train half, refitted per DGP seed",
                      "by_dgp_seed": floors},
    "copula_seeds": list(COPULA_SEEDS),
    "R7_definition": "Current_Smoker == 1 -> Cigs_Per_Day >= 1 (filter only, not in dgp.RULES)",
    "rules_tol": 0.5,
    "cell_means": {key: {c: float(mean_.loc[key, c]) for c in CELLS}
                   for key, _ in ROW_SETS},
    "across_dgp_seed_sd": {key: {c: (None if pd.isna(across_dgp_sd.loc[key, c])
                                     else float(across_dgp_sd.loc[key, c])) for c in CELLS}
                           for key, _ in ROW_SETS},
    "across_copula_seed_sd": {key: {c: (None if pd.isna(across_cop_sd.loc[key, c])
                                        else float(across_cop_sd.loc[key, c])) for c in CELLS}
                              for key in across_cop_sd.index},
    "sign_condition_3_by_dgp_seed": sign3,
    "exact_conditions_by_dgp_seed": exact,
    "real_all_null": real_null,
    **PROV,
    "outputs": ["results/r7_decomp.csv", "results/r7_decomp_diagnostics.csv"],
}
with open(MANIFESTS / "r7_decomp.json", "w") as f:
    json.dump(manifest, f, indent=2)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print("\n==== %d DGP seeds x %d copula seeds, n_samples=%d ====" %
      (len(list(DGP_SEEDS)), len(list(COPULA_SEEDS)), N_SAMPLES))
print(tab.round(4).to_string(index=False))
print("\nacross-DGP-seed sd (of the five per-seed means):")
print(across_dgp_sd.round(5).to_string())
print("\nacross-copula-seed sd (within a DGP seed, averaged over DGP seeds):")
print(across_cop_sd.round(5).to_string())

print("\nsign condition 3 -- frac_density_zero among on-manifold rows, per DGP seed:")
for s, v in sign3.items():
    ok = v["post_R7_frac_density_zero_on_manifold"] == 0.0
    print(f"  seed {s}: pre-R7 {v['pre_R7_frac_density_zero_on_manifold']:.4f}"
          f"  ->  post-R7 {v['post_R7_frac_density_zero_on_manifold']:.4f}"
          f"   {'ok (exactly 0.0000)' if ok else 'NONZERO'}")

print("\nexact conditions, per DGP seed:")
for s, v in exact.items():
    print(f"  seed {s}: zero-rows-1,3,4 {'PASS' if v['frac_density_zero_exactly_zero_rows_1_3_4'] else 'FAIL'}"
          f" | rows 3==4 {'PASS' if v['rows_3_and_4_identical'] else 'FAIL'}"
          f" | R7 silent on real {'PASS' if v['r7_never_fires_on_real'] else 'FAIL'}")

print("\nreal-data null against 0.01 / 0.05 / 0.10 (five-seed means):")
for c, t in (("frac_below_p1", 0.01), ("frac_below_p5", 0.05), ("frac_below_p10", 0.10)):
    m, sd = real_null["mean"][c], real_null["sd"][c]
    print(f"  {c:16s} {m:.5f} [{sd:.5f}]  target {t:.2f}  delta {m - t:+.5f}")

print()
print(pd.DataFrame(diag).round(4).to_string(index=False))
