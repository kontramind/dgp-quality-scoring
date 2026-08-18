"""
Parameterised structural causal model for evaluating per-record synthetic data quality.
 
Design goals
------------
1. Orthogonal knobs. prevalence, Bayes-optimal AUC and constraint-coupling are set
   independently; none is an emergent side effect of the others.
2. Exact God-view. p_true(x) and H(p_true(x)) are returned per row. These are the
   ground truth for label-consistency scoring and the aleatoric correction.
3. Self-verifying. Every call returns a manifest with realised prevalence, realised
   Bayes AUC, and per-rule binding and violation rates. A rule that never binds
   cannot generate signal; a rule violated on the real data is a specification bug.
 
Structural coefficients are taken from published Mendelian randomization estimates
where available (see COEFFICIENT_PROVENANCE). The cohort age range follows the AHA
PREVENT equations (Khan et al., Circulation 2024;149:430-449), validated for 30-79.
 
Note: the literature reports several of these edges as bidirectional (BMI <-> fasting
glucose; glucose <-> systolic BP). An SCM must be acyclic, so a direction is imposed.
This is a stipulation of the simulation, not an estimate.
"""
 
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import brentq
 
COEFFICIENT_PROVENANCE = {
    "bmi_to_glucose": (
        0.70,
        "mg/dL per kg/m^2; Zou, Talluri & Shete, PLOS ONE 2024, bidirectional MR "
        "(White/Caucasian, 95% CI 0.35-1.05)",
    ),
    "hyperglycaemia_to_sbp": (
        1.76,
        "mmHg for FPG >= 126; NHANES causal analysis, PSM estimate (95% CI 0.58-2.96)",
    ),
    "age_range": ((30, 79), "AHA PREVENT validated range; Khan et al., Circulation 2024"),
}
 
# Variables carrying hard logical constraints. Coupling (lambda) controls how much
# of the outcome logit these contribute relative to the continuous core.
TRAP_VARS = ["Diabetic", "On_BP_Medication", "Is_Pregnant", "Current_Smoker"]
CORE_VARS = ["Age", "BMI", "Fasting_Glucose", "Systolic_BP", "Cigs_Per_Day"]
FEATURES = CORE_VARS + ["Biological_Sex"] + TRAP_VARS[:-1] + ["Current_Smoker"]
 
 
@dataclass
class DGPConfig:
    # --- sweep knobs (the experimental axes) -------------------------------
    coupling: float = 0.5          # lambda in [0,1]: 0 = traps invisible to Y
    bayes_auc_target: float = 0.85 # irreducible separability ceiling
    prevalence: float = 0.10       # P(Y=1)
    n_nuisance: int = 0            # irrelevant extra columns (10 -> 20 -> 40 features)
    nuisance_corr: float = 0.0     # correlation of nuisance cols with the score
 
    # --- cohort / marginals (declared, not tuned for dramatic results) -----
    n_samples: int = 50_000
    seed: int = 0
    age_range: tuple = (30.0, 79.0)
    p_male: float = 0.50
    p_smoker: float = 0.20
    p_pregnant: float = 0.04       # among eligible women; low by design
    preg_age_max: float = 55.0
    sd_bmi: float = 3.0
    glucose_base: float = 95.0   # tune via solve_binding_rates
    sd_glucose: float = 15.0
    sbp_base: float = 110.0
    sd_sbp: float = 12.0
    p_diabetic_midband: float = 0.15   # P(Diabetic|100<=glucose<126)
    meds_age_scale: float = 0.60       # scales P(Meds|SBP<=140) with age
 
    # --- structural coefficients ------------------------------------------
    b_bmi_glucose: float = 0.70    # MR
    b_hyperglyc_sbp: float = 1.76  # MR
    b_age_bmi: float = 0.10
    b_male_bmi: float = 2.00
    b_age_sbp: float = 0.50
    b_bmi_sbp: float = 1.20
    b_smoker_sbp: float = 5.00
 
    # relative weights inside each score component (sign = direction of risk)
    core_weights: dict = field(default_factory=lambda: {
        "Age": 1.00, "BMI": 0.30, "Fasting_Glucose": 0.40,
        "Systolic_BP": 0.60, "Cigs_Per_Day": 0.50})
    trap_weights: dict = field(default_factory=lambda: {
        "Diabetic": 1.00, "On_BP_Medication": -0.80,
        "Is_Pregnant": 0.30, "Current_Smoker": 0.70})
 
 
# ---------------------------------------------------------------------------
# Bayes-optimal AUC, computed in closed form (no label sampling, no O(n^2))
# ---------------------------------------------------------------------------
def expected_bayes_auc(p: np.ndarray) -> float:
    """E[AUC] when scoring with the true probability p, over the label draw.
 
    AUC = sum_{i,j} p_i (1-p_j) 1[p_i > p_j] / (sum p)(sum 1-p).
    Sorting by p turns the double sum into a cumulative sum: O(n log n).
    """
    order = np.argsort(p, kind="stable")
    ps = p[order]
    q = 1.0 - ps
    cum_q = np.concatenate([[0.0], np.cumsum(q)[:-1]])   # sum of (1-p_j) for p_j < p_i
    num = np.sum(ps * cum_q)
    den = np.sum(ps) * np.sum(q)
    return float(num / den)
 
 
def _solve_alpha(z: np.ndarray, gamma: float, target_prev: float) -> float:
    f = lambda a: expit(a + gamma * z).mean() - target_prev
    return brentq(f, -60.0, 60.0, xtol=1e-10)
 
 
def _solve_gamma(z: np.ndarray, target_auc: float, target_prev: float) -> tuple:
    """Jointly solve (gamma, alpha): gamma sets Bayes AUC, alpha sets prevalence."""
    def auc_at(g):
        a = _solve_alpha(z, g, target_prev)
        return expected_bayes_auc(expit(a + g * z)) - target_auc
    lo, hi = 1e-3, 30.0
    if auc_at(hi) < 0:
        raise ValueError(f"bayes_auc_target={target_auc} unreachable for this score.")
    gamma = brentq(auc_at, lo, hi, xtol=1e-8)
    return gamma, _solve_alpha(z, gamma, target_prev)
 
 
def _std(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()
 
 
# ---------------------------------------------------------------------------
# Rule set. Each rule: antecedent mask, and satisfaction mask given antecedent.
# Kept deliberately small; every rule must be violatable by a smooth generator.
# ---------------------------------------------------------------------------
RULES = {
    "R1_smoker_cigs":   ("Current_Smoker == 0 -> Cigs_Per_Day == 0",
                         lambda d: d.Current_Smoker == 0,
                         lambda d: d.Cigs_Per_Day == 0.0),
    "R2_glucose_floor": ("Fasting_Glucose < 100 -> Diabetic == 0",
                         lambda d: d.Fasting_Glucose < 100.0,
                         lambda d: d.Diabetic == 0),
    "R3_glucose_ceil":  ("Fasting_Glucose >= 126 -> Diabetic == 1",
                         lambda d: d.Fasting_Glucose >= 126.0,
                         lambda d: d.Diabetic == 1),
    "R4_bp_mandatory":  ("Systolic_BP > 140 -> On_BP_Medication == 1",
                         lambda d: d.Systolic_BP > 140.0,
                         lambda d: d.On_BP_Medication == 1),
    "R5_anatomical":    ("Biological_Sex == 1 -> Is_Pregnant == 0",
                         lambda d: d.Biological_Sex == 1,
                         lambda d: d.Is_Pregnant == 0),
    "R6_preg_age":      ("Is_Pregnant == 1 -> Age <= 55",
                         lambda d: d.Is_Pregnant == 1,
                         lambda d: d.Age <= 55.0),
}
 
 
def check_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row, per-rule binding and violation flags."""
    out = {}
    for name, (_, ante, cons) in RULES.items():
        a = np.asarray(ante(df))
        out[f"{name}__bound"] = a
        out[f"{name}__violated"] = a & ~np.asarray(cons(df))
    return pd.DataFrame(out, index=df.index)
 
 
def rule_report(df: pd.DataFrame) -> pd.DataFrame:
    flags = check_rules(df)
    rows = []
    for name, (desc, _, _) in RULES.items():
        b = flags[f"{name}__bound"]
        v = flags[f"{name}__violated"]
        rows.append({"rule": name, "statement": desc,
                     "binding_rate": float(b.mean()),
                     "violation_rate": float(v.mean()),
                     "violations": int(v.sum())})
    return pd.DataFrame(rows)
 
 
# ---------------------------------------------------------------------------
def generate(cfg: Optional[DGPConfig] = None, **overrides):
    """Returns (X, godview, manifest).
 
    X        : features + Heart_Disease_Risk. This is what a generator sees.
    godview  : p_true, aleatoric_entropy, score components, per-rule flags.
    manifest : realised properties, for the self-check.
    """
    cfg = cfg or DGPConfig()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"unknown config field: {k}")
        setattr(cfg, k, v)
    if not 0.0 <= cfg.coupling <= 1.0:
        raise ValueError("coupling must be in [0,1]")
 
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_samples
 
    # --- exogenous roots ---------------------------------------------------
    age = rng.uniform(*cfg.age_range, n)
    sex = rng.binomial(1, cfg.p_male, n)                       # 1 = male
    smoker = rng.binomial(1, cfg.p_smoker, n)
 
    eligible = (sex == 0) & (age <= cfg.preg_age_max)
    pregnant = np.where(eligible, rng.binomial(1, cfg.p_pregnant, n), 0)
 
    cigs = np.where(smoker == 1, np.clip(rng.normal(15, 5, n), 1, 40), 0.0)
 
    # --- continuous layer (linear-Gaussian, MR coefficients) ---------------
    bmi = np.clip(25 + cfg.b_age_bmi * (age - 30) + cfg.b_male_bmi * sex
                  + rng.normal(0, cfg.sd_bmi, n), 15, 50)
    glucose = (cfg.glucose_base + cfg.b_bmi_glucose * (bmi - 25)
               + rng.normal(0, cfg.sd_glucose, n))
 
    # --- deterministic flags (the rule layer) ------------------------------
    diabetic = np.where(glucose >= 126.0, 1,
               np.where(glucose < 100.0, 0, rng.binomial(1, cfg.p_diabetic_midband, n)))
 
    sbp = np.clip(cfg.sbp_base + cfg.b_age_sbp * (age - 30) + cfg.b_bmi_sbp * (bmi - 25)
                  + cfg.b_hyperglyc_sbp * (glucose >= 126.0)
                  + cfg.b_smoker_sbp * smoker + rng.normal(0, cfg.sd_sbp, n), 70, 200)
 
    # hard floor above threshold; stochastic below -> a step a smooth model must learn
    p_meds_below = np.clip(cfg.meds_age_scale * (age - 30) / 100.0, 0.0, 1.0)
    meds = np.where(sbp > 140.0, 1, rng.binomial(1, p_meds_below))
 
    df = pd.DataFrame({
        "Age": age, "Biological_Sex": sex, "Is_Pregnant": pregnant,
        "Current_Smoker": smoker, "Cigs_Per_Day": cigs, "BMI": bmi,
        "Fasting_Glucose": glucose, "Diabetic": diabetic,
        "Systolic_BP": sbp, "On_BP_Medication": meds})
 
    # --- target layer: orthogonal knobs ------------------------------------
    s_core = _std(sum(w * _std(df[v].to_numpy(float)) for v, w in cfg.core_weights.items()))
    s_trap = _std(sum(w * _std(df[v].to_numpy(float)) for v, w in cfg.trap_weights.items()))
    lam = cfg.coupling
    z = _std((1 - lam) * s_core + lam * s_trap)
 
    gamma, alpha = _solve_gamma(z, cfg.bayes_auc_target, cfg.prevalence)
    p_true = expit(alpha + gamma * z)
    y = rng.binomial(1, p_true)
    df["Heart_Disease_Risk"] = y
 
    # --- nuisance columns (dimensionality sweep) ---------------------------
    for j in range(cfg.n_nuisance):
        e = rng.standard_normal(n)
        col = (cfg.nuisance_corr * z + np.sqrt(max(0.0, 1 - cfg.nuisance_corr**2)) * e
               if cfg.nuisance_corr else e)
        df.insert(len(df.columns) - 1, f"Nuisance_{j:02d}", col)
 
    # --- God-view ----------------------------------------------------------
    eps = 1e-12
    pc = np.clip(p_true, eps, 1 - eps)
    entropy = -(pc * np.log(pc) + (1 - pc) * np.log(1 - pc))   # nats
    godview = pd.concat([
        pd.DataFrame({"p_true": p_true, "aleatoric_entropy": entropy,
                      "z": z, "s_core": s_core, "s_trap": s_trap}),
        check_rules(df)], axis=1)
 
    rep = rule_report(df)
    manifest = {
        "config": asdict(cfg),
        "solved_gamma": gamma,
        "solved_alpha": alpha,
        "realised_prevalence": float(y.mean()),
        "expected_bayes_auc": expected_bayes_auc(p_true),
        "empirical_bayes_auc": _emp_auc(p_true, y),
        "mean_aleatoric_entropy": float(entropy.mean()),
        "corr_core_trap": float(np.corrcoef(s_core, s_trap)[0, 1]),
        "trap_variance_share": float(np.var(lam * s_trap) / np.var(z)) if lam else 0.0,
        "n_features": int(df.shape[1] - 1),
        "rules": rep.to_dict("records"),
        "total_violations_on_real": int(rep.violations.sum()),
    }
    return df, godview, manifest
 
 
def _emp_auc(score, y):
    from scipy.stats import rankdata
    y = np.asarray(y); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
 
 
def assert_valid(manifest, tol_prev=0.01, tol_auc=0.01):
    """Gate. Raises if the generator does not satisfy its own specification."""
    errs = []
    if manifest["total_violations_on_real"] != 0:
        errs.append(f"{manifest['total_violations_on_real']} rule violations on real data")
    if abs(manifest["realised_prevalence"] - manifest["config"]["prevalence"]) > tol_prev:
        errs.append(f"prevalence {manifest['realised_prevalence']:.4f} off target")
    if abs(manifest["expected_bayes_auc"] - manifest["config"]["bayes_auc_target"]) > tol_auc:
        errs.append(f"Bayes AUC {manifest['expected_bayes_auc']:.4f} off target")
    dead = [r["rule"] for r in manifest["rules"] if r["binding_rate"] < 0.01]
    if errs:
        raise AssertionError("; ".join(errs))
    return dead
 
 
def solve_binding_rates(targets: dict, cfg: Optional[DGPConfig] = None,
                        n_probe: int = 40_000, tol: float = 2e-3) -> DGPConfig:
    """Tune marginal locations so chosen rules bind at declared rates.
 
    Boundary pressure becomes a stated design parameter rather than an accident.
    Supported: 'R3_glucose_ceil' (via glucose_base), 'R4_bp_mandatory' (via sbp_base).
    """
    import copy
    cfg = copy.deepcopy(cfg or DGPConfig())
    cfg.n_samples = n_probe
    knob = {"R3_glucose_ceil": "glucose_base", "R4_bp_mandatory": "sbp_base"}
    for rule, target in targets.items():
        if rule not in knob:
            raise KeyError(f"no knob registered for {rule}")
        field_name = knob[rule]
        def gap(v, _f=field_name, _r=rule, _t=target):
            setattr(cfg, _f, float(v))
            df, _, _ = generate(cfg)
            return float(np.asarray(RULES[_r][1](df)).mean()) - _t
        lo, hi = getattr(cfg, field_name) - 40.0, getattr(cfg, field_name) + 60.0
        if gap(lo) * gap(hi) > 0:
            raise ValueError(f"target binding rate {target} unreachable for {rule}")
        setattr(cfg, field_name, float(brentq(gap, lo, hi, xtol=tol)))
    return cfg
 