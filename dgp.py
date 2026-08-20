"""Data-generating process for a binary ``Heart_Disease_Risk`` classification cohort.

Independent blind implementation written from the written specification.

Layers
------
1. **Cohort** -- all features are drawn first and are completely independent of the
   label layer, so the coupling knob ``lambda`` cannot change any ``X`` column.
2. **Label** -- two standardised composite scores (``s_core``, ``s_trap``) are blended
   by ``lambda`` into ``z``; ``(gamma, alpha)`` are then jointly solved so that the
   realised prevalence and the closed-form Bayes-optimal AUC both hit their targets.
3. **Rules** -- seven hard logical constraints, checked per row. On real (unperturbed)
   data they must never be violated.

``generate()`` returns ``(df, godview, manifest)``; the manifest is self-verifying via
:func:`assert_valid`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import rankdata

__all__ = [
    "DGPConfig",
    "generate",
    "assert_valid",
    "solve_binding_rates",
    "bayes_auc",
    "bayes_auc_naive",
    "evaluate_rules",
    "RULES",
    "COEFFICIENT_PROVENANCE",
    "COHORT_STIPULATIONS",
]

# --------------------------------------------------------------------------------------
# Provenance: what is sourced, what is stipulated
# --------------------------------------------------------------------------------------
#
# Both structures below are inert documentation. Nothing reads them at runtime and they
# cannot perturb generation. They exist because the paper's claim is that *only some* of
# this model's numbers are literature-backed, and that claim is only checkable if the
# rest are enumerated rather than left to inference from silence.
#
# COEFFICIENT_PROVENANCE is copied verbatim from reference/dgp_frozen.py -- the citation
# wording there is the only authority for it, and the copy is text, not a runtime
# dependency on the frozen reference. Do not reword, reformat or "tidy" a citation
# string. The values are duplicated from the DGPConfig defaults on purpose:
# test_dgp.py::test_coefficient_provenance_matches_config asserts they agree, so the
# documentation cannot drift away from the code it documents.

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

# `age_range` above is the one entry whose citation covers only half of what the config
# does with it, and it must be read that way:
#
#   * The AHA PREVENT citation licenses the **window**. The equations are validated for
#     ages 30-79, and that is what "validated range" means.
#   * It says **nothing about the distribution inside that window**. The DGP draws
#     `Age ~ Uniform(30, 79)` -- a stipulation chosen for even coverage of the validated
#     range, not for population realism. No cohort is uniform in age. A reader who sees
#     only the citation will infer otherwise, which is why the split is written down here
#     and repeated in COHORT_STIPULATIONS.
#
# `age_range` is also **not a free knob**: the calibration constants `glucose_base` and
# `sbp_base` are solved by solve_binding_rates() conditional on the age window, and the
# rule layer is not orthogonalized against it. Changing the window without re-running the
# calibration silently invalidates every declared binding rate. Sweeping the window on the
# frozen implementation while holding the seed-0 calibration fixed moves R4 from 0.2017 at
# (30, 79) to 0.2467 at (40, 79) and 0.1328 at (30, 69) -- against a declared band of
# 0.200 +/- 0.005, roughly nine and fourteen band widths out. Prevalence and Bayes AUC are
# unmoved, because those are solved for; the binding rates are not. Contrast `p_smoker`,
# which leaves R2 at 0.378860 across its whole range. (Frozen's digits, quoted as
# mechanism; canonical's differ and nothing here is a target.)

# Knobs that fix the composition of the simulated cohort. Every one is **declared, not
# estimated**: none is an estimate of any real population and none may be cited to a
# prevalence source. No result in this project depends on a marginal's epidemiological
# accuracy -- the marginals exist to give each rule enough binding mass to carry signal,
# which is a design requirement rather than a modelling claim. (`p_smoker = 0.20` is
# roughly double contemporary US adult cigarette-smoking prevalence; sourcing it would
# invite an audit of this DGP as a cardiovascular model, which it is not.)
#
# Notes only, no values: a second copy of the defaults would be a second drift surface,
# and unlike COEFFICIENT_PROVENANCE there is no citation here for a value to drift away
# from.
COHORT_STIPULATIONS: dict[str, str] = {
    "p_male": "sex split of the cohort; set to 0.50 for balance, not measured",
    "p_smoker": "smoking prevalence; chosen to give R1 and R7 binding mass on both sides",
    "p_pregnant": "pregnancy rate among eligible women; sets R6's binding mass",
    "preg_age_max": "upper age for pregnancy eligibility; the domain constant R6 is stated over",
    "sd_bmi": "residual spread of BMI about its structural mean",
    "sd_glucose": "residual spread of Fasting_Glucose about its structural mean",
    "sd_sbp": "residual spread of Systolic_BP about its structural mean",
    "p_diabetic_midband": "P(Diabetic = 1 | 100 <= glucose < 126); the one non-deterministic band",
    "meds_age_scale": "age slope of medication uptake below the SBP > 140 hard floor",
    "age_range": (
        "SOURCED WINDOW, STIPULATED SHAPE. The 30-79 window is cited in "
        "COEFFICIENT_PROVENANCE; the Uniform distribution within it is not, and the "
        "calibration constants are solved conditional on the window. See the note above."
    ),
}

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

CORE_WEIGHTS: dict[str, float] = {
    "Age": 1.00,
    "BMI": 0.30,
    "Fasting_Glucose": 0.40,
    "Systolic_BP": 0.60,
    "Cigs_Per_Day": 0.50,
}

TRAP_WEIGHTS: dict[str, float] = {
    "Diabetic": 1.00,
    "On_BP_Medication": -0.80,
    "Is_Pregnant": 0.30,
    "Current_Smoker": 0.70,
}

FEATURE_ORDER: list[str] = [
    "Age",
    "Biological_Sex",
    "Current_Smoker",
    "Is_Pregnant",
    "Cigs_Per_Day",
    "BMI",
    "Fasting_Glucose",
    "Diabetic",
    "Systolic_BP",
    "On_BP_Medication",
]

LABEL = "Heart_Disease_Risk"


@dataclass
class DGPConfig:
    """Every knob of the DGP. All fields are overridable via keyword arguments."""

    coupling: float = 0.5              # lambda in [0, 1]: weight on trap vars in the logit
    bayes_auc_target: float = 0.85     # ceiling AUC of the true probability
    prevalence: float = 0.10           # target P(Y = 1)
    n_nuisance: int = 0
    nuisance_corr: float = 0.0
    n_samples: int = 50_000
    seed: int = 0

    age_range: tuple[float, float] = (30.0, 79.0)
    p_male: float = 0.50
    p_smoker: float = 0.20
    p_pregnant: float = 0.04           # among eligible women
    preg_age_max: float = 55.0
    sd_bmi: float = 3.0
    glucose_base: float = 95.0         # tuned by solve_binding_rates()
    sd_glucose: float = 15.0
    sbp_base: float = 110.0            # tuned by solve_binding_rates()
    sd_sbp: float = 12.0
    p_diabetic_midband: float = 0.15   # P(Diabetic = 1 | 100 <= glucose < 126)
    meds_age_scale: float = 0.60

    b_bmi_glucose: float = 0.70        # MR estimate, mg/dL per kg/m^2
    b_hyperglyc_sbp: float = 1.76      # MR estimate, mmHg
    b_age_bmi: float = 0.10
    b_male_bmi: float = 2.00
    b_age_sbp: float = 0.50
    b_bmi_sbp: float = 1.20
    b_smoker_sbp: float = 5.00

    core_weights: dict[str, float] = field(default_factory=lambda: dict(CORE_WEIGHTS))
    trap_weights: dict[str, float] = field(default_factory=lambda: dict(TRAP_WEIGHTS))


def _copy_config(cfg: DGPConfig | None, **overrides: Any) -> DGPConfig:
    """Return an independent copy of ``cfg`` (or a default config) with overrides applied."""
    base = DGPConfig() if cfg is None else cfg
    out = replace(base, **overrides)
    # `replace` shares the weight dicts with the source; detach them so mutation is safe.
    out.core_weights = dict(out.core_weights)
    out.trap_weights = dict(out.trap_weights)
    return out


# --------------------------------------------------------------------------------------
# Cohort generation
# --------------------------------------------------------------------------------------

def _generate_cohort(cfg: DGPConfig) -> tuple[dict[str, np.ndarray], np.random.Generator]:
    """Draw every feature column. Returns the columns and the live RNG.

    Split out from :func:`generate` so that :func:`solve_binding_rates` can root-find on
    the features alone -- binding rates never depend on the label layer.
    """
    rng = np.random.default_rng(cfg.seed)
    n = int(cfg.n_samples)
    age_lo, age_hi = cfg.age_range

    age = rng.uniform(age_lo, age_hi, n)
    sex = (rng.random(n) < cfg.p_male).astype(np.int64)              # 1 = male
    smoker = (rng.random(n) < cfg.p_smoker).astype(np.int64)

    eligible = (sex == 0) & (age <= cfg.preg_age_max)
    pregnant = (eligible & (rng.random(n) < cfg.p_pregnant)).astype(np.int64)

    # Draw the full-length normal unconditionally so RNG consumption is shape-stable.
    cigs = np.where(smoker == 1, np.clip(rng.normal(15.0, 5.0, n), 1.0, 40.0), 0.0)

    bmi = np.clip(
        25.0 + cfg.b_age_bmi * (age - 30.0) + cfg.b_male_bmi * sex
        + rng.normal(0.0, cfg.sd_bmi, n),
        15.0, 50.0,
    )

    glucose = cfg.glucose_base + cfg.b_bmi_glucose * (bmi - 25.0) + rng.normal(0.0, cfg.sd_glucose, n)

    hyperglycaemic = glucose >= 126.0
    midband_draw = rng.random(n) < cfg.p_diabetic_midband
    diabetic = np.where(
        hyperglycaemic, True, np.where(glucose < 100.0, False, midband_draw)
    ).astype(np.int64)

    sbp = np.clip(
        cfg.sbp_base
        + cfg.b_age_sbp * (age - 30.0)
        + cfg.b_bmi_sbp * (bmi - 25.0)
        + cfg.b_hyperglyc_sbp * hyperglycaemic
        + cfg.b_smoker_sbp * smoker
        + rng.normal(0.0, cfg.sd_sbp, n),
        70.0, 200.0,
    )

    meds_draw = rng.random(n) < np.clip(cfg.meds_age_scale * (age - 30.0) / 100.0, 0.0, 1.0)
    meds = np.where(sbp > 140.0, True, meds_draw).astype(np.int64)   # SBP > 140 is a hard floor

    cols = {
        "Age": age,
        "Biological_Sex": sex,
        "Current_Smoker": smoker,
        "Is_Pregnant": pregnant,
        "Cigs_Per_Day": cigs,
        "BMI": bmi,
        "Fasting_Glucose": glucose,
        "Diabetic": diabetic,
        "Systolic_BP": sbp,
        "On_BP_Medication": meds,
    }
    return cols, rng


# --------------------------------------------------------------------------------------
# Score construction
# --------------------------------------------------------------------------------------

def _standardize(x: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance (population sd, ddof=0)."""
    x = np.asarray(x, dtype=float)
    sd = float(x.std())
    if not np.isfinite(sd) or sd <= 0.0:
        raise ValueError("cannot standardize a constant or non-finite vector")
    return (x - x.mean()) / sd


def _weighted_score(cols: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    """Standardize each component, combine with weights, re-standardize the combination."""
    acc: np.ndarray | None = None
    for name, w in weights.items():
        term = w * _standardize(cols[name])
        acc = term if acc is None else acc + term
    if acc is None:
        raise ValueError("weights must not be empty")
    return _standardize(acc)


# --------------------------------------------------------------------------------------
# Closed-form Bayes AUC
# --------------------------------------------------------------------------------------

def bayes_auc(p: np.ndarray) -> float:
    """Exact expected AUC of ``p`` as a score for labels ``Y_i ~ Bernoulli(p_i)``.

        AUC = [ sum_ij p_i(1-p_j) 1[p_i > p_j]
                + 1/2 sum_{i != j} p_i(1-p_j) 1[p_i == p_j] ] / [ (sum p)(sum 1-p) ]

    Tied pairs take **half credit** (Mann-Whitney convention). Because ``p_i == p_j``
    on a tie implies ``p_i(1-p_j) == p_j(1-p_i)``, half credit on both orderings equals
    full credit counted once -- so accumulating "everything at an earlier sorted
    position" is already exactly right and ties must *not* be special-cased out.

    O(n log n). Cross-checked against :func:`bayes_auc_naive`.
    """
    p = np.asarray(p, dtype=float).ravel()
    order = np.argsort(p, kind="stable")
    ps = p[order]
    q = 1.0 - ps
    cum_below = np.concatenate(([0.0], np.cumsum(q)[:-1]))
    denom = float(ps.sum()) * float(q.sum())
    if denom <= 0.0:
        raise ValueError("bayes_auc is undefined when every p is 0 or every p is 1")
    return float(ps @ cum_below) / denom


def bayes_auc_naive(p: np.ndarray) -> float:
    """Obviously-correct O(n^2) double sum. Reference only -- used to test :func:`bayes_auc`."""
    p = np.asarray(p, dtype=float).ravel()
    n = p.size
    num = 0.0
    for i in range(n):
        for j in range(n):
            if p[i] > p[j]:
                num += p[i] * (1.0 - p[j])
            elif i != j and p[i] == p[j]:
                num += 0.5 * p[i] * (1.0 - p[j])
    denom = float(p.sum()) * float((1.0 - p).sum())
    if denom <= 0.0:
        raise ValueError("bayes_auc is undefined when every p is 0 or every p is 1")
    return num / denom


def _empirical_auc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney / rank estimator of AUC for sampled labels."""
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(np.asarray(score, dtype=float))
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# --------------------------------------------------------------------------------------
# Joint (gamma, alpha) solve
# --------------------------------------------------------------------------------------

_GAMMA_MAX = 30.0     # hard ceiling on the logit slope searched by the outer root-find
_GAMMA_MIN = 1e-8
_ALPHA_PAD = 1e-9


class _AlphaSolveError(RuntimeError):
    """Internal: the alpha root-find could not bracket the requested prevalence."""


def _solve_alpha(z: np.ndarray, gamma: float, prevalence: float) -> float:
    """Solve ``mean(expit(alpha + gamma z)) == prevalence`` for alpha (monotone in alpha).

    The bracket is **analytically guaranteed and scales with gamma**: at
    ``alpha = logit(prev) - gamma*max|z|`` every ``p_i <= prev`` so the mean is at most
    ``prev``, and at ``alpha = logit(prev) + gamma*max|z|`` every ``p_i >= prev`` so the
    mean is at least ``prev``. A *fixed* bracket is the trap here: the alpha a low
    prevalence needs grows like ``gamma * max|z|``, so once the score has a long tail
    (coupling = 1 is the bad case) a fixed bracket cannot reach it, the reachable mean
    acquires a floor above the target, and the prevalence looks infeasible when it is not.
    Measured z range, required alpha and the resulting floor are in
    ``manifests/s0_coupling1_ceiling.json`` under ``alpha_bracket_diagnostic``.
    """
    target_logit = float(logit(prevalence))
    if gamma <= 0.0:
        return target_logit

    span = gamma * float(np.max(np.abs(z)))
    lo = target_logit - span - _ALPHA_PAD
    hi = target_logit + span + _ALPHA_PAD

    def f(a: float) -> float:
        return float(np.mean(expit(a + gamma * z))) - prevalence

    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0.0 or f_hi < 0.0:
        raise _AlphaSolveError(
            f"cannot bracket prevalence={prevalence} at gamma={gamma}: "
            f"mean(p) spans [{f_lo + prevalence:.6f}, {f_hi + prevalence:.6f}]"
        )
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    return float(brentq(f, lo, hi, xtol=1e-10, rtol=1e-13, maxiter=200))


def _solve_gamma_alpha(z: np.ndarray, prevalence: float, target_auc: float) -> tuple[float, float]:
    """Jointly solve for the logit slope and intercept.

    Nested 1-D root-find: for a candidate gamma the inner solve pins alpha to the exact
    prevalence, and the outer solve moves gamma until the closed-form Bayes AUC hits the
    target (monotone increasing in gamma given the inner solve).
    """
    if not 0.0 < prevalence < 1.0:
        raise ValueError(f"prevalence must lie in (0, 1), got {prevalence!r}")
    if not 0.0 < target_auc < 1.0:
        raise ValueError(f"bayes_auc_target must lie in (0, 1), got {target_auc!r}")
    if target_auc <= 0.5:
        raise ValueError(
            f"bayes_auc_target={target_auc!r} unreachable for this score "
            "(a monotone score cannot be scheduled below chance, AUC 0.5)"
        )

    def auc_at(g: float) -> float:
        return bayes_auc(expit(_solve_alpha(z, g, prevalence) + g * z))

    # Single translation point for alpha-solve failures. The bracket in _solve_alpha is
    # analytically guaranteed, so this cannot trigger through the public API -- it exists
    # only so a bracketing failure can never escape as a raw brentq message, and is
    # exercised by test_alpha_failure_is_translated via monkeypatch.
    try:
        auc_hi = auc_at(_GAMMA_MAX)
        if target_auc >= auc_hi:
            raise ValueError(
                f"bayes_auc_target={target_auc!r} unreachable for this score "
                f"(max achievable {auc_hi:.4f} at gamma={_GAMMA_MAX:.3f})"
            )

        auc_lo = auc_at(_GAMMA_MIN)
        if target_auc <= auc_lo:
            raise ValueError(
                f"bayes_auc_target={target_auc!r} unreachable for this score "
                f"(min achievable {auc_lo:.4f} at gamma={_GAMMA_MIN:g})"
            )

        gamma = float(
            brentq(lambda g: auc_at(g) - target_auc, _GAMMA_MIN, _GAMMA_MAX,
                   xtol=1e-10, rtol=1e-13, maxiter=200)
        )
        alpha = _solve_alpha(z, gamma, prevalence)
    except _AlphaSolveError as exc:
        raise ValueError(
            f"prevalence={prevalence!r} is not attainable for this score: {exc}"
        ) from exc

    return gamma, alpha


# --------------------------------------------------------------------------------------
# Rule set
# --------------------------------------------------------------------------------------

# name -> (statement, antecedent_fn, consequent_fn). Adding a rule is one entry: the set
# is read late everywhere, and nothing downstream of it consumes RNG.
# The 55 in R6 is the domain constant the rule is stated over; it coincides with the
# default `preg_age_max`, which is what makes the rule hold by construction.
RULES: dict[str, tuple[str, Callable[[Mapping[str, np.ndarray]], np.ndarray],
                       Callable[[Mapping[str, np.ndarray]], np.ndarray]]] = {
    "R1_smoker_cigs": (
        "Current_Smoker == 0 -> Cigs_Per_Day == 0",
        lambda d: np.asarray(d["Current_Smoker"]) == 0,
        lambda d: np.asarray(d["Cigs_Per_Day"]) == 0,
    ),
    "R2_glucose_floor": (
        "Fasting_Glucose < 100 -> Diabetic == 0",
        lambda d: np.asarray(d["Fasting_Glucose"]) < 100.0,
        lambda d: np.asarray(d["Diabetic"]) == 0,
    ),
    "R3_glucose_ceil": (
        "Fasting_Glucose >= 126 -> Diabetic == 1",
        lambda d: np.asarray(d["Fasting_Glucose"]) >= 126.0,
        lambda d: np.asarray(d["Diabetic"]) == 1,
    ),
    "R4_bp_mandatory": (
        "Systolic_BP > 140 -> On_BP_Medication == 1",
        lambda d: np.asarray(d["Systolic_BP"]) > 140.0,
        lambda d: np.asarray(d["On_BP_Medication"]) == 1,
    ),
    "R5_anatomical": (
        "Biological_Sex == 1 -> Is_Pregnant == 0",
        lambda d: np.asarray(d["Biological_Sex"]) == 1,
        lambda d: np.asarray(d["Is_Pregnant"]) == 0,
    ),
    "R6_preg_age": (
        "Is_Pregnant == 1 -> Age <= 55",
        lambda d: np.asarray(d["Is_Pregnant"]) == 1,
        lambda d: np.asarray(d["Age"]) <= 55.0,
    ),
    # R1 is one-directional: it catches `smoker == 0 & cigs > 0` and never
    # `smoker == 1 & cigs == 0`. That gap is the sole cause of the exact-zero-density
    # rows found among R1-R6-clean copula output (12.6% of them; results/r7_decomp.csv).
    # R7 closes the other direction, so it is part of the definition of on-manifold
    # rather than a filter -- which is why it lives here and not in a runner.
    #
    # The consequent is exact `>= 1.0` and carries no tolerance. Under the SCM
    # `cigs = where(smoker == 1, clip(N(15, 5), 1, 40), 0.0)`, so the clip lower bound
    # returns literal 1.0 and a smoker's Cigs_Per_Day is exactly >= 1.0 with no float
    # hazard. `Cigs in (0, 1)` moreover has zero density for every row -- non-smokers sit
    # on the atom at 0, smokers on [1, 40] -- so the rule edge sits exactly on the support
    # edge and a tolerance would forgive a strip that is structurally impossible.
    # See rules_tol.py: R1's tolerance is a patch for exact equality to a point mass on a
    # continuous column; R7's consequent is a half-line and needs no such patch.
    "R7_smoker_cigs_min": (
        "Current_Smoker == 1 -> Cigs_Per_Day >= 1",
        lambda d: np.asarray(d["Current_Smoker"]) == 1,
        lambda d: np.asarray(d["Cigs_Per_Day"]) >= 1.0,
    ),
}


def evaluate_rules(
    data: Mapping[str, np.ndarray]
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], int]:
    """Evaluate every rule. Returns ``(per_rule_summaries, per_row_flags, total_violations)``.

    A row is *binding* for a rule when its antecedent holds, and *violated* when the
    antecedent holds but the consequent does not. ``violation_rate`` is over all rows
    (not conditional on binding).
    """
    n = len(np.asarray(data["Age"]))
    summaries: list[dict[str, Any]] = []
    flags: dict[str, np.ndarray] = {}
    total = 0

    for name, (statement, antecedent, consequent) in RULES.items():
        binding = np.asarray(antecedent(data), dtype=bool)
        violated = binding & ~np.asarray(consequent(data), dtype=bool)
        flags[f"{name}__binding"] = binding
        flags[f"{name}__violated"] = violated
        n_violations = int(violated.sum())
        total += n_violations
        summaries.append(
            {
                "name": name,
                "statement": statement,
                "binding_rate": float(binding.mean()),
                "violation_rate": float(n_violations / n),
                "violations": n_violations,
            }
        )
    return summaries, flags, total


# --------------------------------------------------------------------------------------
# Top-level generation
# --------------------------------------------------------------------------------------

def generate(
    cfg: DGPConfig | None = None, **overrides: Any
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Simulate a cohort. Returns ``(df, godview, manifest)``."""
    cfg = _copy_config(cfg, **overrides)
    n = int(cfg.n_samples)

    cols, rng = _generate_cohort(cfg)

    s_core = _weighted_score(cols, cfg.core_weights)
    s_trap = _weighted_score(cols, cfg.trap_weights)
    lam = float(cfg.coupling)
    z = _standardize((1.0 - lam) * s_core + lam * s_trap)

    df = pd.DataFrame({name: cols[name] for name in FEATURE_ORDER})

    if cfg.n_nuisance > 0:
        rho = float(cfg.nuisance_corr)
        scale = float(np.sqrt(1.0 - rho * rho))
        for k in range(int(cfg.n_nuisance)):
            e = rng.standard_normal(n)
            df[f"nuisance_{k:02d}"] = rho * z + scale * e if rho != 0.0 else e

    gamma, alpha = _solve_gamma_alpha(z, cfg.prevalence, cfg.bayes_auc_target)
    p_true = expit(alpha + gamma * z)
    y = (rng.random(n) < p_true).astype(np.int64)
    df[LABEL] = y

    summaries, flags, total_violations = evaluate_rules(cols)

    p_clipped = np.clip(p_true, 1e-12, 1.0 - 1e-12)
    entropy = -(p_clipped * np.log(p_clipped) + (1.0 - p_clipped) * np.log(1.0 - p_clipped))

    godview = pd.DataFrame(
        {
            "p_true": p_true,
            "aleatoric_entropy": entropy,
            "z": z,
            "s_core": s_core,
            "s_trap": s_trap,
            **flags,
        }
    )

    manifest: dict[str, Any] = {
        "config": asdict(cfg),
        "solved_gamma": float(gamma),
        "solved_alpha": float(alpha),
        "realised_prevalence": float(y.mean()),
        "expected_bayes_auc": bayes_auc(p_true),
        "empirical_bayes_auc": _empirical_auc(p_true, y),
        "mean_aleatoric_entropy": float(entropy.mean()),
        "corr_core_trap": float(np.corrcoef(s_core, s_trap)[0, 1]),
        # z is standardized, so this is exactly lambda^2.
        "trap_variance_share": 0.0 if lam == 0.0 else float(np.var(lam * s_trap) / np.var(z)),
        "n_features": int(df.shape[1] - 1),
        "rules": summaries,
        "total_violations_on_real": total_violations,
    }
    return df, godview, manifest


# --------------------------------------------------------------------------------------
# Manifest validation
# --------------------------------------------------------------------------------------

# A rule binding on fewer than 0.5% of rows carries too little mass to be worth testing.
#
# The spec's original 1% threshold sat exactly on R6_preg_age's expected binding rate
# (P(pregnant) ~= 0.0102), making its dead/alive status a coin flip across seeds -- the
# reference flips True/False/True/False/False over seeds 0-4. Lowering the *diagnostic*
# threshold to 0.005 puts R6 ~11 SE clear of it (SE ~= 0.00045 at n=50000) and settles
# the question. The alternative -- raising `p_pregnant` until R6 binds harder -- was
# rejected because Is_Pregnant is a trap variable, so it would move s_trap, z, gamma,
# alpha and every E0 golden value to fix what is only a reporting threshold.
_DEAD_RULE_THRESHOLD = 0.005


def assert_valid(
    manifest: Mapping[str, Any], tol_prev: float = 0.01, tol_auc: float = 0.01
) -> list[str]:
    """Raise one ``AssertionError`` listing every failure; return the list of dead rules.

    A rule is *dead* when its binding rate falls below ``_DEAD_RULE_THRESHOLD`` --
    reported, never fatal.
    """
    cfg = manifest["config"]
    failures: list[str] = []

    if manifest["total_violations_on_real"] != 0:
        failures.append(
            f"total_violations_on_real = {manifest['total_violations_on_real']}, expected 0"
        )

    d_prev = abs(manifest["realised_prevalence"] - cfg["prevalence"])
    if d_prev > tol_prev:
        failures.append(
            f"realised_prevalence = {manifest['realised_prevalence']:.6f} is "
            f"{d_prev:.6f} from target {cfg['prevalence']} (tol {tol_prev})"
        )

    d_auc = abs(manifest["expected_bayes_auc"] - cfg["bayes_auc_target"])
    if d_auc > tol_auc:
        failures.append(
            f"expected_bayes_auc = {manifest['expected_bayes_auc']:.6f} is "
            f"{d_auc:.6f} from target {cfg['bayes_auc_target']} (tol {tol_auc})"
        )

    if failures:
        raise AssertionError("manifest failed validation:\n  - " + "\n  - ".join(failures))

    return [r["name"] for r in manifest["rules"] if r["binding_rate"] < _DEAD_RULE_THRESHOLD]


# --------------------------------------------------------------------------------------
# Binding-rate calibration
# --------------------------------------------------------------------------------------

_KNOB_REGISTRY: dict[str, str] = {
    "R3_glucose_ceil": "glucose_base",
    "R4_bp_mandatory": "sbp_base",
}

_KNOB_WINDOW: dict[str, float] = {"glucose_base": 40.0, "sbp_base": 60.0}


def solve_binding_rates(
    targets: Mapping[str, float],
    cfg: DGPConfig | None = None,
    n_probe: int = 40_000,
    tol: float = 2e-3,
) -> DGPConfig:
    """Tune generation knobs so each named rule binds at its target rate.

    .. warning::
       **The returned config carries ``n_samples = n_probe``, not the caller's original
       value.** The search runs at ``n_probe``, and the returned config is the config that
       was searched. Callers who want production-size data MUST reset ``n_samples``
       themselves before calling :func:`generate`::

           cfg = solve_binding_rates({"R3_glucose_ceil": 0.08, "R4_bp_mandatory": 0.20})
           cfg.n_samples = 50_000          # <- required; not done for you
           df, godview, manifest = generate(cfg)

    .. note::
       Rules are solved in the caller's dict order, and that order matters:
       ``b_hyperglyc_sbp`` makes Systolic_BP depend on the glucose threshold, so
       ``R3_glucose_ceil`` must be listed before ``R4_bp_mandatory``.

    Binding rates depend only on the features, so the search regenerates the cohort
    alone and never runs the label layer.
    """
    work = _copy_config(cfg, n_samples=int(n_probe))

    for rule, target in targets.items():
        if rule not in _KNOB_REGISTRY:
            raise KeyError(
                f"no calibration knob registered for rule {rule!r}; "
                f"known rules: {sorted(_KNOB_REGISTRY)}"
            )
        knob = _KNOB_REGISTRY[rule]
        antecedent = RULES[rule][1]
        centre = float(getattr(work, knob))
        window = _KNOB_WINDOW[knob]

        def rate_error(x: float, _knob: str = knob, _ante=antecedent, _target: float = target) -> float:
            setattr(work, _knob, float(x))
            probe_cols, _ = _generate_cohort(work)
            return float(np.asarray(_ante(probe_cols), dtype=bool).mean()) - _target

        root = brentq(rate_error, centre - window, centre + window, xtol=tol, maxiter=200)
        setattr(work, knob, float(root))

    return work
