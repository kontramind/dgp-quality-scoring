"""Differential agreement against ``reference/dgp_frozen.py``.

The reference is the prior implementation, kept read-only; nothing imports it outside
this test and ``scripts/export_differential.py``. Neither implementation is adapted to
the other -- the naming differences are bridged here:

    ours                        reference
    ----                        ---------
    bayes_auc()                 expected_bayes_auc()
    evaluate_rules()            check_rules() / rule_report()
    "<rule>__binding"           "<rule>__bound"
    manifest rules[i]["name"]   manifest rules[i]["rule"]

Row-by-row comparison is impossible: the two draw from the RNG in different orders (we
use ``rng.random(n) < p``, the reference uses ``rng.binomial(1, p, n)``), so identical
seeds give different samples. This file therefore covers only the **deterministic**
layer, where agreement must be exact. The stochastic pipeline is compared
distributionally by ``scripts/export_differential.py`` into ``results/s0_differential.csv``.

**One rule deliberately diverges, and the divergence is asserted, not tolerated.**
Canonical has seven rules; the frozen reference has six. ``R7_smoker_cigs_min``
(``Current_Smoker == 1 -> Cigs_Per_Day >= 1``) was appended to ``dgp.RULES`` to close the
direction ``R1_smoker_cigs`` leaves open, and the reference is frozen so it will never
gain it. The contract here is therefore *not* "canonical and frozen agree" but
"**identical on the six shared rules, and R7 is the sole divergence**": the per-rule
comparisons run over ``set(dgp.RULES) & set(ref.RULES)``, and the set comparison asserts
the difference is exactly ``{"R7_smoker_cigs_min"}`` in one direction and empty in the
other. A *seventh* divergence, or R7 silently vanishing, still fails. This is not a
bug to be "fixed" back to a plain set equality -- a carve-out that cannot fail is worse
than no differential test at all.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

import dgp
from dgp import DGPConfig, bayes_auc, bayes_auc_naive, evaluate_rules, generate

_REF_PATH = Path(__file__).resolve().parent.parent / "reference" / "dgp_frozen.py"


def _load_reference():
    spec = importlib.util.spec_from_file_location("dgp_frozen", _REF_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = _load_reference()

# The comparison set is the intersection, not `list(dgp.RULES)`: R7 exists only in
# canonical, so comparing it against the reference would KeyError. The divergence itself
# is asserted separately in test_rule_summaries_match_reference.
RULE_NAMES = [r for r in dgp.RULES if r in ref.RULES]
CANONICAL_ONLY = {"R7_smoker_cigs_min"}
EXACT = 1e-12


# --------------------------------------------------------------------------------------
# Layer 1a -- the closed-form Bayes AUC. This is where the tie-convention error lived,
# so it is the comparison that matters most.
# --------------------------------------------------------------------------------------

def _p_vectors() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "continuous_n500": rng.uniform(0.001, 0.999, 500),
        "continuous_n5000": rng.uniform(0.001, 0.999, 5000),
        "two_distinct_n1000": rng.choice(np.array([0.2, 0.8]), 1000),
        "three_distinct_n5000": rng.choice(np.array([0.1, 0.5, 0.9]), 5000),
        "five_distinct_n5000": rng.choice(np.array([0.05, 0.2, 0.5, 0.8, 0.95]), 5000),
        "sixteen_distinct_n5000": rng.choice(np.linspace(0.05, 0.95, 16), 5000),
        "all_identical_n500": np.full(500, 0.3),
        "extreme_spread_n2000": rng.choice(np.array([1e-6, 0.5, 1 - 1e-6]), 2000),
    }


@pytest.mark.parametrize("case", sorted(_p_vectors()))
def test_bayes_auc_matches_reference(case):
    """Exact agreement on shared p-vectors, continuous and heavily tied alike."""
    p = _p_vectors()[case]
    ours, theirs = bayes_auc(p), ref.expected_bayes_auc(p)
    assert ours == pytest.approx(theirs, abs=EXACT), f"{case}: {ours!r} vs {theirs!r}"


def _naive_fsum(p: np.ndarray) -> float:
    """The same O(n^2) double sum, accumulated with math.fsum (exactly rounded)."""
    terms = []
    for i in range(len(p)):
        for j in range(len(p)):
            if p[i] > p[j]:
                terms.append(p[i] * (1.0 - p[j]))
            elif i != j and p[i] == p[j]:
                terms.append(0.5 * p[i] * (1.0 - p[j]))
    return math.fsum(terms) / (float(p.sum()) * float((1.0 - p).sum()))


def test_bayes_auc_naive_matches_both():
    """The O(n^2) double sum agrees with both fast implementations.

    Two tolerances, for a reason. ``bayes_auc_naive`` accumulates n^2 = 250,000 terms
    sequentially, so its floating-point floor is ~n^2 * eps = 5.6e-11 -- it cannot be held
    to EXACT, and loosening it is arithmetic, not a concession. Summing the identical terms
    with math.fsum recovers agreement to ~4e-15, which is what shows the gap is
    accumulation order and not a difference in the tie convention.
    """
    rng = np.random.default_rng(1)
    naive_floor = 1e-9                     # comfortably above n^2 * eps

    for p in (rng.uniform(0.001, 0.999, 500), rng.choice(np.array([0.2, 0.5, 0.8]), 500)):
        assert bayes_auc_naive(p) == pytest.approx(bayes_auc(p), abs=naive_floor)
        assert bayes_auc_naive(p) == pytest.approx(ref.expected_bayes_auc(p), abs=naive_floor)

        # Exact accumulation: the convention itself agrees to machine precision.
        assert _naive_fsum(p) == pytest.approx(bayes_auc(p), abs=EXACT)
        assert _naive_fsum(p) == pytest.approx(ref.expected_bayes_auc(p), abs=EXACT)


# --------------------------------------------------------------------------------------
# Layer 1b -- the rule layer, on a shared dataframe.
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def violating_frame():
    """Real data carries zero violations, which would make the comparison vacuous.

    Each rule is deliberately broken on a slice of rows so both the binding and the
    violated masks are non-trivial -- for all six shared rules, and for R7, which only
    canonical has and which is checked separately for that reason.
    """
    df, _, _ = generate(DGPConfig(n_samples=4000, seed=0))
    df = df.copy()

    def some(mask, k=25):
        return df.index[np.asarray(mask)][:k]

    df.loc[some(df.Current_Smoker == 0), "Cigs_Per_Day"] = 5.0
    df.loc[some(df.Fasting_Glucose < 100.0), "Diabetic"] = 1
    df.loc[some(df.Fasting_Glucose >= 126.0), "Diabetic"] = 0
    df.loc[some(df.Systolic_BP > 140.0), "On_BP_Medication"] = 0
    df.loc[some(df.Biological_Sex == 1), "Is_Pregnant"] = 1
    df.loc[some(df.Is_Pregnant == 1), "Age"] = 70.0
    # R7: smokers pushed to zero cigarettes. Disjoint from the R1 slice above, which
    # touches non-smokers only.
    df.loc[some(df.Current_Smoker == 1), "Cigs_Per_Day"] = 0.0
    return df


def test_rule_flags_match_reference(violating_frame):
    """Identical per-row binding and violated flags for the six shared rules."""
    summaries, flags, total = evaluate_rules(violating_frame)
    their_flags = ref.check_rules(violating_frame)

    assert len(RULE_NAMES) == 6, RULE_NAMES     # an empty comparison set would pass vacuously

    for name in RULE_NAMES:
        assert np.array_equal(flags[f"{name}__binding"], their_flags[f"{name}__bound"].to_numpy()), name
        assert np.array_equal(flags[f"{name}__violated"], their_flags[f"{name}__violated"].to_numpy()), name

    # Every rule must actually be exercised, or the comparison proves nothing.
    for name in RULE_NAMES:
        assert flags[f"{name}__violated"].sum() > 0, f"{name} was never violated"


def test_canonical_only_rule_is_exercised(violating_frame):
    """R7 is exercised in canonical, and the reference has nothing to exercise.

    Held separately from the shared-rule comparison because there is no reference side
    to compare against -- but an unexercised rule proves as little here as anywhere else.
    """
    _, flags, _ = evaluate_rules(violating_frame)

    for name in CANONICAL_ONLY:
        assert name in dgp.RULES and name not in ref.RULES, name
        assert flags[f"{name}__binding"].sum() > 0, f"{name} never bound"
        assert flags[f"{name}__violated"].sum() > 0, f"{name} was never violated"
        assert f"{name}__bound" not in ref.check_rules(violating_frame).columns


def test_rule_summaries_match_reference(violating_frame):
    """binding_rate / violation_rate / violations agree, modulo the name/rule key."""
    ours = {s["name"]: s for s in evaluate_rules(violating_frame)[0]}
    theirs = {r["rule"]: r for r in ref.rule_report(violating_frame).to_dict("records")}

    # The contract, stated so it can still fail: exactly R7 on our side, nothing on theirs.
    assert set(ours) - set(theirs) == CANONICAL_ONLY
    assert set(theirs) - set(ours) == set()
    for name in RULE_NAMES:
        assert ours[name]["statement"] == theirs[name]["statement"]
        assert ours[name]["violations"] == theirs[name]["violations"]
        assert ours[name]["binding_rate"] == pytest.approx(theirs[name]["binding_rate"], abs=EXACT)
        assert ours[name]["violation_rate"] == pytest.approx(theirs[name]["violation_rate"], abs=EXACT)


# --------------------------------------------------------------------------------------
# Layer 1c -- assert_valid on shared hand-built manifests.
# --------------------------------------------------------------------------------------

def _shared_manifest(**overrides) -> dict:
    """Carries both key spellings so either implementation can read it unmodified."""
    rules = [
        {"name": "R1_smoker_cigs", "rule": "R1_smoker_cigs", "binding_rate": 0.803},
        {"name": "R6_preg_age", "rule": "R6_preg_age", "binding_rate": 0.0102},
    ]
    manifest = {
        "config": {"prevalence": 0.10, "bayes_auc_target": 0.85},
        "realised_prevalence": 0.1003,
        "expected_bayes_auc": 0.8500,
        "total_violations_on_real": 0,
        "rules": rules,
    }
    manifest.update(overrides)
    return manifest


@pytest.mark.parametrize("overrides, should_raise", [
    ({}, False),
    ({"total_violations_on_real": 1}, True),
    ({"realised_prevalence": 0.15}, True),
    ({"expected_bayes_auc": 0.90}, True),
    ({"total_violations_on_real": 3, "realised_prevalence": 0.15}, True),
])
def test_assert_valid_agrees_on_raise_or_pass(overrides, should_raise):
    """Same gate decision on the same manifest. Messages differ; the verdict must not."""
    manifest = _shared_manifest(**overrides)

    def verdict(fn):
        try:
            fn(manifest)
            return "pass"
        except AssertionError:
            return "raise"

    ours, theirs = verdict(dgp.assert_valid), verdict(ref.assert_valid)
    assert ours == theirs
    assert (ours == "raise") is should_raise


def test_dead_rule_threshold_diverges_intentionally():
    """The dead-rule lists differ, and that divergence is deliberate.

    We lowered the threshold to 0.005; the reference is at 0.01. R6_preg_age binds at
    ~0.0102, which straddles 0.01 -- its dead/alive status flips across seeds under the
    reference rule. Asserted here rather than allowed to surface as a failure.
    """
    assert dgp._DEAD_RULE_THRESHOLD == 0.005

    straddling = _shared_manifest()
    straddling["rules"] = [{"name": "R6_preg_age", "rule": "R6_preg_age", "binding_rate": 0.007}]

    assert ref.assert_valid(straddling) == ["R6_preg_age"]     # dead under 0.01
    assert dgp.assert_valid(straddling) == []                  # alive under 0.005

    # Below both thresholds the two agree again.
    clearly_dead = _shared_manifest()
    clearly_dead["rules"] = [{"name": "R6_preg_age", "rule": "R6_preg_age", "binding_rate": 0.001}]
    assert ref.assert_valid(clearly_dead) == dgp.assert_valid(clearly_dead) == ["R6_preg_age"]


# --------------------------------------------------------------------------------------
# Layer 1d -- known, intentional behavioural divergences.
# --------------------------------------------------------------------------------------

def test_reference_raises_where_ours_solves():
    """coupling=1.0 x prevalence=0.05: the reference's fixed alpha bracket cannot reach.

    The reference solves alpha with brentq over a fixed [-60, 60]. The alpha a low
    prevalence needs grows like gamma * max|z|, and at coupling=1.0 the trap score has a
    long right tail, so at gamma=30 the reachable mean is floored above 0.05 and brentq
    fails to bracket -- surfacing as a raw solver message rather than a clean diagnosis.
    Ours uses logit(p) +/- gamma*max|z| and solves. Measured floor:
    manifests/s0_coupling1_ceiling.json -> alpha_bracket_diagnostic.
    """
    kwargs = dict(n_samples=5000, coupling=1.0, prevalence=0.05, bayes_auc_target=0.85, seed=0)

    with pytest.raises(ValueError, match="must have different signs"):
        ref.generate(ref.DGPConfig(**kwargs))

    _, _, manifest = generate(DGPConfig(**kwargs))
    assert manifest["expected_bayes_auc"] == pytest.approx(0.85, abs=1e-6)
    dgp.assert_valid(manifest, tol_prev=0.02)


def test_solve_binding_rates_footgun_reproduced():
    """Both return a config carrying n_samples = n_probe. Deliberately not 'fixed'."""
    targets = {"R3_glucose_ceil": 0.08, "R4_bp_mandatory": 0.20}
    ours = dgp.solve_binding_rates(targets)
    theirs = ref.solve_binding_rates(targets)

    assert ours.n_samples == theirs.n_samples == 40_000

    # Independent samples of the same quantile, so they agree loosely, not exactly.
    # SE of the calibration constants is ~0.14 (glucose) and ~0.11 (sbp).
    assert ours.glucose_base == pytest.approx(theirs.glucose_base, abs=0.5)
    assert ours.sbp_base == pytest.approx(theirs.sbp_base, abs=0.5)
