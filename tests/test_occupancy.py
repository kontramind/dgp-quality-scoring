"""Unit tests for the occupancy harness's estimator and masks.

The harness's numbers are only as good as ``excess_and_se``, and a silent error there
would corrupt every downstream figure while still producing a plausible table. These
tests pin the two things that can go wrong quietly: the closed-form standard error, and
whether the rule layer and the density layer agree about which rows are impossible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dgp
from density import log_density_X
from occupancy import RULE_NAMES, TOL_BAND, evaluate_frame, excess_and_se, occupancy_masks


def test_excess_is_zero_on_labels_drawn_from_p_star():
    """The null: y ~ Bernoulli(p*) carries no contradiction, so excess is 0 in expectation.

    Checked against the estimator's own se rather than a hand-picked tolerance -- if the
    se is wrong this test is wrong in the same direction, which is why the next test pins
    the se independently.
    """
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 200_000)
    y = (rng.random(p.size) < p).astype(int)

    excess, se, _abs_res, _floor = excess_and_se(p, y)
    assert abs(excess) < 4.0 * se


def test_se_matches_monte_carlo_spread():
    """sqrt(mean[p(1-p)(1-2p)^2] / m) is the sd of the excess across repeated draws."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.99, 4_000)

    draws = [excess_and_se(p, (rng.random(p.size) < p).astype(int))[0] for _ in range(400)]
    empirical = float(np.std(draws, ddof=1))
    closed_form = excess_and_se(p, (rng.random(p.size) < p).astype(int))[1]

    assert closed_form == pytest.approx(empirical, rel=0.10)


def test_excess_detects_injected_contradiction():
    """Flipping labels away from p* must raise excess well clear of its se."""
    rng = np.random.default_rng(2)
    p = rng.uniform(0.01, 0.99, 100_000)
    y = (rng.random(p.size) < p).astype(int)

    flip = rng.random(p.size) < 0.10
    y_bad = np.where(flip, 1 - y, y)

    clean_excess, clean_se, _, _ = excess_and_se(p, y)
    dirty_excess, dirty_se, _, _ = excess_and_se(p, y_bad)

    assert abs(clean_excess) < 4.0 * clean_se
    assert dirty_excess > clean_excess + 20.0 * dirty_se


def _frame(**overrides) -> pd.DataFrame:
    """One on-manifold row, overridable into any single rule's violation region."""
    row = {
        "Age": 40.0, "Biological_Sex": 0, "Current_Smoker": 0, "Is_Pregnant": 0,
        "Cigs_Per_Day": 0.0, "BMI": 25.0, "Fasting_Glucose": 110.0, "Diabetic": 0,
        "Systolic_BP": 120.0, "On_BP_Medication": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


# One violation per rule, each landing in a different -inf branch of log_density_X.
_VIOLATIONS = {
    "R1_smoker_cigs": {"Current_Smoker": 0, "Cigs_Per_Day": 5.0},
    "R2_glucose_floor": {"Fasting_Glucose": 90.0, "Diabetic": 1},
    "R3_glucose_ceil": {"Fasting_Glucose": 130.0, "Diabetic": 0},
    "R4_bp_mandatory": {"Systolic_BP": 150.0, "On_BP_Medication": 0},
    "R5_anatomical": {"Biological_Sex": 1, "Is_Pregnant": 1},
    "R6_preg_age": {"Biological_Sex": 0, "Is_Pregnant": 1, "Age": 70.0},
    "R7_smoker_cigs_min": {"Current_Smoker": 1, "Cigs_Per_Day": 0.0},
}


@pytest.mark.parametrize("rule", RULE_NAMES)
def test_every_rule_violation_has_zero_density(rule):
    """The harness asserts this as exact array equality; here is why it holds.

    Each rule's violation region is a deterministic or structural branch of
    density.log_density_X, so a violating row is not merely unlikely -- it is impossible,
    and its log-density is -inf. A rule whose violations had finite density would break
    the claim the harness's frac_density_zero_violating column exists to check.
    """
    X = _frame(**_VIOLATIONS[rule])
    violating, per_rule = occupancy_masks(X)

    assert per_rule[rule][0], f"{rule} was not violated by its own violation row"
    assert violating[0]
    assert not np.isfinite(log_density_X(X, dgp.DGPConfig()))[0]


def test_r1_tolerance_does_not_leak_onto_r7():
    """A smoker at 0.4 cigarettes violates R7 exactly; a non-smoker there does not violate R1.

    R1's tolerance patches exact equality to a point mass on a continuous column. R7's
    consequent is a half-line and takes no tolerance, and the two share a column -- so
    this is the place a tolerance would leak.
    """
    below = 0.4
    assert below < TOL_BAND

    _v, non_smoker = occupancy_masks(_frame(Current_Smoker=0, Cigs_Per_Day=below))
    assert not non_smoker["R1_smoker_cigs"][0]

    _v, smoker = occupancy_masks(_frame(Current_Smoker=1, Cigs_Per_Day=below))
    assert smoker["R7_smoker_cigs_min"][0]


def test_tol_band_rows_are_rule_clean_and_zero_density():
    """The gap the n_cigs_in_tol_band column exists to count.

    A non-smoker at 0.4 cigarettes is clean under every rule and yet impossible under the
    SCM, whose Cigs_Per_Day support is {0} u [1, 40]. Rules and density disagree here by
    construction, which is the whole reason both are measured.
    """
    X = _frame(Current_Smoker=0, Cigs_Per_Day=0.4)
    violating, _per_rule = occupancy_masks(X)

    assert not violating[0]
    assert not np.isfinite(log_density_X(X, dgp.DGPConfig()))[0]


def test_evaluate_frame_counts_the_tol_band():
    """0 and values at or above the tolerance are excluded; the open interval is counted."""
    cfg = dgp.DGPConfig()
    X = pd.concat(
        [_frame(Cigs_Per_Day=c) for c in (0.0, 0.1, 0.4, TOL_BAND, 1.0)], ignore_index=True
    )
    out = evaluate_frame(X, np.zeros(len(X), dtype=int), _IdentityRisk(), cfg)
    assert out["n_cigs_in_tol_band"] == 2


class _IdentityRisk:
    """Stands in for TrueRisk where the test is about masks, not about p*."""

    def p(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 0.1)
