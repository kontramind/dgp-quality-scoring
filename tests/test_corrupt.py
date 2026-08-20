"""Acceptance tests for ``corrupt.apply_violations``.

The natural single gate is "every corrupted row has zero density", and it is true: each
of the four mutations lands in a deterministic or structural branch of
``density.log_density_X``. But it passes vacuously on a harness that mutates nothing,
that mutates rows which were already violating, or that corrupts rows it was not asked
to. So the gate is five checks, not one:

1. every row the harness claims to have corrupted has non-finite log-density -- as exact
   boolean-mask equality, not as a fraction;
2. every row it claims not to have touched is bit-identical to its input row;
3. exactly one column differs per corrupted row, and it is the column mapped to the rule
   named in ``applied``;
4. each corrupted row violates that named rule when checked through
   ``dgp.evaluate_rules`` -- this is what ties the harness to the rule layer rather than
   to density alone;
5. all four mutations fire at least once, on a frame built to make each eligible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dgp
from corrupt import _MUTATIONS, apply_violations
from density import log_density_X
from dgp import DGPConfig, generate, solve_binding_rates

E0_TARGETS = {"R3_glucose_ceil": 0.08, "R4_bp_mandatory": 0.20}

# rule -> the single column that rule's mutation is allowed to touch.
_COLUMN_OF = {name: col for name, _elig, col, _val in _MUTATIONS}


@pytest.fixture(scope="module")
def corrupted():
    """E0 data with a large random slice of rows put through the harness.

    n is large enough, and the selected slice wide enough, that all four mutations are
    eligible somewhere -- check 5 asserts that rather than assuming it.
    """
    cfg = solve_binding_rates(E0_TARGETS)
    cfg = DGPConfig(**{**vars(cfg), "n_samples": 20_000, "seed": 0})
    df, _, _ = generate(cfg)
    X = df.drop(columns=[dgp.LABEL]).reset_index(drop=True)

    rng = np.random.default_rng(0)
    rows = rng.choice(len(X), size=4_000, replace=False)
    Xc, applied = apply_violations(X, rows, rng)
    return cfg, X, Xc, applied


def test_check_1_corrupted_rows_have_zero_density(corrupted):
    """Exact mask equality: corrupted <-> non-finite log-density, row for row."""
    cfg, X, Xc, applied = corrupted

    assert np.isfinite(log_density_X(X, cfg)).all(), "input frame was not on-manifold"

    is_corrupted = applied != ""
    nonfinite = ~np.isfinite(log_density_X(Xc, cfg))

    print(f"\ncheck_1: n_corrupted={int(is_corrupted.sum())} "
          f"n_nonfinite={int(nonfinite.sum())}")
    assert np.array_equal(is_corrupted, nonfinite)


def test_check_2_untouched_rows_are_bit_identical(corrupted):
    """Rows the harness did not claim must be unchanged, to the bit."""
    _cfg, X, Xc, applied = corrupted
    untouched = applied == ""
    assert untouched.sum() > 0
    assert X.loc[untouched].equals(Xc.loc[untouched])


def test_check_3_exactly_one_mapped_column_differs(corrupted):
    """One column per corrupted row, and it is the one the named rule maps to."""
    _cfg, X, Xc, applied = corrupted
    differs = X.to_numpy() != Xc.to_numpy()
    n_diff = differs.sum(axis=1)
    is_corrupted = applied != ""

    assert np.array_equal(n_diff > 0, is_corrupted)
    assert (n_diff[is_corrupted] == 1).all()

    cols = list(X.columns)
    for r in np.flatnonzero(is_corrupted):
        changed = cols[int(np.flatnonzero(differs[r])[0])]
        assert changed == _COLUMN_OF[applied[r]], (r, applied[r], changed)


def test_check_4_each_corrupted_row_violates_its_named_rule(corrupted):
    """Checked through evaluate_rules, so the harness is tied to the rule layer."""
    _cfg, _X, Xc, applied = corrupted
    _summaries, flags, _total = dgp.evaluate_rules(Xc)

    for r in np.flatnonzero(applied != ""):
        name = applied[r]
        assert flags[f"{name}__violated"][r], (r, name)


def test_check_5_all_four_mutations_fire(corrupted):
    """A harness that only ever reaches one branch would pass checks 1-4."""
    _cfg, _X, _Xc, applied = corrupted
    fired = set(applied[applied != ""])
    assert fired == {name for name, _e, _c, _v in _MUTATIONS}, fired


def test_ineligible_rows_are_left_alone():
    """A row eligible for nothing is selected, reported as "", and not mutated.

    Without this, "" could mean "never selected" and the check-2 guarantee would only
    ever be exercised on rows the harness never looked at.
    """
    row = {
        "Age": 40.0, "Biological_Sex": 0, "Current_Smoker": 1, "Is_Pregnant": 0,
        "Cigs_Per_Day": 15.0, "BMI": 25.0, "Fasting_Glucose": 110.0, "Diabetic": 0,
        "Systolic_BP": 120.0, "On_BP_Medication": 0,
    }
    X = pd.DataFrame([row])
    Xc, applied = apply_violations(X, [0], np.random.default_rng(0))

    assert applied[0] == ""
    assert X.equals(Xc)
