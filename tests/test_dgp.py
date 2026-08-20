"""Tests for the DGP. Fixed seeds and tolerance bands only -- no property-based search,
no p-value assertions.

Two tolerance decisions differ from the first draft of the specification and are
justified in-line where they are used:

* R3/R4 binding rates are centred on their **targets** (0.080 / 0.200) with a +/-0.005
  band, not on the recorded realisation 0.202 with +/-0.002. See
  ``test_e0_golden_values`` for the standard-error derivation.
* The tight, deterministic part of that contract is tested separately and much more
  strictly by ``test_calibration_exact``.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from dgp import (
    DGPConfig,
    assert_valid,
    bayes_auc,
    bayes_auc_naive,
    generate,
    solve_binding_rates,
)

E0_TARGETS = {"R3_glucose_ceil": 0.08, "R4_bp_mandatory": 0.20}

# Binding rates that are not solved for by solve_binding_rates(). Every rule falls into
# exactly one of three classes, and the class decides where its band centre comes from:
#
#   derived   The antecedent is a config parameter read back, so the rate has a closed
#             form in DGPConfig. The centre is that closed form. R1 `1 - p_smoker`,
#             R5 `p_male`, R7 `p_smoker`, and R6 the pregnancy composite
#             `(1 - p_male) * (preg_age_max - age_lo)/(age_hi - age_lo) * p_pregnant`
#             -- Is_Pregnant is drawn only for women at or below preg_age_max, over a
#             uniform age range. On the defaults: 0.800, 0.500, 0.200, 0.0102040816.
#
#   solved    The rate is a target handed to solve_binding_rates(). The centre is the
#             declared target. R3 and R4, in SOLVED_BANDS below.
#
#   emergent  No closed form; the centre can only be measured. R2 alone. Checked rather
#             than assumed: it moves with several parameters at once and is a Gaussian
#             mixture over a clipped BMI over a uniform age. Sweeping one parameter at a
#             time moves it well outside any band -- p_male=0.3 -> 0.385,
#             sd_glucose=25 -> 0.425, b_bmi_glucose=1.4 -> 0.325 -- which is what a
#             genuinely emergent rate looks like and what R1/R5/R6/R7 do not do.
#
# **A derived or solved centre never comes from an observed draw.** Four of these centres
# were originally set from single realisations (R1 0.803, R5 0.497, R6 0.0100, R7 0.197)
# and were recentred on their derivations. That is the same single-draw failure mode this
# project has already paid for twice -- see decisions.md. The recentring is not cosmetic:
# canonical measures R5 at 0.502520, which spent 55% of the +/-0.01 against the old
# centre and spends 25% against the derived one.
#
# Widths are unchanged by the recentring and are still sampling-error arguments.
EMERGENT_BANDS = {
    "R1_smoker_cigs": (0.800, 0.01),        # derived: 1 - p_smoker
    "R2_glucose_floor": (0.379, 0.01),      # emergent: measured, no closed form
    "R5_anatomical": (0.500, 0.01),         # derived: p_male
    # R6 needs its own width. At p=0.01 the spec's blanket +/-0.01 admits [0, 0.02], so it
    # passes even when R6 never binds at all -- which is the dead-rule condition itself.
    # SE = sqrt(0.01*0.99/50000) = 0.00044, so +/-0.002 is ~4.5 SE.
    "R6_preg_age": (0.010204, 0.002),       # derived: the pregnancy composite above
    # R7 was appended to dgp.RULES after this table was first recorded. Adding it edited
    # no cell above: evaluate_rules consumes no RNG and runs after all sampling, so a
    # rule addition cannot perturb the sampler and every other golden value is
    # bit-identical (asserted by scripts/export_e0_r7_invariance.py).
    "R7_smoker_cigs_min": (0.200, 0.01),    # derived: p_smoker
}

# Solved binding rates, re-measured on an independent n=50000 draw.
#
# solve_binding_rates() calibrates at n_probe=40000 and E0 then regenerates at
# n_samples=50000, so this is an effectively independent sample of the same rate.
# SE = sqrt(p(1-p)/n) with n=50000  ->  0.00121 for R3 (p=.08), 0.00179 for R4 (p=.20).
# A +/-0.005 band is ~4 SE / ~3 SE. The spec's original 0.202 +/- 0.002 centred the band
# on one recorded realisation rather than on the target, leaving it off-centre by ~1.1 SE;
# the reference implementation itself falls outside it at seed 4.
SOLVED_BANDS = {"R3_glucose_ceil": (0.080, 0.005), "R4_bp_mandatory": (0.200, 0.005)}


def _rates(manifest) -> dict[str, float]:
    return {r["name"]: r["binding_rate"] for r in manifest["rules"]}


def _derived_centres(cfg: DGPConfig) -> dict[str, float]:
    """Closed forms for the four `derived` binding rates, straight off the config."""
    age_lo, age_hi = cfg.age_range
    return {
        "R1_smoker_cigs": 1.0 - cfg.p_smoker,
        "R5_anatomical": cfg.p_male,
        "R7_smoker_cigs_min": cfg.p_smoker,
        "R6_preg_age": ((1.0 - cfg.p_male)
                        * (cfg.preg_age_max - age_lo) / (age_hi - age_lo)
                        * cfg.p_pregnant),
    }


def test_derived_band_centres_match_their_closed_forms():
    """The four `derived` centres are the derivation, not a remembered draw.

    Without this the classification above is a comment, and a comment does not stop the
    next person from pasting in an observed rate. R2, R3 and R4 are deliberately absent:
    R2 has no closed form and R3/R4 take their centres from SOLVED_BANDS' targets.
    """
    derived = _derived_centres(DGPConfig())
    assert set(derived) | {"R2_glucose_floor"} == set(EMERGENT_BANDS)

    for rule, centre in derived.items():
        recorded = EMERGENT_BANDS[rule][0]
        # R6's closed form is 1/98, which does not terminate; the table carries it to the
        # six figures the other centres are exact at.
        assert recorded == pytest.approx(centre, abs=5e-7), (
            f"{rule}: band centre {recorded} != derived {centre}"
        )

    assert set(SOLVED_BANDS) == set(E0_TARGETS)
    for rule, target in E0_TARGETS.items():
        assert SOLVED_BANDS[rule][0] == target


@pytest.fixture(scope="module")
def e0_config() -> DGPConfig:
    """The E0 calibration, solved once at seed 0 and reused (it is a design constant)."""
    return solve_binding_rates(E0_TARGETS)


def _check_e0_manifest(manifest, cfg) -> None:
    rates = _rates(manifest)

    assert manifest["total_violations_on_real"] == 0

    # Centred on the configured target, not on the recorded 0.1003 realisation.
    assert manifest["realised_prevalence"] == pytest.approx(cfg.prevalence, abs=0.01)
    assert manifest["expected_bayes_auc"] == pytest.approx(cfg.bayes_auc_target, abs=0.01)

    for rule, (target, tol) in {**EMERGENT_BANDS, **SOLVED_BANDS}.items():
        assert rates[rule] == pytest.approx(target, abs=tol), (
            f"{rule}: {rates[rule]:.4f} outside {target} +/- {tol}"
        )


def test_e0_golden_values(e0_config):
    """The E0 reproduction recipe, seed 0."""
    cfg = e0_config

    # Loose sanity checks: these are seed-0 calibration constants with ~0.11-0.14 of
    # calibration noise, not exact quantities.
    assert cfg.glucose_base == pytest.approx(102.2, abs=0.5)
    assert cfg.sbp_base == pytest.approx(109.3, abs=0.5)

    # The documented footgun: solve_binding_rates() hands back n_samples = n_probe.
    assert cfg.n_samples == 40_000
    cfg.n_samples = 50_000

    df, godview, manifest = generate(cfg)

    print("\n--- E0 golden values (seed 0) ---")
    print(f"  glucose_base        = {cfg.glucose_base:.4f}   (~102.2)")
    print(f"  sbp_base            = {cfg.sbp_base:.4f}   (~109.3)")
    print(f"  solved_gamma        = {manifest['solved_gamma']:.6f}")
    print(f"  solved_alpha        = {manifest['solved_alpha']:.6f}")
    print(f"  realised_prevalence = {manifest['realised_prevalence']:.6f}   (0.10 +/- 0.01)")
    print(f"  expected_bayes_auc  = {manifest['expected_bayes_auc']:.6f}   (0.8500 +/- 0.01)")
    print(f"  empirical_bayes_auc = {manifest['empirical_bayes_auc']:.6f}")
    print(f"  total_violations    = {manifest['total_violations_on_real']}   (exactly 0)")
    for name, rate in _rates(manifest).items():
        target, tol = {**EMERGENT_BANDS, **SOLVED_BANDS}[name]
        print(f"  {name:<18} = {rate:.4f}   ({target} +/- {tol})")

    assert df.shape == (50_000, 11)
    assert len(godview) == 50_000
    _check_e0_manifest(manifest, cfg)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_e0_golden_values_stability(e0_config, seed):
    """Same targets and tolerances across seeds -- stable within tolerance, not bit-identical.

    The calibration constants are solved once at seed 0 and reused; they are part of the
    DGP's design, not something re-derived per draw. Only the generation seed varies.
    """
    cfg = DGPConfig(**{**vars(e0_config), "n_samples": 50_000, "seed": seed})
    _, _, manifest = generate(cfg)

    rates = _rates(manifest)
    print(
        f"\n  seed={seed}  prev={manifest['realised_prevalence']:.4f} "
        f"auc={manifest['expected_bayes_auc']:.4f}  "
        + "  ".join(f"{n.split('_')[0]}={rates[n]:.4f}" for n in sorted(rates))
    )
    _check_e0_manifest(manifest, cfg)


@pytest.mark.parametrize("targets", [
    {"R3_glucose_ceil": 0.08, "R4_bp_mandatory": 0.20},
    {"R3_glucose_ceil": 0.05, "R4_bp_mandatory": 0.30},
])
def test_calibration_exact(targets):
    """The solver contract: at the calibration sample itself the fit is exact.

    Generates on the returned config *without touching n_samples*, so this runs at
    n_probe=40000 -- the very sample the root-find optimised. Any drift here means the
    root-find is broken, so the band is 1e-4 rather than the loose bands used for a
    regenerated production draw.
    """
    cfg = solve_binding_rates(targets)
    assert cfg.n_samples == 40_000

    _, _, manifest = generate(cfg)
    rates = _rates(manifest)
    for rule, target in targets.items():
        assert rates[rule] == pytest.approx(target, abs=1e-4), (
            f"{rule}: {rates[rule]:.6f} != {target} at the calibration sample"
        )


def test_unreachable_auc_raises():
    """An unreachable AUC target raises a clear error rather than returning garbage.

    At coupling=1.0 the score is a function of four binary variables, so it takes only
    16 distinct values and tied pairs cap the achievable AUC.

    This is a *solver window* limit, not a hard structural one: the exact gamma -> infinity
    limit at prevalence 0.30 is above 0.95, but most of the mass sits in a single z level,
    so the approach to that limit is far too slow to be usable. Refusing is the right
    behaviour, and the error reports the max achievable and the gamma it was measured at.
    Measured ceilings live in results/s0_coupling1_ceiling.csv (with the n and seed they
    were taken at, since the level masses are seed-dependent) -- this docstring deliberately
    quotes none of them.

    (0.999 -- the spec's original example -- is *not* unreachable: at the default config it
    solves cleanly well under the solver's gamma ceiling.)
    """
    with pytest.raises(ValueError, match=r"unreachable for this score.*max achievable") as excinfo:
        generate(DGPConfig(n_samples=5000, coupling=1.0, prevalence=0.30,
                           bayes_auc_target=0.95, seed=0))

    # Assert the reported ceiling rather than narrating it: it must be a real number
    # below the requested target, which is the whole justification for refusing.
    reported = float(re.search(r"max achievable ([0-9.]+)", str(excinfo.value)).group(1))
    assert 0.5 < reported < 0.95


def test_alpha_failure_is_translated(monkeypatch):
    """A bracketing failure in the alpha solve must never escape as a raw brentq message.

    The alpha bracket is analytically guaranteed (logit(p) +/- gamma*max|z|), so this path
    is unreachable through the public API. It is forced here rather than left as an
    untested branch: the gamma-scaled bracket is the real fix, and the only thing left
    around it is this one error translation, which is verified rather than assumed.
    """
    import dgp as dgp_module

    real_solve_alpha = dgp_module._solve_alpha

    def failing(z, gamma, prevalence):
        if gamma > 1.0:
            raise dgp_module._AlphaSolveError("forced bracketing failure")
        return real_solve_alpha(z, gamma, prevalence)

    monkeypatch.setattr(dgp_module, "_solve_alpha", failing)
    with pytest.raises(ValueError, match="not attainable for this score"):
        generate(DGPConfig(n_samples=2000, seed=0))


def _clean_manifest() -> dict:
    return {
        "config": {"prevalence": 0.10, "bayes_auc_target": 0.85},
        "realised_prevalence": 0.1003,
        "expected_bayes_auc": 0.8500,
        "total_violations_on_real": 0,
        "rules": [
            {"name": "R1_smoker_cigs", "binding_rate": 0.803},
            {"name": "R6_preg_age", "binding_rate": 0.003},   # below _DEAD_RULE_THRESHOLD
        ],
    }


def test_assert_valid_gates():
    clean = _clean_manifest()
    assert assert_valid(clean) == ["R6_preg_age"]     # dead rule reported, never fatal

    violating = {**clean, "total_violations_on_real": 1}
    with pytest.raises(AssertionError, match="total_violations_on_real"):
        assert_valid(violating)

    off_prevalence = {**clean, "realised_prevalence": 0.15}
    with pytest.raises(AssertionError, match="realised_prevalence"):
        assert_valid(off_prevalence)

    # All failures are collected into a single AssertionError.
    both = {**clean, "total_violations_on_real": 3, "realised_prevalence": 0.15}
    with pytest.raises(AssertionError) as excinfo:
        assert_valid(both)
    assert "total_violations_on_real" in str(excinfo.value)
    assert "realised_prevalence" in str(excinfo.value)


# The single grid cell whose AUC target exceeds what is achievable within the solver's
# gamma ceiling. See test_unreachable_auc_raises for why, and
# results/s0_coupling1_ceiling.csv for the measured ceilings.
_UNREACHABLE_CELL = (1.0, 0.95, 0.30)


@pytest.mark.parametrize("n_nuisance", [0, 20])
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("prevalence", [0.05, 0.10, 0.30])
@pytest.mark.parametrize("bayes_auc_target", [0.65, 0.85, 0.95])
@pytest.mark.parametrize("coupling", [0.0, 0.5, 1.0])
def test_property_config_space(coupling, bayes_auc_target, prevalence, seed, n_nuisance):
    """assert_valid must pass across the configuration grid.

    n_samples=5000 (not 50000) keeps 432 parametrisations fast; tol_prev is widened to
    0.02 to match the larger standard error of the smaller sample. That is a deliberate
    speed/precision tradeoff, not a workaround for a failing assertion.
    """
    cfg = DGPConfig(
        n_samples=5000, coupling=coupling, bayes_auc_target=bayes_auc_target,
        prevalence=prevalence, seed=seed, n_nuisance=n_nuisance,
    )

    if (coupling, bayes_auc_target, prevalence) == _UNREACHABLE_CELL:
        with pytest.raises(ValueError, match="unreachable for this score"):
            generate(cfg)
        return

    _, _, manifest = generate(cfg)
    if coupling == 1.0:
        print(f"\n  coupling=1.0 prev={prevalence} target={bayes_auc_target} seed={seed} "
              f"nuis={n_nuisance}: expected_bayes_auc={manifest['expected_bayes_auc']:.6f} "
              f"gamma={manifest['solved_gamma']:.3f}")
    assert_valid(manifest, tol_prev=0.02)


def test_knob_orthogonality():
    """Coupling moves the trap variance share 0 -> 1 without disturbing prevalence or AUC."""
    shares = []
    for coupling in [0.0, 0.25, 0.5, 0.75, 1.0]:
        _, _, manifest = generate(
            DGPConfig(n_samples=20_000, coupling=coupling, seed=0)
        )
        shares.append(manifest["trap_variance_share"])
        assert manifest["realised_prevalence"] == pytest.approx(0.10, abs=0.01)
        assert manifest["expected_bayes_auc"] == pytest.approx(0.85, abs=0.01)

    assert shares[0] == 0.0
    assert shares[-1] == pytest.approx(1.0, abs=1e-9)
    assert all(b > a for a, b in zip(shares, shares[1:])), shares


def test_lambda_invariance():
    """Coupling enters only after every X column is drawn, so features are bit-identical.

    n_nuisance=0: nuisance columns are built from z, so a non-zero nuisance_corr would
    legitimately break this.
    """
    frames = [
        generate(DGPConfig(n_samples=5000, coupling=lam, n_nuisance=0, seed=0))[0]
        for lam in (0.0, 0.5, 1.0)
    ]
    base, *rest = frames
    features = [c for c in base.columns if c != "Heart_Disease_Risk"]

    for other in rest:
        assert list(other.columns) == list(base.columns)
        for col in features:
            assert np.array_equal(base[col].to_numpy(), other[col].to_numpy()), col

    labels = [f["Heart_Disease_Risk"].to_numpy() for f in frames]
    assert not np.array_equal(labels[0], labels[1])


def test_bayes_auc_closed_form_vs_naive():
    rng = np.random.default_rng(0)

    # Continuous case: no ties, exercises the ordering and normalisation.
    p = rng.uniform(0.001, 0.999, 500)
    assert bayes_auc(p) == pytest.approx(bayes_auc_naive(p), abs=1e-9)

    # Heavy-ties case: this is what actually pins the half-credit convention. A strict
    # 1[p_i > p_j] implementation drops tied pairs entirely and disagrees here.
    tied = rng.choice(np.array([0.05, 0.2, 0.5, 0.8, 0.95]), size=200)
    assert bayes_auc(tied) == pytest.approx(bayes_auc_naive(tied), abs=1e-9)
    assert len(np.unique(tied)) == 5


def test_determinism():
    cfg = DGPConfig(n_samples=5000, n_nuisance=3, nuisance_corr=0.3, seed=7)
    df_a, gv_a, man_a = generate(cfg)
    df_b, gv_b, man_b = generate(cfg)

    assert df_a.equals(df_b)
    assert gv_a.equals(gv_b)
    assert man_a["solved_gamma"] == man_b["solved_gamma"]
    assert man_a["realised_prevalence"] == man_b["realised_prevalence"]
