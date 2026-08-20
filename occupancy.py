"""Generator occupancy and contradiction harness.

Measures, for a fitted generator: how much of its output lands on the DGP's manifold,
whether the rule layer and the density layer agree about that, and where the label
contradiction it produces lives.

Why a harness and not a script
------------------------------
``occ_cop_r7.py`` measured this for one generator and is stranded: it derives its
"R1-R6" row from ``rules_tol.check_rules_tol``, which reads ``dgp.RULES`` with a late
lookup, so once R7 joined ``dgp.RULES`` that row silently became an R1-R7 row and the
two rows it contrasts collapsed. Its committed results stand at their own SHA and it
must not be re-run in place. The R1-R7 measurement it made still needs re-establishing
with provenance at HEAD, and the same measurement on further generators is what the
A2/A3/A4 rebuild consumes -- hence a harness whose generator is a parameter.

Design is taken from ``occ_cop_r7.py`` at ``eb5c040`` and reproduced, not reinvented:
n=40000 real cohort, a 0.6/0.4 stratified split at ``random_state=0``, ``TrueRisk``
frozen on the *train* half, the generator fitted on the train half with its label column
and asked for ``len(train)`` rows, and a 5 DGP-seed x 5 generator-seed grid whose
generator seed does not depend on the DGP seed. Aggregation matches too: mean over
generator seeds within a DGP seed first, so one DGP seed is one unit, then across DGP
seeds.

Scope
-----
Occupancy and contradiction only. No detection metrics, no recipient models, no
injection arms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

import dgp
from copula_fix import GaussianCopulaDT
from corrupt import TrueRisk
from density import log_density_X
from rules_tol import check_rules_tol

__all__ = [
    "ARMS",
    "Arm",
    "CopulaDTArm",
    "RULE_NAMES",
    "TOL_BAND",
    "evaluate_frame",
    "excess_and_se",
    "occupancy_masks",
]

RULE_NAMES = list(dgp.RULES)

# rules_tol's R1 tolerance: a non-smoker below this many cigarettes a day is not counted
# as violating R1. The open interval (0, TOL_BAND) is therefore rule-clean and yet has
# zero density under the SCM -- real Cigs_Per_Day has support {0} u [1, 40]. Counting
# rows that land there is how the rule layer and the density layer are held apart.
TOL_BAND = 0.5


# --------------------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------------------

class Arm(Protocol):
    """A generator, reduced to the only thing the harness needs from one.

    Everything downstream of ``fit_sample`` -- rule evaluation, density, p*, excess -- is
    arm-agnostic and must stay that way. Adding a generator is adding an adapter here and
    a line in ARMS; if a new arm ever needs a branch further down, the harness is wrong
    and that is worth raising before it is built in.
    """

    name: str

    def fit_sample(self, train: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
        """Fit on ``train`` (features **and** label) and return ``n`` sampled rows.

        The returned frame carries the same columns as ``train``, label included.
        """


@dataclass
class CopulaDTArm:
    """``copula_fix.GaussianCopulaDT`` -- Gaussian copula with the distributional transform.

    One RNG for both fit and sample, matching ``occ_cop_r7.py``: the fit consumes draws
    for the within-tie randomisation, so sharing the stream is part of what the recorded
    seed means.
    """

    name: str = "copula_dt"

    def fit_sample(self, train: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return GaussianCopulaDT().fit(train, rng).sample(n, rng)


ARMS: dict[str, Callable[[], Arm]] = {
    "copula_dt": CopulaDTArm,
}


# --------------------------------------------------------------------------------------
# Occupancy
# --------------------------------------------------------------------------------------

def occupancy_masks(X: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """``(violating, per_rule_violated)`` under R1-R7 via ``rules_tol.check_rules_tol``.

    ``check_rules_tol`` iterates ``dgp.RULES`` with a late lookup, so R7 is picked up
    automatically and with its exact ``>= 1.0`` consequent through the ``else`` branch --
    the R1 tolerance is R1's alone and does not leak onto it.
    """
    flags = check_rules_tol(X)
    per_rule = {n: flags[f"{n}__violated"].to_numpy() for n in RULE_NAMES}
    violating = np.zeros(len(X), dtype=bool)
    for v in per_rule.values():
        violating |= v
    return violating, per_rule


# --------------------------------------------------------------------------------------
# Contradiction
# --------------------------------------------------------------------------------------

def excess_and_se(p: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """``(excess, se, abs_res, bayes_floor)`` for one subset.

    ``excess = mean|y - p*| - 2 E[p*(1 - p*)]``. The subtrahend is the Bayes floor: for
    ``y ~ Bernoulli(p*)``, ``E|y - p*| = 2 p*(1 - p*)`` exactly, so excess is zero in
    expectation on data whose labels were drawn from ``p*``.

    ``se`` is closed-form from *this subset's own* ``p*`` values, not a constant.
    ``|y - p*|`` takes ``1 - p*`` with probability ``p*`` and ``p*`` otherwise, so its
    variance is ``p*(1 - p*)(1 - 2p*)^2``; the standard error of the mean is the root of
    that averaged over the subset, divided by the subset size. The ``0.173/sqrt(m)``
    shortcut in the S1 record is this quantity evaluated on real data's p* distribution
    under the E0 config, and must not be substituted for the formula on a subset whose
    p* distribution differs -- which every off-manifold subset's does.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    m = p.size
    abs_res = float(np.abs(y - p).mean())
    floor = float(np.mean(2.0 * p * (1.0 - p)))
    var = p * (1.0 - p) * (1.0 - 2.0 * p) ** 2
    se = float(np.sqrt(float(var.mean()) / m)) if m else float("nan")
    return abs_res - floor, se, abs_res, floor


# NOTE -- excess must never be computed inside a label-defined subset, and this is the
# only place that could be done by accident.
#
# The estimator compares realised |y - p*| against the floor 2 E[p*(1-p*)], which is the
# expectation of |y - p*| *given p* and given that y was drawn from it*. Conditioning on y
# breaks the second half: within `y == 1`, E[y | p*] is 1 rather than p*, so the residual
# no longer has mean 2p*(1-p*) and the difference stops measuring contradiction and starts
# measuring the conditioning. On data carrying zero injected contradiction it reads
# +0.37394 on the `y == 1` stratum and -0.04087 on `y == 0` -- both large, both artefacts.
#
# The on/off-manifold split is safe because it is *feature*-defined: every rule in
# dgp.RULES is a function of X alone and none reads the label. Keep it that way. Do not
# add a per-label breakdown to any excess column.
_LABEL_DEFINED_SUBSETS_ARE_FORBIDDEN = True


# --------------------------------------------------------------------------------------
# One frame, fully evaluated
# --------------------------------------------------------------------------------------

_MIN_SUBSET = 50        # occ_cop_r7.py's guard: below this a subset is not described


def evaluate_frame(
    X: pd.DataFrame, y: np.ndarray, true_risk: TrueRisk, cfg: dgp.DGPConfig
) -> dict[str, Any]:
    """Every quantity this harness reports, for one frame of rows.

    ``X`` carries features only; ``y`` is the frame's label column. ``true_risk`` is
    frozen on the run's reference frame and is passed in rather than rebuilt -- see
    the runner.
    """
    p = true_risk.p(X)
    violating, per_rule = occupancy_masks(X)
    clean = ~violating
    n = len(X)

    logp = log_density_X(X, cfg)
    zero_density = ~np.isfinite(logp)

    # Exact array equality, not a rounded fraction: the claim is that every rule-violating
    # row is structurally impossible, so its density is -inf. Each rule's violation region
    # lands in a deterministic or structural branch of density.log_density_X. A single
    # violating row with finite density falsifies that, and a fraction rounded to four
    # places would hide it.
    violating_all_zero = bool(np.array_equal(violating, violating & zero_density))

    cigs = X["Cigs_Per_Day"].to_numpy(float)
    out: dict[str, Any] = {
        "n": n,
        "on_manifold_row_share": float(clean.mean()),
        "n_on_manifold": int(clean.sum()),
        "n_off_manifold": int(violating.sum()),
        "frac_density_zero_ruleclean": (
            float(zero_density[clean].mean()) if clean.any() else float("nan")),
        "frac_density_zero_violating": (
            float(zero_density[violating].mean()) if violating.any() else float("nan")),
        "density_zero_violating_exact": violating_all_zero,
        # Rule-clean under the R1 tolerance and yet off the support: real Cigs_Per_Day is
        # {0} u [1, 40], so (0, TOL_BAND) has zero density for every row.
        "n_cigs_in_tol_band": int(((cigs > 0.0) & (cigs < TOL_BAND)).sum()),
        "mean_pstar": float(p.mean()),
        "prev": float(np.asarray(y, dtype=float).mean()),
    }
    for name in RULE_NAMES:
        out[f"{name}__violation_rate"] = float(per_rule[name].mean())

    for tag, mask in (("all", np.ones(n, dtype=bool)),
                      ("on_manifold", clean),
                      ("off_manifold", violating)):
        if int(mask.sum()) < _MIN_SUBSET:
            out[f"excess_{tag}"] = float("nan")
            out[f"se_{tag}"] = float("nan")
            out[f"mass_{tag}"] = float(mask.mean())
            continue
        excess, se, _abs_res, _floor = excess_and_se(p[mask], np.asarray(y)[mask])
        out[f"excess_{tag}"] = excess
        out[f"se_{tag}"] = se
        out[f"mass_{tag}"] = float(mask.mean())

    # Share of *excess contradiction* carried by on-manifold rows, weighted by subset
    # mass. Distinct from on_manifold_row_share, which is a row count. Reported per row
    # here; the run-level figure is formed from the per-DGP-seed means of mass and excess,
    # matching occ_cop_r7.py, and not by averaging this column.
    on_em = out["mass_on_manifold"] * out["excess_on_manifold"]
    off_em = out["mass_off_manifold"] * out["excess_off_manifold"]
    total = on_em + off_em
    out["excess_mass_on_manifold"] = on_em
    out["excess_mass_off_manifold"] = off_em
    out["on_manifold_excess_share"] = float(on_em / total) if total else float("nan")
    return out
