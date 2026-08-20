"""Exact recomputation of the DGP's true risk p*(x) for arbitrary feature vectors.

This is what makes "the label agrees with the boundary" ground truth rather than an
assumption: p* is rebuilt from the constants fixed at generation time (gamma, alpha,
coupling) and the reference moments of the frame it is constructed on, so any row --
real, corrupted or synthetic -- can be scored on the same scale the labels were drawn
against.

Three standardisations, matching ``dgp._weighted_score`` -> ``dgp.generate``:
per-variable z-scores, then per-composite, then on the blend. Dropping the third
silently rescales gamma and every excess number moves.
"""
import numpy as np
import pandas as pd
from scipy.special import expit

CORE_W = {"Age": 1.00, "BMI": 0.30, "Fasting_Glucose": 0.40,
          "Systolic_BP": 0.60, "Cigs_Per_Day": 0.50}
TRAP_W = {"Diabetic": 1.00, "On_BP_Medication": -0.80,
          "Is_Pregnant": 0.30, "Current_Smoker": 0.70}


def _mean(x) -> float:
    return float(np.asarray(x, dtype=float).mean())


def _sd(x) -> float:
    """Population sd (ddof=0), matching ``dgp._standardize``."""
    return float(np.asarray(x, dtype=float).std())


class TrueRisk:
    """p*(x) under the DGP, with moments frozen from ``X_ref``.

    ddof is 0 (numpy's default, and ``dgp._standardize``'s) at every stage. It cancels
    at the first two -- both composites carry the same sqrt((n-1)/n) factor and the next
    z-score divides it out -- but NOT at the third: the blend's z is not standardised
    again, it is multiplied by ``gamma``, which ``dgp._solve_gamma_alpha`` fitted against
    a ddof=0 z. Using ddof=1 there rescales z by sqrt((n-1)/n) and moves p* by ~1e-5.
    """

    def __init__(self, X_ref: pd.DataFrame, manifest: dict):
        self.gamma = float(manifest["solved_gamma"])
        self.alpha = float(manifest["solved_alpha"])
        self.lam = float(manifest["config"]["coupling"])

        self.stats = {v: (_mean(X_ref[v]), _sd(X_ref[v]))
                      for v in list(CORE_W) + list(TRAP_W)}

        core = self._comb(X_ref, CORE_W)
        trap = self._comb(X_ref, TRAP_W)
        self.core_ms = (_mean(core), _sd(core))
        self.trap_ms = (_mean(trap), _sd(trap))

        mix = self._mix(core, trap)
        self.mix_ms = (_mean(mix), _sd(mix))

    def _comb(self, X, W):
        m, s = self.stats, None
        for v, w in W.items():
            mu, sd = m[v]
            term = w * (X[v] - mu) / sd
            s = term if s is None else s + term
        return s

    def _mix(self, core, trap):
        zc = (core - self.core_ms[0]) / self.core_ms[1]
        zt = (trap - self.trap_ms[0]) / self.trap_ms[1]
        return (1.0 - self.lam) * zc + self.lam * zt

    def p(self, X: pd.DataFrame) -> np.ndarray:
        mix = self._mix(self._comb(X, CORE_W), self._comb(X, TRAP_W))
        z = (mix - self.mix_ms[0]) / self.mix_ms[1]
        return expit(self.alpha + self.gamma * np.asarray(z, dtype=float))


# --------------------------------------------------------------------------------------
# Corruption harness
# --------------------------------------------------------------------------------------
#
# One mutation per selected row, chosen uniformly among whichever of the four are
# eligible for that row. Each lands in a deterministic or structural branch of
# ``density.log_density_X`` -- `Diabetic|Glucose` low band, `Meds|SBP` above 140,
# `Is_Pregnant|Sex,Age` ineligible, `Cigs|Smoker` non-smoker atom -- so a corrupted row
# has exactly zero density, not merely a small one.
#
# The harness breaks R1, R2, R4 and R5. Not R3, not R6, and not R7: that is its existing
# scope, and widening it would change what the harness represents.
#
# Note on R1 eligibility: it is exact ``Cigs_Per_Day == 0``. Applied to a *synthetic*
# pool, a non-smoker sitting at 0.3 cigarettes is silently ineligible. That is a mild
# selection effect on which rows get corrupted, not a correctness bug; recorded here so
# it is a known property rather than an oversight.
#
# There is deliberately no ``VIOLATIONS`` table. The prior implementation carried one, as
# ``lambda d, m: d.loc[m, "Diabetic"].__setitem__(slice(None), 1)`` -- chained assignment
# onto a temporary, i.e. a silent no-op. Nothing referenced it, and ``apply_violations``
# mutated through its own inline mapping. It read like the specification while not being
# the code that ran. Do not reintroduce it from the old file.

# (rule broken, eligibility predicate, column mutated, new value). Declaration order is
# load-bearing: it fixes which index ``rng.integers(len(opts))`` selects.
_MUTATIONS = [
    ("R2_glucose_floor",
     lambda r: r["Fasting_Glucose"] < 100.0 and r["Diabetic"] == 0,
     "Diabetic", 1),
    ("R4_bp_mandatory",
     lambda r: r["Systolic_BP"] > 140.0 and r["On_BP_Medication"] == 1,
     "On_BP_Medication", 0),
    ("R5_anatomical",
     lambda r: r["Biological_Sex"] == 1 and r["Is_Pregnant"] == 0,
     "Is_Pregnant", 1),
    ("R1_smoker_cigs",
     lambda r: r["Current_Smoker"] == 0 and r["Cigs_Per_Day"] == 0,
     "Cigs_Per_Day", 12.0),
]


def apply_violations(X: pd.DataFrame, rows, rng) -> tuple[pd.DataFrame, np.ndarray]:
    """Break one rule per selected row. Returns ``(Xc, applied)``.

    ``rows`` are *positional* row indices into ``X``. ``applied`` is a length-``len(X)``
    object array naming the rule broken on each row, and ``""`` wherever no mutation was
    applied -- both for rows that were never selected and for selected rows where none of
    the four mutations was eligible. Every row with ``applied[r] == ""`` is therefore
    bit-identical to its input row.
    """
    Xc = X.copy()
    applied = np.full(len(Xc), "", dtype=object)

    for r in rows:
        r = int(r)
        row = Xc.iloc[r]
        opts = [m for m in _MUTATIONS if m[1](row)]
        if not opts:
            continue
        name, _elig, col, val = opts[int(rng.integers(len(opts)))]
        Xc.iat[r, Xc.columns.get_loc(col)] = val
        applied[r] = name

    return Xc, applied
