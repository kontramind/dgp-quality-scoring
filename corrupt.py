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
