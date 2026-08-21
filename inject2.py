"""C1 injection, rebuilt.
 
Every arm delivers the SAME number of rows at the SAME class prevalence and (for the
four corrupted arms) the SAME feature distribution. The only thing that varies is
WHICH rows get the positive label.
 
  clean          uncorrupted rows, generator's own labels        (volume control)
  balance        uncorrupted rows, positives placed at random    (label-noise control)
  random         corrupted rows,   positives placed at random    (off-manifold + noise)
  creative       corrupted rows,   positives placed ~ p*(x)      (off-manifold, coherent)
  harmful_rank   corrupted rows,   positives placed at LOWEST p* (off-manifold, contradicts)
  harmful_orig   corrupted rows,   y = 1[p* < 0.5], NO matching  (reproduces E5 exactly)
 
random -> creative isolates "label carries boundary information".
random -> harmful_rank isolates "label actively contradicts the boundary".
balance -> random isolates the off-manifold features themselves.
harmful_orig is kept only so the old -0.21 can be shown to be the class-balance shift.
"""
import numpy as np, pandas as pd
from corrupt import TrueRisk, apply_violations
 
 
def _corrupt_pool(X_pool, X_ref, manifest, n_want, rng, oversample=3):
    tr = TrueRisk(X_ref, manifest)
    idx = rng.choice(len(X_pool), size=min(n_want*oversample, len(X_pool)), replace=False)
    Xs = X_pool.iloc[idx].reset_index(drop=True)
    Xc, applied = apply_violations(Xs, np.arange(len(Xs)), rng)
    hit = np.array([a != "" for a in applied])
    Xc = Xc[hit].reset_index(drop=True)
    if len(Xc) > n_want:
        Xc = Xc.iloc[:n_want].reset_index(drop=True)
    return Xc, tr.p(Xc)
 
 
def _place(n, n_pos, rng, weights=None, lowest=None):
    """Return a 0/1 vector of length n with exactly n_pos ones."""
    y = np.zeros(n, int)
    if n_pos <= 0:
        return y
    if lowest is not None:                      # positives at the smallest p*
        y[np.argsort(lowest)[:n_pos]] = 1
    elif weights is not None:                   # positives sampled with prob ~ p*
        w = np.clip(weights, 1e-9, None); w = w / w.sum()
        y[rng.choice(n, size=min(n_pos, n), replace=False, p=w)] = 1
    else:                                       # positives uniformly at random
        y[rng.choice(n, size=min(n_pos, n), replace=False)] = 1
    return y
 
 
def make_arm(kind, X_pool, y_pool, X_ref, manifest, n_want, target_prev, rng):
    """Returns (X, y, diagnostics dict). Diagnostics are mandatory reporting."""
    n_pos = int(round(target_prev * n_want))
    if kind in ("clean", "balance"):
        idx = rng.choice(len(X_pool), size=min(n_want, len(X_pool)), replace=False)
        Xc = X_pool.iloc[idx].reset_index(drop=True)
        tr = TrueRisk(X_ref, manifest); p = tr.p(Xc)
        if kind == "clean":
            y = np.asarray(y_pool)[idx].copy()
        else:
            y = _place(len(Xc), n_pos, rng)
    else:
        Xc, p = _corrupt_pool(X_pool, X_ref, manifest, n_want, rng)
        if len(Xc) == 0:
            return Xc, np.array([]), {}
        n_pos = int(round(target_prev * len(Xc)))
        if kind == "random":        y = _place(len(Xc), n_pos, rng)
        elif kind == "creative":    y = _place(len(Xc), n_pos, rng, weights=p)
        elif kind == "harmful_rank":y = _place(len(Xc), n_pos, rng, lowest=p)
        elif kind == "harmful_orig":y = (p < 0.5).astype(int)      # E5's definition
        else: raise ValueError(kind)
    d = {"delivered": len(Xc), "requested": n_want, "inj_prev": float(np.mean(y)),
         "mean_pstar": float(np.mean(p)), "mean_abs_y_minus_pstar": float(np.mean(np.abs(y - p)))}
    return Xc, y, d
 
