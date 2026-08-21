"""
Two-score per-record metric derived from a GBDT's own leaf partition.
 
Axis 1 (support)  : how much real training mass occupies the same partition cells.
                    An epistemic proxy - low support means the model has little basis
                    for judgement at this location.
 
Axis 2 (residual) : how surprising this row's LABEL is, given the labels of real rows
                    in the same cells, NET of the local aleatoric noise. Zero when the
                    label is exactly as surprising as noise alone predicts.
 
Both come from one predict(pred_leaf=True) call. Cost is O(T) per row - no pairwise
distance matrix.
 
Kernel
------
k(x, x_i) = (1/T) sum_t 1[leaf_t(x) == leaf_t(x_i)]      (GBDT proximity, Breiman)
 
Support(x)  = mean_t log1p(n_{t, leaf_t(x)})
p_hat(x)    = sum_t pos_{t,leaf} / sum_t n_{t,leaf}      (kernel-weighted class rate)
surprisal   = -log p_hat(y | x)
residual    = surprisal - H(p_hat(x))                    (excess over local noise)
"""
 
import numpy as np
import lightgbm as lgb
 
 
class LeafGeometry:
    """Fit reference leaf statistics from real training data, then score any rows."""
 
    def __init__(self, model: lgb.LGBMClassifier, smoothing: float = 1.0):
        self.model = model
        self.smoothing = smoothing
        self._fitted = False
 
    def fit_reference(self, X_ref, y_ref):
        """Tabulate per-tree, per-leaf counts and positive-class sums on real data."""
        leaves = self.model.predict(X_ref, pred_leaf=True).astype(np.int32)
        y = np.asarray(y_ref, dtype=np.float64)
        n_rows, self.n_trees = leaves.shape
        self.max_leaf = int(leaves.max()) + 1
        self.prior = float(y.mean())
 
        self.counts = np.zeros((self.n_trees, self.max_leaf))
        self.possum = np.zeros((self.n_trees, self.max_leaf))
        for t in range(self.n_trees):
            col = leaves[:, t]
            self.counts[t] = np.bincount(col, minlength=self.max_leaf)
            self.possum[t] = np.bincount(col, weights=y, minlength=self.max_leaf)
        self.n_ref = n_rows
        self._fitted = True
        return self
 
    def _leaves(self, X):
        L = self.model.predict(X, pred_leaf=True).astype(np.int32)
        return np.clip(L, 0, self.max_leaf - 1)
 
    def score(self, X, y=None):
        """Returns dict of per-row scores. y is required for axis 2."""
        assert self._fitted, "call fit_reference first"
        L = self._leaves(X)
        tree_idx = np.arange(self.n_trees)[None, :]
 
        n_hit = self.counts[tree_idx, L]        # (n, T) real rows sharing each leaf
        p_hit = self.possum[tree_idx, L]
 
        support = np.log1p(n_hit).mean(axis=1)
        min_support = np.log1p(n_hit.min(axis=1))
        frac_empty = (n_hit == 0).mean(axis=1)
 
        tot_n = n_hit.sum(axis=1)
        tot_p = p_hit.sum(axis=1)
        a = self.smoothing
        p_hat = (tot_p + a * self.prior) / (tot_n + a)
        p_hat = np.clip(p_hat, 1e-9, 1 - 1e-9)
 
        # disagreement among trees about the local class rate: information the
        # aggregated prediction discards
        with np.errstate(invalid="ignore", divide="ignore"):
            per_tree = np.where(n_hit > 0, p_hit / np.maximum(n_hit, 1), np.nan)
        tree_var = np.nanvar(per_tree, axis=1)
 
        out = {"support": support, "min_support": min_support,
               "frac_empty_leaf": frac_empty, "p_hat": p_hat,
               "tree_rate_var": tree_var}
 
        if y is not None:
            y = np.asarray(y, dtype=np.float64)
            p_y = np.where(y == 1, p_hat, 1.0 - p_hat)
            surprisal = -np.log(p_y)
            H = -(p_hat * np.log(p_hat) + (1 - p_hat) * np.log(1 - p_hat))
            out["surprisal"] = surprisal
            out["entropy_local"] = H
            out["residual"] = surprisal - H          # aleatoric-corrected axis 2
        return out
 
