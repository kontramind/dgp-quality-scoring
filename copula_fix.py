"""Gaussian copula with Ruschendorf's distributional transform for ties.

Gaussianising average ranks breaks on binary columns: every row of a class shares one
rank, so Z collapses to two atoms with variance p(1-p)*delta^2 << 1. ``norm.cdf`` then
maps resampled Z into the middle of [0, 1] and the empirical-quantile inverse returns
the modal category almost always. Randomising within ties makes U exactly uniform
before gaussianising, which is what the distributional transform buys.

Three details are load-bearing:
  1. U is formed by standardising Z with the marginal sd before ``norm.cdf``, not by
     using a correlation matrix.
  2. The quantile inverse is a floor index into the sorted training marginal, not
     interpolation -- so every sampled value is literally a value that appeared in
     training.
  3. Discrete columns are ``np.rint``ed after that lookup.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

DISCRETE = ["Biological_Sex", "Is_Pregnant", "Current_Smoker", "Diabetic",
            "On_BP_Medication", "Heart_Disease_Risk"]


class GaussianCopulaDT:
    def fit(self, df: pd.DataFrame, rng=None) -> "GaussianCopulaDT":
        rng = rng or np.random.default_rng(0)
        self.cols = list(df.columns)
        n = len(df)
        self.marginals: dict[str, np.ndarray] = {}
        U = np.empty((n, len(self.cols)), dtype=float)

        for j, c in enumerate(self.cols):
            x = df[c].to_numpy(float)
            self.marginals[c] = np.sort(x)          # the full sorted column, kept
            hi = rankdata(x, method="max") / n
            lo = (rankdata(x, method="min") - 1) / n
            U[:, j] = lo + rng.random(n) * (hi - lo)

        U = np.clip(U, 1e-6, 1 - 1e-6)
        Z = norm.ppf(U)
        self.mu = Z.mean(axis=0)
        self.cov = np.cov(Z, rowvar=False) + 1e-6 * np.eye(len(self.cols))
        return self

    def sample(self, n: int, rng) -> pd.DataFrame:
        Z = rng.multivariate_normal(self.mu, self.cov, size=n)
        sd = np.sqrt(np.diag(self.cov))
        U = norm.cdf((Z - self.mu) / sd)

        out: dict[str, np.ndarray] = {}
        for j, c in enumerate(self.cols):
            m = self.marginals[c]
            v = m[np.clip((U[:, j] * len(m)).astype(int), 0, len(m) - 1)]
            if c in DISCRETE:
                v = np.rint(v)
            out[c] = v
        return pd.DataFrame(out)[self.cols]
