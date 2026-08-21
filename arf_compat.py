"""ARF (adversarial random forests, Watson et al. AISTATS 2023) via arfpy.

arfpy 0.1.x predates numpy 2 / modern pandas. Two compat fixes:
  - np.in1d removed in numpy 2  -> alias to np.isin
  - arf.py line 329 indexed a name-indexed Series positionally -> patched to .iloc
Discrete columns MUST be passed as pandas 'category' dtype or ARF models them as
truncated normals and returns non-integer values for binary variables.
"""
import numpy as np, pandas as pd
if not hasattr(np, "in1d"): np.in1d = np.isin
from arfpy import arf  # noqa: E402

DISCRETE = ["Biological_Sex","Is_Pregnant","Current_Smoker","Diabetic",
            "On_BP_Medication","Heart_Disease_Risk"]

def fit_arf(df, n_out, seed=0, num_trees=30, min_node_size=5, verbose=False):
    np.random.seed(seed)
    d = df.copy()
    for c in DISCRETE:
        if c in d.columns: d[c] = d[c].astype(int).astype("category")
    m = arf.arf(x=d, num_trees=num_trees, min_node_size=min_node_size, verbose=verbose)
    m.forde()
    out = m.forge(n=n_out)
    out = out[df.columns]
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out.dropna().reset_index(drop=True)
