"""Applies the arfpy compat patches that arf_compat.py only documents.
Import this BEFORE arf_compat. Idempotent."""
import os, numpy as np, arfpy
if not hasattr(np, "in1d"): np.in1d = np.isin
_p = os.path.join(os.path.dirname(arfpy.__file__), "arf.py")
_s = open(_p).read()
if "self.factor_cols[j]" in _s:
    open(_p, "w").write(_s.replace("self.factor_cols[j]", "self.factor_cols.iloc[j]"))
