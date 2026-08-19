"""Rule checking with a tolerance on R1's exact-equality constraint.

R1 is ``Current_Smoker == 0 -> Cigs_Per_Day == 0``. Exact equality on a continuous
column survives no generator that does not reproduce point masses exactly, so under a
change of basis it reads as near-100% violated and inflates every off-manifold count.
Restated as a tolerance: a non-smoker smoking less than ``tol`` cigarettes a day is not
a meaningful violation.

Rules and their order come from ``dgp.RULES``; nothing here is added to it.
"""
import numpy as np
import pandas as pd

import dgp

R1 = "R1_smoker_cigs"


def check_rules_tol(df, tol: float = 0.5) -> pd.DataFrame:
    """Per-row ``__bound`` / ``__violated`` flags for every rule in ``dgp.RULES``.

    ``__bound`` is the antecedent, and is identical to ``evaluate_rules``' ``__binding``.
    The suffix differs; the pairing with ``__violated`` is what the runners rely on.
    """
    out: dict[str, np.ndarray] = {}
    for name, (_statement, antecedent, consequent) in dgp.RULES.items():
        a = np.asarray(antecedent(df))
        ok = np.asarray(df.Cigs_Per_Day < tol) if name == R1 else np.asarray(consequent(df))
        out[f"{name}__bound"] = a
        out[f"{name}__violated"] = a & ~ok
    return pd.DataFrame(out, index=df.index)
