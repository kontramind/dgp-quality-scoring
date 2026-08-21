"""Rotation test: is ARF's advantage generator quality, or alignment with an
axis-aligned DGP?
 
The DGP's constraints are threshold rules on single variables (glucose >= 126,
SBP > 140). A random forest partitions on single variables, so it can represent
those boundaries exactly. A rotation of the continuous block makes the same
constraints oblique in the coordinates the generator sees, while leaving the joint
distribution unchanged up to an orthogonal map -- so any change in generator
performance is representation alignment, not task difficulty.
 
Protocol: standardise the continuous block on real train, rotate by R(theta),
hand the rotated frame to the generator, un-rotate its output, score rules and
downstream utility in the ORIGINAL space. Both generators get identical treatment.
"""
import numpy as np, pandas as pd
from scipy.linalg import expm
 
CONT = ["Age","BMI","Fasting_Glucose","Systolic_BP","Cigs_Per_Day"]
 
def rotation(theta, d=5, seed=7):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d,d)); A = A - A.T          # skew-symmetric
    A = A / np.linalg.norm(A, 2)                          # normalise generator
    return expm(theta * A)
 
class Rotator:
    def __init__(self, X_ref, theta, rot_seed=7):
        self.mu = X_ref[CONT].mean().to_numpy()
        self.sd = X_ref[CONT].std().to_numpy()
        self.R = rotation(theta, seed=rot_seed)
        self.rot_seed = rot_seed
        self.theta = theta
    def forward(self, df):
        d = df.copy()
        Z = (d[CONT].to_numpy(float) - self.mu) / self.sd
        Zr = Z @ self.R
        d = d.drop(columns=CONT)
        for j in range(len(CONT)):
            d.insert(j, f"Rot_{j}", Zr[:, j])
        return d
    def inverse(self, df):
        d = df.copy()
        Zr = d[[f"Rot_{j}" for j in range(len(CONT))]].to_numpy(float)
        Z = Zr @ self.R.T
        d = d.drop(columns=[f"Rot_{j}" for j in range(len(CONT))])
        for j, c in enumerate(CONT):
            # round: the round-trip leaves ~1e-14 residue, which breaks the exact
            # equality in R1 (Cigs==0) for every non-smoker. 6dp is far finer than
            # any real deviation a generator produces.
            d.insert(j, c, np.round(Z[:, j] * self.sd[j] + self.mu[j], 6))
        return d
 
