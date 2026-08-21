"""Occupancy, redefined: where does a generator's label error live, and is it structured?
 
excess = mean|y - p*|  -  2*E[p*(1-p*)]   computed WITHIN each subset.
The Bayes floor must be local: corruption raises p* and raises the floor with it, so
comparing a subset's mean|y-p*| against real data's global 0.1375 (as E12 did)
overstates contradiction wherever p* is elevated. Symmetric noise -> excess ~ 0.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from dgp import generate, solve_binding_rates
from copula_fix import GaussianCopulaDT
from corrupt import TrueRisk
from rules_tol import check_rules_tol
 
cfg=solve_binding_rates({"R3_glucose_ceil":0.08,"R4_bp_mandatory":0.20}); cfg.n_samples=40000
X,G,M=generate(cfg); FEAT=[c for c in X.columns if c!="Heart_Disease_Risk"]
Xtr,Xho=train_test_split(X,test_size=0.4,random_state=0,stratify=X.Heart_Disease_Risk)
ytr,yho=Xtr.pop("Heart_Disease_Risk"),Xho.pop("Heart_Disease_Risk")
tr=TrueRisk(Xtr,M)
print("Bayes AUC of the DGP:",round(M["expected_bayes_auc"],4),flush=True)
 
def viol(d):
    v=check_rules_tol(d.assign(Heart_Disease_Risk=0))
    return v[[c for c in v.columns if c.endswith("__violated")]].any(axis=1).to_numpy()
 
def describe(tag,Xs,ys,rows):
    p=tr.p(Xs); vm=viol(Xs)
    for nm,k in [("ALL",np.ones(len(p),bool)),("on-manifold",~vm),("off-manifold",vm)]:
        if k.sum()<50: continue
        pk,yk=p[k],ys[k]
        floor=float(np.mean(2*pk*(1-pk)))
        rows.append({"gen":tag,"subset":nm,"mass":k.mean(),"prev":yk.mean(),
                     "mean_pstar":pk.mean(),"abs_res":np.abs(yk-pk).mean(),
                     "bayes_floor":floor,"excess":np.abs(yk-pk).mean()-floor,
                     "AUC_pstar_y":roc_auc_score(yk,pk) if len(np.unique(yk))>1 else np.nan})
 
rows=[]
for seed in range(5):
    rng=np.random.default_rng(seed)
    describe("REAL",Xho.reset_index(drop=True),yho.to_numpy(),rows)
    cop=GaussianCopulaDT().fit(Xtr.assign(Heart_Disease_Risk=ytr.values),rng).sample(len(Xtr),rng)
    yc=cop.pop("Heart_Disease_Risk").astype(int).to_numpy()
    describe("copula",cop[FEAT],yc,rows)
    print("seed",seed,"done",flush=True)
 
d=pd.DataFrame(rows).groupby(["gen","subset"]).mean(numeric_only=True)
d["excess_mass"]=d["mass"]*d["excess"]
print("\n==== WHERE THE LABEL ERROR LIVES (5 seeds) ====")
print(d.round(4).to_string())
s=d.loc["copula"]
tot=s.loc["on-manifold","excess_mass"]+s.loc["off-manifold","excess_mass"]
print(f"\ncopula share of excess contradiction: on-manifold {100*s.loc['on-manifold','excess_mass']/tot:.1f}%  "
      f"off-manifold {100*s.loc['off-manifold','excess_mass']/tot:.1f}%  "
      f"(off-manifold is {100*s.loc['off-manifold','mass']:.1f}% of rows)")
pd.DataFrame(rows).to_csv("occupancy_copula.csv",index=False)
 
