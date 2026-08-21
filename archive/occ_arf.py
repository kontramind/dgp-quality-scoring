"""Same measurement for ARF. NOTE: ARF is fitted on a 12k subsample of real train
(not the full 24k) for runtime; the quantity measured is a distributional property,
but the deviation from E12's protocol is stated rather than hidden."""
import arf_patch, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from dgp import generate, solve_binding_rates
from arf_compat import fit_arf
from corrupt import TrueRisk
from rules_tol import check_rules_tol
 
cfg=solve_binding_rates({"R3_glucose_ceil":0.08,"R4_bp_mandatory":0.20}); cfg.n_samples=40000
X,G,M=generate(cfg); FEAT=[c for c in X.columns if c!="Heart_Disease_Risk"]
Xtr,Xho=train_test_split(X,test_size=0.4,random_state=0,stratify=X.Heart_Disease_Risk)
ytr,yho=Xtr.pop("Heart_Disease_Risk"),Xho.pop("Heart_Disease_Risk")
tr=TrueRisk(Xtr,M)
def viol(d):
    v=check_rules_tol(d.assign(Heart_Disease_Risk=0))
    return v[[c for c in v.columns if c.endswith("__violated")]].any(axis=1).to_numpy()
def describe(tag,Xs,ys,rows):
    p=tr.p(Xs); vm=viol(Xs)
    for nm,k in [("ALL",np.ones(len(p),bool)),("on-manifold",~vm),("off-manifold",vm)]:
        if k.sum()<50: continue
        pk,yk=p[k],ys[k]; floor=float(np.mean(2*pk*(1-pk)))
        rows.append({"gen":tag,"subset":nm,"mass":k.mean(),"prev":yk.mean(),
                     "mean_pstar":pk.mean(),"abs_res":np.abs(yk-pk).mean(),
                     "bayes_floor":floor,"excess":np.abs(yk-pk).mean()-floor,
                     "AUC_pstar_y":roc_auc_score(yk,pk)})
rows=[]
full=Xtr.assign(Heart_Disease_Risk=ytr.values)
for seed in range(3):
    rng=np.random.default_rng(seed)
    sub=full.sample(12000,random_state=seed)
    a=fit_arf(sub,12000,seed=seed,num_trees=30)
    ya=a.pop("Heart_Disease_Risk").astype(int).to_numpy()
    describe("ARF",a[FEAT],ya,rows)
    print("seed",seed,"done",flush=True)
    pd.DataFrame(rows).to_csv("occupancy_arf.csv",index=False)
d=pd.DataFrame(rows).groupby(["gen","subset"]).mean(numeric_only=True)
d["excess_mass"]=d["mass"]*d["excess"]
print("\n==== ARF ====");print(d.round(4).to_string())
s=d.loc["ARF"]; tot=s.loc["on-manifold","excess_mass"]+s.loc["off-manifold","excess_mass"]
print(f"\nARF share of excess: on-manifold {100*s.loc['on-manifold','excess_mass']/tot:.1f}%  "
      f"off-manifold {100*s.loc['off-manifold','excess_mass']/tot:.1f}%  "
      f"(off-manifold {100*s.loc['off-manifold','mass']:.1f}% of rows)")
 
