"""Is harmful_ON simply a stronger contradiction than harmful_rank?
Report the contradiction magnitude of each arm (no model fitting)."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from dgp import generate, solve_binding_rates
from copula_fix import GaussianCopulaDT
from corrupt import TrueRisk, apply_violations
def place(n,n_pos,rng,weights=None,lowest=None):
    yv=np.zeros(n,int); n_pos=min(n_pos,n)
    if n_pos<=0: return yv
    if lowest is not None: yv[np.argsort(lowest)[:n_pos]]=1
    elif weights is not None:
        w=np.clip(weights,1e-9,None); w=w/w.sum()
        yv[rng.choice(n,size=n_pos,replace=False,p=w)]=1
    else: yv[rng.choice(n,size=n_pos,replace=False)]=1
    return yv
 
cfg = solve_binding_rates({"R3_glucose_ceil":0.08,"R4_bp_mandatory":0.20}); cfg.n_samples=40000
X,G,M = generate(cfg)
FEAT=[c for c in X.columns if c!="Heart_Disease_Risk"]
Xtr,Xho=train_test_split(X,test_size=0.4,random_state=0,stratify=X.Heart_Disease_Risk)
ytr,yho=Xtr.pop("Heart_Disease_Risk"),Xho.pop("Heart_Disease_Risk")
tr=TrueRisk(Xtr,M)
out=[]
for seed in range(3):
    rng=np.random.default_rng(seed)
    syn=GaussianCopulaDT().fit(Xtr.assign(Heart_Disease_Risk=ytr.values),rng).sample(len(Xtr),rng)
    ysyn=syn.pop("Heart_Disease_Risk").astype(int).values; syn=syn[FEAT]
    nb=int(0.7*len(syn)); Xp=syn.iloc[nb:].reset_index(drop=True)
    prev=float(np.mean(ysyn[:nb])); n_want=int(0.20*nb)
    # ON pool
    idx=rng.choice(len(Xp),size=n_want,replace=False)
    Xon=Xp.iloc[idx].reset_index(drop=True); pon=tr.p(Xon)
    # OFF pool
    idx2=rng.choice(len(Xp),size=min(n_want*3,len(Xp)),replace=False)
    Xs=Xp.iloc[idx2].reset_index(drop=True)
    Xc,ap=apply_violations(Xs,np.arange(len(Xs)),rng)
    hit=np.array([a!="" for a in ap]); Xoff=Xc[hit].reset_index(drop=True).iloc[:n_want].reset_index(drop=True)
    poff=tr.p(Xoff)
    for nm,p in [("ON (uncorrupted)",pon),("OFF (corrupted)",poff)]:
        npos=int(round(prev*len(p)))
        yh=place(len(p),npos,rng,lowest=p)      # harmful
        yc=place(len(p),npos,rng,weights=p)     # creative
        yr=place(len(p),npos,rng)               # random
        for arm,yv in [("harmful",yh),("creative",yc),("random",yr)]:
            out.append({"pool":nm,"arm":arm,"mean_pstar":p.mean(),
                        "mean_abs_y_minus_pstar":np.abs(yv-p).mean(),
                        "mean_pstar_of_positives":p[yv==1].mean(),
                        "corr_y_pstar":np.corrcoef(yv,p)[0,1],
                        "excess_over_bayes":np.abs(yv-p).mean()-(p*(1-p)*2).mean()})
print(pd.DataFrame(out).groupby(["pool","arm"]).mean(numeric_only=True).round(4).to_string())
 
