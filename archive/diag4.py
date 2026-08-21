"""Diagnostic 4: does harm require the rows to be OFF-MANIFOLD at all?
 
E11 has arms: balance (on-manifold, labels at random) and random (off-manifold, labels
at random) -- both ~zero. It has creative and harmful_rank, both OFF-manifold. It has
never run the on-manifold x label-contradicts cell -- the paper's 'boundary corruption'
quadrant. If on-manifold anti-sorted labels do the same damage as off-manifold ones,
the 2x2 collapses to one axis and 'low support' is not part of the mechanism.
"""
import numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from dgp import generate, solve_binding_rates
from copula_fix import GaussianCopulaDT
from corrupt import TrueRisk, apply_violations
 
cfg = solve_binding_rates({"R3_glucose_ceil":0.08,"R4_bp_mandatory":0.20}); cfg.n_samples=40000
X,G,M = generate(cfg)
FEAT=[c for c in X.columns if c!="Heart_Disease_Risk"]
Xtr,Xho=train_test_split(X,test_size=0.4,random_state=0,stratify=X.Heart_Disease_Risk)
ytr,yho=Xtr.pop("Heart_Disease_Risk"),Xho.pop("Heart_Disease_Risk")
 
def place(n,n_pos,rng,weights=None,lowest=None):
    yv=np.zeros(n,int); n_pos=min(n_pos,n)
    if n_pos<=0: return yv
    if lowest is not None: yv[np.argsort(lowest)[:n_pos]]=1
    elif weights is not None:
        w=np.clip(weights,1e-9,None); w=w/w.sum()
        yv[rng.choice(n,size=n_pos,replace=False,p=w)]=1
    else: yv[rng.choice(n,size=n_pos,replace=False)]=1
    return yv
 
def recip(Xa,ya,seed):
    if len(np.unique(ya))<2: return {}
    o={}
    o["LogReg"]=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(Xa,ya)
    o["MLP"]=make_pipeline(StandardScaler(),MLPClassifier((32,),max_iter=120,random_state=seed)).fit(Xa,ya)
    o["LGBM"]=lgb.LGBMClassifier(n_estimators=150,num_leaves=31,verbose=-1,random_state=seed).fit(Xa,ya)
    return {k:roc_auc_score(yho,m.predict_proba(Xho)[:,1]) for k,m in o.items()}
 
ARMS=["clean","balance","random","creative","harmful_rank",
      "creative_ON","harmful_ON"]     # _ON = on-manifold (uncorrupted features)
 
def make_arm(kind,Xp,yp,tr,n_want,prev,rng):
    if kind in ("clean","balance","creative_ON","harmful_ON"):
        idx=rng.choice(len(Xp),size=min(n_want,len(Xp)),replace=False)
        Xc=Xp.iloc[idx].reset_index(drop=True); p=tr.p(Xc)
        npos=int(round(prev*len(Xc)))
        if kind=="clean": yv=np.asarray(yp)[idx].copy()
        elif kind=="balance": yv=place(len(Xc),npos,rng)
        elif kind=="creative_ON": yv=place(len(Xc),npos,rng,weights=p)
        else: yv=place(len(Xc),npos,rng,lowest=p)
    else:
        idx=rng.choice(len(Xp),size=min(n_want*3,len(Xp)),replace=False)
        Xs=Xp.iloc[idx].reset_index(drop=True)
        Xc,ap=apply_violations(Xs,np.arange(len(Xs)),rng)
        hit=np.array([a!="" for a in ap]); Xc=Xc[hit].reset_index(drop=True)
        Xc=Xc.iloc[:n_want].reset_index(drop=True); p=tr.p(Xc)
        npos=int(round(prev*len(Xc)))
        if kind=="random": yv=place(len(Xc),npos,rng)
        elif kind=="creative": yv=place(len(Xc),npos,rng,weights=p)
        else: yv=place(len(Xc),npos,rng,lowest=p)
    return Xc,yv,{"n":len(Xc),"prev":float(np.mean(yv)),"mean_pstar":float(np.mean(p)),
                  "corr_y_pstar":float(np.corrcoef(yv,p)[0,1])}
 
rows=[]; diag=[]
tr=TrueRisk(Xtr,M)
for seed in range(3):
    rng=np.random.default_rng(seed)
    syn=GaussianCopulaDT().fit(Xtr.assign(Heart_Disease_Risk=ytr.values),rng).sample(len(Xtr),rng)
    ysyn=syn.pop("Heart_Disease_Risk").astype(int).values; syn=syn[FEAT]
    nb=int(0.7*len(syn))
    Xb,yb=syn.iloc[:nb].reset_index(drop=True),ysyn[:nb]
    Xp,yp=syn.iloc[nb:].reset_index(drop=True),ysyn[nb:]
    prev=float(np.mean(yb))
    rows.append({"seed":seed,"arm":"BASE","rate":0.0,**recip(Xb,yb,seed)})
    for rate in [0.05,0.20]:
        for arm in ARMS:
            Xi,yi,d=make_arm(arm,Xp,yp,tr,int(rate*nb),prev,rng)
            Xa=pd.concat([Xb,Xi],ignore_index=True); ya=np.r_[yb,yi]
            diag.append({"seed":seed,"rate":rate,"arm":arm,"train_prev":float(ya.mean()),**d})
            rows.append({"seed":seed,"arm":arm,"rate":rate,**recip(Xa,ya,seed)})
    print("seed",seed,"done",flush=True)
 
df=pd.DataFrame(rows); df.to_csv("diag4_results.csv",index=False)
b=df[df.arm=="BASE"][["LogReg","MLP","LGBM"]].mean()
print("\nBASE  LogReg %.4f  MLP %.4f  LGBM %.4f"%(b.LogReg,b.MLP,b.LGBM))
print("\n==== ARM DIAGNOSTICS (mean over seeds) ====")
print(pd.DataFrame(diag).groupby(["rate","arm"])[["n","prev","train_prev","mean_pstar","corr_y_pstar"]].mean().round(4).to_string())
print("\n==== DELTA vs BASE ====")
for rate in [0.05,0.20]:
    print(f"\n-- injection rate {rate:.0%}")
    s=df[df.rate==rate]
    g=s.groupby("arm")[["LogReg","MLP","LGBM"]].agg(["mean","std"])
    for a in ARMS:
        if a not in g.index: continue
        print("  %-14s "%a + "  ".join(
            "d%s %+.4f+/-%.4f"%(m,g.loc[a,(m,'mean')]-b[m],g.loc[a,(m,'std')])
            for m in ["LogReg","MLP","LGBM"]))
 
