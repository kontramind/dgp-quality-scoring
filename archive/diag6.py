"""Decisive 2x2 test: match the p* distribution across the ON and OFF pools, then
place positives by the SAME rank rule in both. Identical label-vs-boundary
relationship; the only difference is whether the features violate a joint constraint.
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
from rules_tol import check_rules_tol
 
cfg = solve_binding_rates({"R3_glucose_ceil":0.08,"R4_bp_mandatory":0.20}); cfg.n_samples=40000
X,G,M = generate(cfg)
FEAT=[c for c in X.columns if c!="Heart_Disease_Risk"]
Xtr,Xho=train_test_split(X,test_size=0.4,random_state=0,stratify=X.Heart_Disease_Risk)
ytr,yho=Xtr.pop("Heart_Disease_Risk"),Xho.pop("Heart_Disease_Risk")
tr=TrueRisk(Xtr,M)
 
def place(n,n_pos,rng,weights=None,lowest=None):
    yv=np.zeros(n,int); n_pos=min(n_pos,n)
    if n_pos<=0: return yv
    if lowest is not None: yv[np.argsort(lowest)[:n_pos]]=1
    elif weights is not None:
        w=np.clip(weights,1e-9,None); w=w/w.sum()
        yv[rng.choice(n,size=n_pos,replace=False,p=w)]=1
    else: yv[rng.choice(n,size=n_pos,replace=False)]=1
    return yv
 
def match_to(p_src, X_tgt, p_tgt, rng, nbins=20):
    """Resample X_tgt so its p* distribution matches p_src's."""
    edges=np.quantile(p_src,np.linspace(0,1,nbins+1)); edges[0]=-np.inf; edges[-1]=np.inf
    want=np.histogram(p_src,bins=edges)[0]
    bi=np.digitize(p_tgt,edges)-1
    take=[]
    for b in range(nbins):
        pool=np.flatnonzero(bi==b)
        if len(pool)==0 or want[b]==0: continue
        take.append(rng.choice(pool,size=want[b],replace=len(pool)<want[b]))
    take=np.concatenate(take)
    return X_tgt.iloc[take].reset_index(drop=True), p_tgt[take]
 
def recip(Xa,ya,seed):
    o={}
    o["LogReg"]=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(Xa,ya)
    o["MLP"]=make_pipeline(StandardScaler(),MLPClassifier((32,),max_iter=120,random_state=seed)).fit(Xa,ya)
    o["LGBM"]=lgb.LGBMClassifier(n_estimators=150,num_leaves=31,verbose=-1,random_state=seed).fit(Xa,ya)
    return {k:roc_auc_score(yho,m.predict_proba(Xho)[:,1]) for k,m in o.items()}
 
rows=[]; dg=[]
for seed in range(5):
    rng=np.random.default_rng(seed)
    syn=GaussianCopulaDT().fit(Xtr.assign(Heart_Disease_Risk=ytr.values),rng).sample(len(Xtr),rng)
    ysyn=syn.pop("Heart_Disease_Risk").astype(int).values; syn=syn[FEAT]
    nb=int(0.7*len(syn))
    Xb,yb=syn.iloc[:nb].reset_index(drop=True),ysyn[:nb]
    Xp=syn.iloc[nb:].reset_index(drop=True)
    prev=float(np.mean(yb)); n_want=int(0.20*nb)
    rows.append({"seed":seed,"arm":"BASE",**recip(Xb,yb,seed)})
 
    # OFF pool (corrupted) -- the reference p* distribution
    i2=rng.choice(len(Xp),size=min(n_want*3,len(Xp)),replace=False)
    Xs=Xp.iloc[i2].reset_index(drop=True)
    Xc,ap=apply_violations(Xs,np.arange(len(Xs)),rng)
    hit=np.array([a!="" for a in ap])
    Xoff=Xc[hit].reset_index(drop=True).iloc[:n_want].reset_index(drop=True); poff=tr.p(Xoff)
    # ON pool resampled to match OFF's p* distribution
    Xon_all=Xp.reset_index(drop=True); pon_all=tr.p(Xon_all)
    Xon,pon=match_to(poff,Xon_all,pon_all,rng)
 
    npos=int(round(prev*n_want))
    for pool,Xpool,p in [("OFF",Xoff,poff),("ON",Xon,pon)]:
        for arm,yv in [("harmful",place(len(p),npos,rng,lowest=p)),
                       ("creative",place(len(p),npos,rng,weights=p)),
                       ("random",place(len(p),npos,rng))]:
            Xa=pd.concat([Xb,Xpool],ignore_index=True); ya=np.r_[yb,yv]
            viol=check_rules_tol(Xpool.assign(Heart_Disease_Risk=yv))
            vr=float(viol[[c for c in viol.columns if c.endswith("__violated")]].any(axis=1).mean())
            dg.append({"seed":seed,"pool":pool,"arm":arm,"mean_pstar":p.mean(),
                       "mean_abs_res":np.abs(yv-p).mean(),"pstar_of_pos":p[yv==1].mean(),
                       "viol_rate":vr,"train_prev":ya.mean()})
            rows.append({"seed":seed,"arm":f"{pool}_{arm}",**recip(Xa,ya,seed)})
    print("seed",seed,"done",flush=True)
 
df=pd.DataFrame(rows); df.to_csv("diag6_results.csv",index=False)
print("\n==== POOLS AFTER p* MATCHING ====")
print(pd.DataFrame(dg).groupby(["pool","arm"])[["mean_pstar","pstar_of_pos","mean_abs_res","viol_rate","train_prev"]].mean().round(4).to_string())
b=df[df.arm=="BASE"][["LogReg","MLP","LGBM"]].mean()
print("\nBASE  LogReg %.4f  MLP %.4f  LGBM %.4f   (5 seeds, 20%% injection)"%(b.LogReg,b.MLP,b.LGBM))
g=df[df.arm!="BASE"].groupby("arm")[["LogReg","MLP","LGBM"]].agg(["mean","std"])
print("\n==== DELTA vs BASE ====")
for a in ["OFF_random","ON_random","OFF_creative","ON_creative","OFF_harmful","ON_harmful"]:
    print("  %-13s "%a + "  ".join("d%s %+.4f+/-%.4f"%(m,g.loc[a,(m,'mean')]-b[m],g.loc[a,(m,'std')])
                                   for m in ["LogReg","MLP","LGBM"]))
 
