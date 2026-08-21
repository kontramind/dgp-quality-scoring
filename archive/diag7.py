"""Same as diag6 but the ON pool is restricted to rows that violate NO rule."""
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
 
cfg=solve_binding_rates({"R3_glucose_ceil":0.08,"R4_bp_mandatory":0.20}); cfg.n_samples=40000
X,G,M=generate(cfg); FEAT=[c for c in X.columns if c!="Heart_Disease_Risk"]
Xtr,Xho=train_test_split(X,test_size=0.4,random_state=0,stratify=X.Heart_Disease_Risk)
ytr,yho=Xtr.pop("Heart_Disease_Risk"),Xho.pop("Heart_Disease_Risk")
tr=TrueRisk(Xtr,M)
def viol_mask(d):
    v=check_rules_tol(d.assign(Heart_Disease_Risk=0))
    return v[[c for c in v.columns if c.endswith("__violated")]].any(axis=1).to_numpy()
def place(n,k,rng,lowest=None,weights=None):
    y=np.zeros(n,int); k=min(k,n)
    if lowest is not None: y[np.argsort(lowest)[:k]]=1
    elif weights is not None:
        w=np.clip(weights,1e-9,None); w/=w.sum(); y[rng.choice(n,k,replace=False,p=w)]=1
    else: y[rng.choice(n,k,replace=False)]=1
    return y
def match_to(p_src,X_t,p_t,rng,nbins=20):
    e=np.quantile(p_src,np.linspace(0,1,nbins+1)); e[0]=-np.inf; e[-1]=np.inf
    want=np.histogram(p_src,bins=e)[0]; bi=np.digitize(p_t,e)-1; take=[]
    for b in range(nbins):
        pool=np.flatnonzero(bi==b)
        if len(pool)==0 or want[b]==0: continue
        take.append(rng.choice(pool,size=want[b],replace=len(pool)<want[b]))
    take=np.concatenate(take); return X_t.iloc[take].reset_index(drop=True),p_t[take]
def recip(Xa,ya,seed):
    o={"LogReg":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(Xa,ya),
       "MLP":make_pipeline(StandardScaler(),MLPClassifier((32,),max_iter=120,random_state=seed)).fit(Xa,ya),
       "LGBM":lgb.LGBMClassifier(n_estimators=150,num_leaves=31,verbose=-1,random_state=seed).fit(Xa,ya)}
    return {k:roc_auc_score(yho,m.predict_proba(Xho)[:,1]) for k,m in o.items()}
 
rows=[];dg=[]
for seed in range(5):
    rng=np.random.default_rng(seed)
    syn=GaussianCopulaDT().fit(Xtr.assign(Heart_Disease_Risk=ytr.values),rng).sample(len(Xtr),rng)
    ysyn=syn.pop("Heart_Disease_Risk").astype(int).values; syn=syn[FEAT]
    nb=int(0.7*len(syn)); Xb,yb=syn.iloc[:nb].reset_index(drop=True),ysyn[:nb]
    Xp=syn.iloc[nb:].reset_index(drop=True); prev=float(np.mean(yb)); nw=int(0.20*nb)
    rows.append({"seed":seed,"arm":"BASE",**recip(Xb,yb,seed)})
    i2=rng.choice(len(Xp),size=min(nw*3,len(Xp)),replace=False)
    Xs=Xp.iloc[i2].reset_index(drop=True); Xc,ap=apply_violations(Xs,np.arange(len(Xs)),rng)
    hit=np.array([a!="" for a in ap])
    Xoff=Xc[hit].reset_index(drop=True).iloc[:nw].reset_index(drop=True); poff=tr.p(Xoff)
    clean=~viol_mask(Xp); Xcl=Xp[clean].reset_index(drop=True); pcl=tr.p(Xcl)
    Xon,pon=match_to(poff,Xcl,pcl,rng)
    k=int(round(prev*nw))
    for pool,Xq,p in [("OFF",Xoff,poff),("ON_strict",Xon,pon)]:
        for arm,yv in [("harmful",place(len(p),k,rng,lowest=p)),("creative",place(len(p),k,rng,weights=p))]:
            Xa=pd.concat([Xb,Xq],ignore_index=True); ya=np.r_[yb,yv]
            dg.append({"pool":pool,"arm":arm,"n":len(p),"mean_pstar":p.mean(),
                       "pstar_of_pos":p[yv==1].mean(),"mean_abs_res":np.abs(yv-p).mean(),
                       "viol_rate":viol_mask(Xq).mean(),"uniq_frac":len(np.unique(p))/len(p)})
            rows.append({"seed":seed,"arm":f"{pool}_{arm}",**recip(Xa,ya,seed)})
    print("seed",seed,"done",flush=True)
df=pd.DataFrame(rows)
print("\n==== POOLS ====");print(pd.DataFrame(dg).groupby(["pool","arm"]).mean(numeric_only=True).round(4).to_string())
b=df[df.arm=="BASE"][["LogReg","MLP","LGBM"]].mean()
print("\nBASE  LogReg %.4f  MLP %.4f  LGBM %.4f  (5 seeds, 20%% injection)"%(b.LogReg,b.MLP,b.LGBM))
g=df[df.arm!="BASE"].groupby("arm")[["LogReg","MLP","LGBM"]].agg(["mean","std"])
for a in ["OFF_creative","ON_strict_creative","OFF_harmful","ON_strict_harmful"]:
    print("  %-19s "%a+"  ".join("d%s %+.4f+/-%.4f"%(m,g.loc[a,(m,'mean')]-b[m],g.loc[a,(m,'std')]) for m in ["LogReg","MLP","LGBM"]))
 
