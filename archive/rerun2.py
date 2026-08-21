"""Detector suite: TUNED holder model, RANK construction. Sized for 1 CPU core."""
import numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from dgp import generate, solve_binding_rates
from corrupt import TrueRisk, apply_violations
from leafscore import LeafGeometry
import baselines as B
 
cfg=solve_binding_rates({"R3_glucose_ceil":0.08,"R4_bp_mandatory":0.20}); cfg.n_samples=30000
X,G,M=generate(cfg); y=X.pop("Heart_Disease_Risk"); BAYES=M["expected_bayes_auc"]
Xtr,Xho,ytr,yho=train_test_split(X,y,test_size=0.5,random_state=0,stratify=y)
FEAT=list(X.columns); CAT=["Biological_Sex","Is_Pregnant","Current_Smoker","Diabetic","On_BP_Medication"]
def asc(d):
    d=d.copy()
    for c in CAT: d[c]=d[c].astype("category")
    return d
def place(n,k,rng,lowest=None,weights=None):
    v=np.zeros(n,int); k=min(k,n)
    if lowest is not None: v[np.argsort(lowest)[:k]]=1
    elif weights is not None:
        w=np.clip(weights,1e-9,None); w=w/w.sum(); v[rng.choice(n,k,replace=False,p=w)]=1
    return v
GRID=[dict(n_estimators=400,num_leaves=31,learning_rate=0.05,min_child_samples=5),
      dict(n_estimators=400,num_leaves=63,learning_rate=0.05,min_child_samples=10),
      dict(n_estimators=400,num_leaves=31,learning_rate=0.05,min_child_samples=200),
      dict(n_estimators=400,num_leaves=15,learning_rate=0.05,min_child_samples=500),
      dict(n_estimators=200,num_leaves=63,learning_rate=0.1,min_child_samples=20),
      dict(n_estimators=600,num_leaves=15,learning_rate=0.02,min_child_samples=50)]
ADV=[dict(n_estimators=200,num_leaves=31,learning_rate=0.05),
     dict(n_estimators=300,num_leaves=63,learning_rate=0.05,min_child_samples=5)]
def adv_score(ref,qry,cols,seed,kw):
    R=ref[cols].to_numpy(float); Q=qry[cols].to_numpy(float)
    Z=np.vstack([R,Q]); lab=np.r_[np.zeros(len(R)),np.ones(len(Q))]; oof=np.zeros(len(Z))
    for a,b in StratifiedKFold(3,shuffle=True,random_state=seed).split(Z,lab):
        m=lgb.LGBMClassifier(verbose=-1,random_state=seed,**kw).fit(Z[a],lab[a])
        oof[b]=m.predict_proba(Z[b])[:,1]
    return oof[len(R):]
 
rows=[]
for seed in range(3):
    rng=np.random.default_rng(100+seed); tr=TrueRisk(Xtr,M)
    Xf,Xv,yf,yv=train_test_split(Xtr,ytr,test_size=0.2,random_state=seed,stratify=ytr)
    cand=[]
    for kw in GRID:
        m=lgb.LGBMClassifier(verbose=-1,random_state=seed,**kw)
        m.fit(asc(Xf),yf,eval_set=[(asc(Xv),yv)],categorical_feature=CAT,
              callbacks=[lgb.early_stopping(50,verbose=False)])
        pv=m.predict_proba(asc(Xv))[:,1]
        cand.append((roc_auc_score(yv,pv),-log_loss(yv,pv),m,kw))
    sel_auc=max(cand,key=lambda t:t[0]); sel_ll=max(cand,key=lambda t:t[1])
    small=[c for c in cand if c[3]["min_child_samples"]<=10]
    sel_sm=max(small,key=lambda t:t[1])
    m_def=lgb.LGBMClassifier(n_estimators=300,num_leaves=63,learning_rate=0.05,verbose=-1,
                             random_state=seed).fit(asc(Xtr),ytr,categorical_feature=CAT)
    Xh=Xho.reset_index(drop=True); yh=yho.to_numpy()
    pm=rng.permutation(len(Xh)); t=len(Xh)//3
    Xcl=Xh.iloc[pm[:t]].reset_index(drop=True); ycl=yh[pm[:t]]
    Xr=Xh.iloc[pm[t:]].reset_index(drop=True)
    Xc,ap=apply_violations(Xr,np.arange(len(Xr)),rng)
    hit=np.array([a!="" for a in ap]); Xc=Xc[hit].reset_index(drop=True); p=tr.p(Xc)
    q=rng.permutation(len(Xc)); h=len(Xc)//2; iC,iH=q[:h],q[h:2*h]
    yy=np.zeros(len(Xc),int); prev=float(ycl.mean())
    yy[iC]=place(len(iC),int(round(prev*len(iC))),rng,weights=p[iC])
    yy[iH]=place(len(iH),int(round(prev*len(iH))),rng,lowest=p[iH])
    sel=np.r_[iC,iH]
    Xoff=Xc.iloc[sel].reset_index(drop=True); yoff=yy[sel]; poff=p[sel]
    isharm=np.r_[np.zeros(h),np.ones(h)].astype(bool)
    Xall=pd.concat([Xcl,Xoff],ignore_index=True); yall=np.r_[ycl,yoff]
    pall=np.r_[tr.p(Xcl),poff]; isoff=np.r_[np.zeros(len(Xcl)),np.ones(len(Xoff))].astype(bool)
    ref=Xtr.copy(); ref["Y"]=ytr.values; qry=Xall.copy(); qry["Y"]=yall; FY=FEAT+["Y"]
    P=lambda mm: mm.predict_proba(asc(Xall))[:,1]
    s=LeafGeometry(sel_ll[2]).fit_reference(asc(Xtr),ytr).score(asc(Xall),yall)
    advs=[adv_score(ref,qry,FY,seed,kw) for kw in ADV]
    D={"ORACLE |y-p*|":np.abs(yall-pall),
       "FITTED |y-p_hat| val-selected":np.abs(yall-P(sel_ll[2])),
       "FITTED |y-p_hat| leaf-constrained":np.abs(yall-P(sel_sm[2])),
       "FITTED |y-p_hat| paper default":np.abs(yall-P(m_def)),
       "leaf axis2 (surprisal)":s["surprisal"],
       "adversarial val. (default)":advs[0],
       "adversarial val. ORACLE-TUNED":advs[int(np.argmax([roc_auc_score(isharm,a[isoff]) for a in advs]))],
       "DCR (feat+label)":B.dcr(ref,qry,FY),
       "kNN density":B.knn_density(ref,qry,FY),
       "alpha-precision (naive)":B.alpha_precision(ref,qry,FY),
       "Isolation Forest":B.isolation(ref,qry,FY,seed=seed),
       "LABEL ALONE y":yall.astype(float)}
    pos=yall==1
    for k,v in D.items():
        rows.append({"seed":seed,"detector":k,"A":roc_auc_score(isoff,v),
                     "Bp":roc_auc_score(isharm,v[isoff]),
                     "Bpos":roc_auc_score(isharm[pos[isoff]],v[isoff][pos[isoff]])})
    print(f"seed {seed} | val-pick mcs={sel_ll[3]['min_child_samples']} "
          f"auc-pick mcs={sel_auc[3]['min_child_samples']}",flush=True)
d=pd.DataFrame(rows); d.to_csv("rerun2_results.csv",index=False)
g=d.groupby("detector").agg(A=("A","mean"),As=("A","std"),Bp=("Bp","mean"),Bps=("Bp","std"),
                            Bpos=("Bpos","mean"),Bpss=("Bpos","std")).sort_values("Bpos",ascending=False)
print(f"\nBayes {BAYES:.4f}\n"+"="*94)
print(f"{'detector':<36}{'Task A off-vs-clean':>20}{'Task B pooled':>18}{'Task B y=1':>20}")
print("-"*94)
for i,r in g.iterrows():
    print(f"{i:<36}{r.A:>12.4f}+/-{r.As:.4f}{r.Bp:>11.4f}+/-{r.Bps:.4f}{r.Bpos:>13.4f}+/-{r.Bpss:.4f}")
 
