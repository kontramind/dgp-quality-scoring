"""(1) Is the oracle gap a property of the boundary, or of one arbitrary LGBM config?
 
The holder's model p_hat is never tuned anywhere in the record (300/63/0.05, no early
stopping, no validation split). Every 'fitted' score in the paper is |y - p_hat|, so
the reported oracle gap could be the cost of THIS FIT rather than the cost of
estimating the boundary at all.
 
Protocol
--------
real train  -> split fit/val, tune by early stopping on val
real holdout-> split in half: half A measures MODEL QUALITY (AUC vs the 0.85 Bayes
               ceiling), half B builds the corrupted population. No reuse.
Task B measured under BOTH constructions:
  threshold : corrupt.py's y = 1[p* < 0.5]        (what E3/E6/E10 report)
  rank      : inject2's placement, prevalence matched, positive-label stratum
Configs deliberately span the quality range, from crippled to heavily tuned.
"""
import numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from dgp import generate, solve_binding_rates
from corrupt import TrueRisk, apply_violations, build_populations
 
cfg=solve_binding_rates({"R3_glucose_ceil":0.08,"R4_bp_mandatory":0.20}); cfg.n_samples=60000
X,G,M=generate(cfg); y=X.pop("Heart_Disease_Risk")
BAYES=M["expected_bayes_auc"]
Xtr,Xho,ytr,yho=train_test_split(X,y,test_size=0.5,random_state=0,stratify=y)
XA,XB,yA,yB=train_test_split(Xho,yho,test_size=0.5,random_state=0,stratify=yho)
CAT=["Biological_Sex","Is_Pregnant","Current_Smoker","Diabetic","On_BP_Medication"]
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
 
rs=np.random.default_rng(0)
CONFIGS=[{"name":"PAPER DEFAULT (300/63/.05)","n_estimators":300,"num_leaves":63,
          "learning_rate":0.05,"min_child_samples":20,"es":False},
         {"name":"crippled (5 trees, 2 leaves)","n_estimators":5,"num_leaves":2,
          "learning_rate":0.1,"min_child_samples":20,"es":False},
         {"name":"crippled (20 trees, 4 leaves)","n_estimators":20,"num_leaves":4,
          "learning_rate":0.05,"min_child_samples":20,"es":False},
         {"name":"overfit (2000/255/.2, no ES)","n_estimators":2000,"num_leaves":255,
          "learning_rate":0.2,"min_child_samples":5,"es":False}]
for i in range(24):
    CONFIGS.append({"name":f"rand{i:02d}",
        "n_estimators":int(rs.choice([100,300,600,1200,3000])),
        "num_leaves":int(rs.choice([7,15,31,63,127,255])),
        "learning_rate":float(rs.choice([0.01,0.02,0.05,0.1])),
        "min_child_samples":int(rs.choice([5,20,50,200,500])),
        "reg_lambda":float(rs.choice([0.0,1.0,10.0])),
        "colsample_bytree":float(rs.choice([0.6,0.8,1.0])),
        "es":True})
 
rows=[]
for seed in range(3):
    rng=np.random.default_rng(100+seed)
    tr=TrueRisk(Xtr,M)
    Xf,Xv,yf,yv=train_test_split(Xtr,ytr,test_size=0.2,random_state=seed,stratify=ytr)
    # threshold construction on half B
    Xa_t,ya_t,quad_t,_=build_populations(XB,yB,Xtr,M,seed=seed,frac=0.25)
    ps_t=tr.p(Xa_t); mt=np.isin(quad_t,["harmful","creative"]); lab_t=(quad_t[mt]=="harmful")
    # rank construction on half B
    Xh=XB.reset_index(drop=True)
    Xc,ap=apply_violations(Xh,np.arange(len(Xh)),rng)
    hit=np.array([a!="" for a in ap]); Xc=Xc[hit].reset_index(drop=True); p=tr.p(Xc)
    perm=rng.permutation(len(Xc)); h=len(Xc)//2; iC,iH=perm[:h],perm[h:2*h]
    yv_r=np.zeros(len(Xc),int)
    yv_r[iC]=place(len(iC),int(round(0.10*len(iC))),rng,weights=p[iC])
    yv_r[iH]=place(len(iH),int(round(0.10*len(iH))),rng,lowest=p[iH])
    idx=np.r_[iC,iH]; lab_r=np.r_[np.zeros(h),np.ones(h)].astype(bool)
    Xq=Xc.iloc[idx].reset_index(drop=True); yq=yv_r[idx]; pq=p[idx]
    posm=yq==1
    orc_t=roc_auc_score(lab_t,np.abs(ya_t-ps_t)[mt])
    orc_rp=roc_auc_score(lab_r[posm],np.abs(yq-pq)[posm])
    orc_rr=roc_auc_score(lab_r,np.abs(yq-pq))
 
    for c in CONFIGS:
        kw={k:v for k,v in c.items() if k not in("name","es")}
        m=lgb.LGBMClassifier(verbose=-1,random_state=seed,**kw)
        if c["es"]:
            m.fit(asc(Xf),yf,eval_set=[(asc(Xv),yv)],categorical_feature=CAT,
                  callbacks=[lgb.early_stopping(50,verbose=False)])
        else:
            m.fit(asc(Xtr),ytr,categorical_feature=CAT)
        qual=roc_auc_score(yA,m.predict_proba(asc(XA))[:,1])
        rows.append({"seed":seed,"config":c["name"],"model_AUC":qual,
                     "gap_to_bayes":BAYES-qual,
                     "taskB_threshold":roc_auc_score(lab_t,np.abs(ya_t-m.predict_proba(asc(Xa_t))[:,1])[mt]),
                     "taskB_rank_pos":roc_auc_score(lab_r[posm],np.abs(yq-m.predict_proba(asc(Xq))[:,1])[posm]),
                     "taskB_rank_pooled":roc_auc_score(lab_r,np.abs(yq-m.predict_proba(asc(Xq))[:,1])),
                     "ORACLE_threshold":orc_t,"ORACLE_rank_pos":orc_rp,"ORACLE_rank_pooled":orc_rr})
    print("seed",seed,"done",flush=True)
 
d=pd.DataFrame(rows); d.to_csv("tune_phat_results.csv",index=False)
g=d.groupby("config").mean(numeric_only=True).sort_values("model_AUC")
g["oracle_gap_thr"]=g.ORACLE_threshold-g.taskB_threshold
g["oracle_gap_rankpos"]=g.ORACLE_rank_pos-g.taskB_rank_pos
print(f"\nBayes ceiling {BAYES:.4f} | oracle: threshold {d.ORACLE_threshold.mean():.4f}  "
      f"rank/pos {d.ORACLE_rank_pos.mean():.4f}  rank/pooled {d.ORACLE_rank_pooled.mean():.4f}")
print("\n"+"="*104)
print(g[["model_AUC","gap_to_bayes","taskB_threshold","oracle_gap_thr",
         "taskB_rank_pos","oracle_gap_rankpos","taskB_rank_pooled"]].round(4).to_string())
best=g.model_AUC.idxmax(); dflt="PAPER DEFAULT (300/63/.05)"
print(f"\nBEST model     {best}: model AUC {g.loc[best,'model_AUC']:.4f}, "
      f"oracle gap (threshold) {g.loc[best,'oracle_gap_thr']:+.4f}, (rank/pos) {g.loc[best,'oracle_gap_rankpos']:+.4f}")
print(f"PAPER DEFAULT  model AUC {g.loc[dflt,'model_AUC']:.4f}, "
      f"oracle gap (threshold) {g.loc[dflt,'oracle_gap_thr']:+.4f}, (rank/pos) {g.loc[dflt,'oracle_gap_rankpos']:+.4f}")
print("\ncorrelation across configs (model quality vs detection):")
for col in ["taskB_threshold","taskB_rank_pos","taskB_rank_pooled"]:
    print(f"  {col:<20} pearson vs model_AUC {np.corrcoef(g.model_AUC,g[col])[0,1]:+.4f}")
 
