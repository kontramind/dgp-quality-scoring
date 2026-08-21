"""Diagnostic 2: rebuild Task B under E11's own definition of harmfulness.
 
corrupt.build_populations still assigns harmful labels as y = 1[p* < 0.5] -- the rule
E5 was superseded for. Here harmful/creative are built the way inject2.py builds them:
same corrupted feature pool, same prevalence, positives placed at the lowest p*
(harmful) or in proportion to p* (creative). Label prevalence is then uninformative
by construction, so any remaining AUROC is boundary information.
"""
import numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from dgp import generate, solve_binding_rates
from corrupt import TrueRisk, apply_violations
from leafscore import LeafGeometry
import baselines as B
 
cfg = solve_binding_rates({"R3_glucose_ceil":0.08,"R4_bp_mandatory":0.20}); cfg.n_samples=60000
X,G,M = generate(cfg); y = X.pop("Heart_Disease_Risk")
Xtr,Xho,ytr,yho = train_test_split(X,y,test_size=0.5,random_state=0,stratify=y)
FEAT=list(X.columns); CAT=["Biological_Sex","Is_Pregnant","Current_Smoker","Diabetic","On_BP_Medication"]
def asc(d):
    d=d.copy()
    for c in CAT: d[c]=d[c].astype("category")
    return d
 
def place(n,n_pos,rng,weights=None,lowest=None):
    yv=np.zeros(n,int)
    if n_pos<=0: return yv
    if lowest is not None: yv[np.argsort(lowest)[:n_pos]]=1
    elif weights is not None:
        w=np.clip(weights,1e-9,None); w=w/w.sum()
        yv[rng.choice(n,size=min(n_pos,n),replace=False,p=w)]=1
    else: yv[rng.choice(n,size=min(n_pos,n),replace=False)]=1
    return yv
 
rows=[]; comp=[]
for seed in range(3):
    rng=np.random.default_rng(100+seed)
    mdl=lgb.LGBMClassifier(n_estimators=300,num_leaves=63,learning_rate=0.05,verbose=-1,
                           random_state=seed).fit(asc(Xtr),ytr,categorical_feature=CAT)
    lg=LeafGeometry(mdl).fit_reference(asc(Xtr),ytr)
    tr=TrueRisk(Xtr,M)
 
    # one corrupted pool, split in half -> identical feature distributions
    Xh=Xho.reset_index(drop=True)
    Xc,applied=apply_violations(Xh,np.arange(len(Xh)),rng)
    hit=np.array([a!="" for a in applied])
    Xc=Xc[hit].reset_index(drop=True); p=tr.p(Xc)
    perm=rng.permutation(len(Xc)); half=len(Xc)//2
    iC,iH=perm[:half],perm[half:2*half]
 
    for prev_name,prev in [("prev=0.10 (matched to base)",0.10),
                           ("prev=mean p* (coherent)",float(p.mean()))]:
        yv=np.zeros(len(Xc),int)
        yv[iC]=place(len(iC),int(round(prev*len(iC))),rng,weights=p[iC])
        yv[iH]=place(len(iH),int(round(prev*len(iH))),rng,lowest=p[iH])
        idx=np.r_[iC,iH]; lab=np.r_[np.zeros(len(iC)),np.ones(len(iH))].astype(bool)
        Xq=Xc.iloc[idx].reset_index(drop=True); yq=yv[idx]; pq=p[idx]
        ph=mdl.predict_proba(asc(Xq))[:,1]
        s=lg.score(asc(Xq),yq)
        ref=Xtr.copy(); ref["Y"]=ytr.values; qry=Xq.copy(); qry["Y"]=yq
        FY=FEAT+["Y"]
        D={"LABEL ALONE  y":yq.astype(float),
           "ORACLE |y-p*|":np.abs(yq-pq),
           "FITTED |y-p_hat|":np.abs(yq-ph),
           "FITTED leaf axis2":s["surprisal"],
           "FITTED adversarial":B.adversarial(ref,qry,FY,seed=seed),
           "FITTED DCR":B.dcr(ref,qry,FY),
           "FITTED alpha-prec(naive)":B.alpha_precision(ref,qry,FY),
           "FITTED kNN density":B.knn_density(ref,qry,FY),
           "FITTED IsolationForest":B.isolation(ref,qry,FY,seed=seed)}
        for k,v in D.items():
            rows.append({"seed":seed,"prev":prev_name,"detector":k,"AUROC":roc_auc_score(lab,v)})
        comp.append({"seed":seed,"prev":prev_name,
                     "prev_creative":yq[~lab].mean(),"prev_harmful":yq[lab].mean(),
                     "absres_creative":np.abs(yq-pq)[~lab].mean(),
                     "absres_harmful":np.abs(yq-pq)[lab].mean()})
    print("seed",seed,"done",flush=True)
 
df=pd.DataFrame(rows)
print("\n==== ARM COMPOSITION ====")
print(pd.DataFrame(comp).groupby("prev").mean(numeric_only=True).round(4).to_string())
print("\n==== TASK B, harmful_rank vs creative, PREVALENCE MATCHED (3 seeds) ====")
for pn in df.prev.unique():
    print("\n--",pn)
    g=df[df.prev==pn].groupby("detector")["AUROC"].agg(["mean","std"]).sort_values("mean",ascending=False)
    print(g.round(4).to_string())
 
