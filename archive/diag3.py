"""Diagnostic 3: decompose Task B by label.
 
If |y-p_hat| separates harmful from creative only because the two arms differ in class
balance, then WITHIN y=1 rows and WITHIN y=0 rows it should be near chance. If it
separates within each label stratum, the pooled number is honest and the label-alone
baseline is the artifact.
Run on BOTH constructions: corrupt.py (threshold) and rank-based (matched).
"""
import numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from dgp import generate, solve_binding_rates
from corrupt import TrueRisk, apply_violations, build_populations
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
    if lowest is not None: yv[np.argsort(lowest)[:n_pos]]=1
    elif weights is not None:
        w=np.clip(weights,1e-9,None); w/=w.sum()
        yv[rng.choice(n,size=n_pos,replace=False,p=w)]=1
    return yv
 
def strat(lab,score,yq,tag,out,det):
    for nm,k in [("pooled",np.ones(len(lab),bool)),("y=1 only",yq==1),("y=0 only",yq==0)]:
        if len(np.unique(lab[k]))<2: continue
        out.append({"build":tag,"detector":det,"stratum":nm,"n":int(k.sum()),
                    "frac_harmful":lab[k].mean(),"AUROC":roc_auc_score(lab[k],score[k])})
 
out=[]
for seed in range(3):
    rng=np.random.default_rng(100+seed)
    mdl=lgb.LGBMClassifier(n_estimators=300,num_leaves=63,learning_rate=0.05,verbose=-1,
                           random_state=seed).fit(asc(Xtr),ytr,categorical_feature=CAT)
    tr=TrueRisk(Xtr,M)
 
    # --- construction A: corrupt.py as used in E3/E6/E10 ---
    Xa,ya,quad,_=build_populations(Xho,yho,Xtr,M,seed=seed,frac=0.25)
    ps=tr.p(Xa); ph=mdl.predict_proba(asc(Xa))[:,1]
    m=np.isin(quad,["harmful","creative"]); lab=(quad[m]=="harmful")
    strat(lab,np.abs(ya-ps)[m],ya[m],"A threshold y=1[p*<0.5]",out,"ORACLE |y-p*|")
    strat(lab,np.abs(ya-ph)[m],ya[m],"A threshold y=1[p*<0.5]",out,"FITTED |y-p_hat|")
    strat(lab,B.dcr(Xtr.assign(Y=ytr.values),Xa.assign(Y=ya),FEAT+["Y"])[m],ya[m],
          "A threshold y=1[p*<0.5]",out,"FITTED DCR")
 
    # --- construction B: rank-based, prevalence matched (inject2 definition) ---
    Xh=Xho.reset_index(drop=True)
    Xc,applied=apply_violations(Xh,np.arange(len(Xh)),rng)
    hit=np.array([a!="" for a in applied]); Xc=Xc[hit].reset_index(drop=True); p=tr.p(Xc)
    perm=rng.permutation(len(Xc)); h=len(Xc)//2; iC,iH=perm[:h],perm[h:2*h]
    yv=np.zeros(len(Xc),int)
    yv[iC]=place(len(iC),int(round(0.10*len(iC))),rng,weights=p[iC])
    yv[iH]=place(len(iH),int(round(0.10*len(iH))),rng,lowest=p[iH])
    idx=np.r_[iC,iH]; lab2=np.r_[np.zeros(h),np.ones(h)].astype(bool)
    Xq=Xc.iloc[idx].reset_index(drop=True); yq=yv[idx]; pq=p[idx]
    ph2=mdl.predict_proba(asc(Xq))[:,1]
    strat(lab2,np.abs(yq-pq),yq,"B rank, prev matched",out,"ORACLE |y-p*|")
    strat(lab2,np.abs(yq-ph2),yq,"B rank, prev matched",out,"FITTED |y-p_hat|")
    strat(lab2,B.dcr(Xtr.assign(Y=ytr.values),Xq.assign(Y=yq),FEAT+["Y"]),yq,
          "B rank, prev matched",out,"FITTED DCR")
    print("seed",seed,"done",flush=True)
 
d=pd.DataFrame(out)
g=d.groupby(["build","detector","stratum"]).agg(n=("n","mean"),frac_harm=("frac_harmful","mean"),
    AUROC=("AUROC","mean"),sd=("AUROC","std")).round(4)
print("\n==== TASK B DECOMPOSED BY LABEL ====")
print(g.to_string())
 
