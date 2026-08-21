"""Diagnostic 1: what is Task B actually measuring?
 
E3/E6/E10 all build the harmful quadrant with corrupt.py's y = 1[p* < 0.5] --
the operationalisation E5 was superseded for. Check the class balance of the two
arms and how much of the detection AUROC is available from the LABEL ALONE.
"""
import numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from dgp import generate, solve_binding_rates
from corrupt import build_populations, TrueRisk
 
cfg = solve_binding_rates({"R3_glucose_ceil":0.08,"R4_bp_mandatory":0.20}); cfg.n_samples=60000
X,G,M = generate(cfg); y = X.pop("Heart_Disease_Risk")
Xtr,Xho,ytr,yho = train_test_split(X,y,test_size=0.5,random_state=0,stratify=y)
CAT=["Biological_Sex","Is_Pregnant","Current_Smoker","Diabetic","On_BP_Medication"]
def asc(d):
    d=d.copy()
    for c in CAT: d[c]=d[c].astype("category")
    return d
 
rows=[]; quaddesc=[]
for seed in range(3):
    m=lgb.LGBMClassifier(n_estimators=300,num_leaves=63,learning_rate=0.05,verbose=-1,
                         random_state=seed).fit(asc(Xtr),ytr,categorical_feature=CAT)
    Xa,ya,quad,_ = build_populations(Xho,yho,Xtr,M,seed=seed,frac=0.25)
    tr=TrueRisk(Xtr,M); p_star=tr.p(Xa)
    p_hat=m.predict_proba(asc(Xa))[:,1]
    for q in ["clean","label","creative","harmful"]:
        k=quad==q
        quaddesc.append({"seed":seed,"quad":q,"n":int(k.sum()),"prev":ya[k].mean(),
                         "mean_pstar":p_star[k].mean(),
                         "mean_abs_y_minus_pstar":np.abs(ya[k]-p_star[k]).mean()})
    mask=np.isin(quad,["harmful","creative"]); lab=(quad[mask]=="harmful")
    D={"LABEL ALONE  y":ya[mask].astype(float),
       "ORACLE |y-p*|":np.abs(ya-p_star)[mask],
       "FITTED |y-p_hat|":np.abs(ya-p_hat)[mask],
       "p_hat alone (inv)":-p_hat[mask],
       "p* alone (inv)":-p_star[mask]}
    for k,v in D.items():
        rows.append({"seed":seed,"detector":k,"AUROC":roc_auc_score(lab,v)})
    print("seed",seed,"done",flush=True)
 
qd=pd.DataFrame(quaddesc).groupby("quad").mean(numeric_only=True).round(4)
print("\n==== QUADRANT COMPOSITION (corrupt.py, 3 seeds) ====")
print(qd.to_string())
print("\n==== TASK B (harmful vs creative) ====")
g=pd.DataFrame(rows).groupby("detector")["AUROC"].agg(["mean","std"]).round(4)
print(g.to_string())
 
