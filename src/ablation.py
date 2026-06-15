"""Ablation: incremental value of each feature class ON TOP OF transaction features.
Sets: transaction, +degree_graph, +temporal, +graph_topology, all. Checkpointed."""
import os, sys, csv, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import data_loader as dl, feature_classes as fc
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

REPO=os.path.join(os.path.dirname(__file__),".."); RES=os.path.join(REPO,"results")
GC=["g_pagerank","g_in_degree","g_out_degree","g_total_degree","g_clustering","g_kcore"]
df,_=dl.load(); df["addr"]=df["Address"].astype(str).str.strip().str.lower()
gt=pd.read_csv(os.path.join(REPO,"data","graph_topology_features.csv"))
gt["addr"]=gt["Address"].astype(str).str.strip().str.lower()
m=df.merge(gt[["addr"]+GC],on="addr",how="inner")
X,y,res=dl.prepare_xy(m)
for c in GC: X[c]=pd.to_numeric(m[c],errors="coerce").fillna(0).values
T=res["transaction"]
sets={"transaction":T,"transaction+degree_graph":T+res["graph"],
      "transaction+temporal":T+res["temporal"],"transaction+topology":T+GC,
      "all":T+res["temporal"]+res["graph"]+GC}
SCOR={"f1":"f1","pr_auc":"average_precision","roc_auc":"roc_auc"}
cv=StratifiedKFold(5,shuffle=True,random_state=42)
def mk():
    return {"RandomForest":RandomForestClassifier(n_estimators=120,class_weight="balanced",random_state=42,n_jobs=1),
            "HistGBDT":HistGradientBoostingClassifier(max_iter=200,class_weight="balanced",random_state=42)}
out=os.path.join(RES,"ablation.csv"); rows=[]; done=set()
if os.path.exists(out):
    rows=list(csv.DictReader(open(out))); done={(r["feature_set"],r["model"]) for r in rows}
fields=["feature_set","n_features","model","f1","pr_auc","roc_auc"]
for fs,cols in sets.items():
    for mn,mdl in mk().items():
        if (fs,mn) in done: continue
        cv_=cross_validate(mdl,X[cols].values,y,cv=cv,scoring=SCOR,n_jobs=1)
        r={"feature_set":fs,"n_features":len(cols),"model":mn,
           "f1":round(float(np.mean(cv_["test_f1"])),4),
           "pr_auc":round(float(np.mean(cv_["test_pr_auc"])),4),
           "roc_auc":round(float(np.mean(cv_["test_roc_auc"])),4)}
        rows.append(r); done.add((fs,mn))
        with open(out,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            for rr in rows: w.writerow({k:rr.get(k) for k in fields})
        print("{:<28} {:<13} F1={} PR-AUC={}".format(fs,mn,r["f1"],r["pr_auc"]),flush=True)
print("done" if len(rows)==len(sets)*2 else f"partial {len(rows)}/{len(sets)*2}")
