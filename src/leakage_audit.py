"""Leakage audit: re-run the key comparisons using TIME-RESPECTING graph-topology
features (each account's earliest 70% of transactions) and compare to the
full-graph results. If topology still beats degree and still adds to transaction,
the headline is robust to temporal leakage."""
import csv, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import data_loader as dl, feature_classes as fc
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

REPO=os.path.join(os.path.dirname(__file__),".."); RES=os.path.join(REPO,"results")
GC=["g_pagerank","g_in_degree","g_out_degree","g_total_degree","g_clustering","g_kcore"]
SCOR={"f1":"f1","pr_auc":"average_precision"}

df,_=dl.load(); df["addr"]=df["Address"].astype(str).str.strip().str.lower()
gt=pd.read_csv(os.path.join(REPO,"data","graph_topology_features_timesplit.csv"))
gt["addr"]=gt["Address"].astype(str).str.strip().str.lower()
m=df.merge(gt[["addr"]+GC],on="addr",how="inner")
X,y,res=dl.prepare_xy(m)
for c in GC: X[c]=pd.to_numeric(m[c],errors="coerce").fillna(0).values
T=res["transaction"]
sets={"degree_graph":res["graph"],"graph_topology_TIMESPLIT":GC,
      "transaction":T,"transaction+topology_TIMESPLIT":T+GC}
cv=StratifiedKFold(5,shuffle=True,random_state=42)
def mk(): return {"RandomForest":RandomForestClassifier(n_estimators=120,class_weight="balanced",random_state=42,n_jobs=1),
                  "HistGBDT":HistGradientBoostingClassifier(max_iter=200,class_weight="balanced",random_state=42)}
out=os.path.join(RES,"leakage_audit.csv"); rows=[]; done=set()
if os.path.exists(out):
    rows=list(csv.DictReader(open(out))); done={(r["feature_set"],r["model"]) for r in rows}
flds=["feature_set","model","f1","pr_auc"]
for fs,cols in sets.items():
    for mn,mdl in mk().items():
        if (fs,mn) in done: continue
        cvr=cross_validate(mdl,X[cols].values,y,cv=cv,scoring=SCOR,n_jobs=1)
        r={"feature_set":fs,"model":mn,"f1":round(float(np.mean(cvr["test_f1"])),4),
           "pr_auc":round(float(np.mean(cvr["test_pr_auc"])),4)}
        rows.append(r); done.add((fs,mn))
        with open(out,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=flds); w.writeheader()
            for rr in rows: w.writerow({k:rr.get(k) for k in flds})
        print("{:<30} {:<13} F1={} PR-AUC={}".format(fs,mn,r["f1"],r["pr_auc"]),flush=True)
print("done" if len(rows)==len(sets)*2 else f"partial {len(rows)}/{len(sets)*2}")
