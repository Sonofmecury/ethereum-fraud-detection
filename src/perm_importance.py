"""Permutation importance (model-agnostic, more reliable than impurity importance)
on the combined model -> which individual features drive detection, and the
per-class share. Substitutes for SHAP (which would not install in-budget).
Outputs: results/top_features.csv, results/feature_importance.csv (by class)."""
import os, sys, csv
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import data_loader as dl, feature_classes as fc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

REPO = os.path.join(os.path.dirname(__file__), ".."); RES = os.path.join(REPO, "results")
GRAPH_COLS = ["g_pagerank","g_in_degree","g_out_degree","g_total_degree","g_clustering","g_kcore"]

df,_ = dl.load(); df["addr"]=df["Address"].astype(str).str.strip().str.lower()
gt=pd.read_csv(os.path.join(REPO,"data","graph_topology_features.csv"))
gt["addr"]=gt["Address"].astype(str).str.strip().str.lower()
merged=df.merge(gt[["addr"]+GRAPH_COLS],on="addr",how="inner")
X,y,resolved=dl.prepare_xy(merged)
for c in GRAPH_COLS: X[c]=pd.to_numeric(merged[c],errors="coerce").fillna(0.0).values
cols=fc.all_feature_columns(resolved)+GRAPH_COLS
Xa=X[cols]

def cls_of(c):
    if c in GRAPH_COLS: return "graph_topology"
    if c in resolved["graph"]: return "degree_graph"
    if c in resolved["transaction"]: return "transaction"
    if c in resolved["temporal"]: return "temporal"
    return "other"

Xtr,Xte,ytr,yte=train_test_split(Xa.values,y,test_size=0.3,stratify=y,random_state=42)
rf=RandomForestClassifier(n_estimators=150,class_weight="balanced",random_state=42,n_jobs=1).fit(Xtr,ytr)
pi=permutation_importance(rf,Xte,yte,scoring="average_precision",n_repeats=5,random_state=42,n_jobs=1)
imp=pi.importances_mean
order=np.argsort(imp)[::-1]
rows=[{"feature":cols[i],"feature_class":cls_of(cols[i]),"perm_importance":round(float(imp[i]),5)} for i in order]
with open(os.path.join(RES,"top_features.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["feature","feature_class","perm_importance"]); w.writeheader(); w.writerows(rows)

# aggregate by class (clip negatives to 0 for share)
agg={}
for r in rows: agg[r["feature_class"]]=agg.get(r["feature_class"],0.0)+max(0.0,r["perm_importance"])
tot=sum(agg.values()) or 1.0
cl=[{"feature_class":k,"importance_share":round(v/tot,4)} for k,v in sorted(agg.items(),key=lambda x:-x[1])]
with open(os.path.join(RES,"feature_importance.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["feature_class","importance_share"]); w.writeheader(); w.writerows(cl)
print("Top 12 features by permutation importance:")
for r in rows[:12]: print("  {:<45} {:<14} {:.4f}".format(r["feature"][:44], r["feature_class"], r["perm_importance"]))
print("\nClass shares:", {c["feature_class"]:c["importance_share"] for c in cl})
