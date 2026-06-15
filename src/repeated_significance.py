"""Repeated-CV significance: RepeatedStratifiedKFold (5x3=15) RandomForest F1 per
feature set, then paired t-tests. Checkpoints per-set fold arrays so it completes
across calls. Run repeatedly until it prints the significance table."""
import os, sys, csv, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import data_loader as dl, feature_classes as fc
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

REPO=os.path.join(os.path.dirname(__file__),".."); RES=os.path.join(RES_:=os.path.join(REPO,"results"))
GC=["g_pagerank","g_in_degree","g_out_degree","g_total_degree","g_clustering","g_kcore"]
df,_=dl.load(); df["addr"]=df["Address"].astype(str).str.strip().str.lower()
gt=pd.read_csv(os.path.join(REPO,"data","graph_topology_features.csv"))
gt["addr"]=gt["Address"].astype(str).str.strip().str.lower()
m=df.merge(gt[["addr"]+GC],on="addr",how="inner")
X,y,res=dl.prepare_xy(m)
for c in GC: X[c]=pd.to_numeric(m[c],errors="coerce").fillna(0).values
sets={"degree_graph":res["graph"],"graph_topology":GC,"transaction":res["transaction"],
      "all":res["transaction"]+res["temporal"]+res["graph"]+GC,
      "transaction+topology":res["transaction"]+GC}
cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42)
fp=os.path.join(RES,"_rep_folds.json"); fold=json.load(open(fp)) if os.path.exists(fp) else {}
for fs,cols in sets.items():
    if fs in fold: continue
    m_=RandomForestClassifier(n_estimators=80,class_weight="balanced",random_state=42,n_jobs=1)
    sc=cross_val_score(m_,X[cols].values,y,cv=cv,scoring="f1",n_jobs=1)
    fold[fs]=list(map(float,sc)); json.dump(fold,open(fp,"w"))
    print(f"computed {fs}: mean F1 {np.mean(sc):.4f} (n={len(sc)})",flush=True)
if not all(s in fold for s in sets):
    print("partial; rerun."); sys.exit(0)
comps=[("graph_topology","degree_graph"),("transaction","graph_topology"),
       ("all","transaction"),("transaction+topology","transaction")]
sig=[]
for a,b in comps:
    t,p=stats.ttest_rel(fold[a],fold[b])
    sig.append({"comparison":f"{a} vs {b}","mean_f1_a":round(np.mean(fold[a]),4),
                "mean_f1_b":round(np.mean(fold[b]),4),"f1_gain":round(np.mean(fold[a])-np.mean(fold[b]),4),
                "t_stat":round(float(t),3),"p_value":float(f"{p:.2e}"),"n_samples":len(fold[a]),
                "significant_0.05":bool(p<0.05)})
    print(f"{a} vs {b}: gain {np.mean(fold[a])-np.mean(fold[b]):+.3f} p={p:.2e}")
with open(os.path.join(RES,"significance.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(sig[0].keys())); w.writeheader(); w.writerows(sig)
print("wrote significance.csv (repeated CV, 15 samples)")
