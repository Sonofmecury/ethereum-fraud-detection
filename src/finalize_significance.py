"""Finalize: compute RF per-fold F1 for the two graph feature sets, combine with
the saved transaction/temporal/all folds, write significance.csv + run_meta.json."""
import json, os, sys, csv
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import data_loader as dl, feature_classes as fc
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

REPO = os.path.join(os.path.dirname(__file__), ".."); RES = os.path.join(REPO, "results")
GRAPH_COLS = ["g_pagerank","g_in_degree","g_out_degree","g_total_degree","g_clustering","g_kcore"]

df, source = dl.load(); df["addr"] = df["Address"].astype(str).str.strip().str.lower()
gt = pd.read_csv(os.path.join(REPO,"data","graph_topology_features.csv"))
gt["addr"] = gt["Address"].astype(str).str.strip().str.lower()
merged = df.merge(gt[["addr"]+GRAPH_COLS], on="addr", how="inner")
X, y, resolved = dl.prepare_xy(merged)
for c in GRAPH_COLS: X[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0).values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
def rf_folds(cols):
    m = RandomForestClassifier(n_estimators=120, class_weight="balanced", random_state=42, n_jobs=1)
    return list(cross_val_score(m, X[cols].values, y, cv=cv, scoring="f1", n_jobs=1))

fold = json.load(open(os.path.join(RES,"_rf_fold_f1.json")))
fold["degree_graph"] = rf_folds(resolved["graph"])
fold["graph_topology"] = rf_folds(GRAPH_COLS)
json.dump(fold, open(os.path.join(RES,"_rf_fold_f1.json"),"w"))

comps = [("graph_topology","degree_graph"),("transaction","graph_topology"),
         ("transaction","temporal"),("all","transaction")]
sig=[]
for a,b in comps:
    t,p = stats.ttest_rel(fold[a], fold[b])
    sig.append({"comparison":f"{a} vs {b}","mean_f1_a":round(float(np.mean(fold[a])),4),
                "mean_f1_b":round(float(np.mean(fold[b])),4),"delta_f1":round(float(np.mean(fold[a])-np.mean(fold[b])),4),
                "t_stat":round(float(t),3),"p_value":round(float(p),4),"significant_0.05":bool(p<0.05)})
    print(f"{a} vs {b}: dF1={np.mean(fold[a])-np.mean(fold[b]):+.3f} p={p:.4f} sig={p<0.05}")
with open(os.path.join(RES,"significance.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(sig[0].keys())); w.writeheader(); w.writerows(sig)

meta={"data_source":source,"n_rows_with_graph":int(len(merged)),"fraud_rate":round(float(np.mean(y)),4),
      "cv_folds":5,"models":["LogReg","RandomForest","HistGBDT"],
      "graph_features_note":"true topology from raw tx graph (242k nodes, 1.65M edges); betweenness omitted (O(VE))",
      "significance_note":"paired t-test on RandomForest per-fold F1, n=5 folds (small-sample)"}
json.dump(meta, open(os.path.join(RES,"run_meta.json"),"w"), indent=2)
print("wrote significance.csv + run_meta.json")
