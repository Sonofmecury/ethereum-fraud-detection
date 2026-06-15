"""
run_experiment.py -- feature-class comparison for Ethereum fraud detection
(extended: adds a REAL graph-topology class, a boosted-tree model, and paired
significance tests).

Feature classes:
  degree_graph    account-level degree/diversity counts (the original "graph")
  graph_topology  TRUE network features from the raw tx graph (PageRank, in/out
                  degree centrality, clustering, k-core) -- see build_graph_features.py
  transaction     value & volume
  temporal        timing
  all             union of the above

Models: Logistic Regression (linear), Random Forest (bagged trees),
HistGradientBoosting (boosted trees; the built-in XGBoost-equivalent).

Evaluation: stratified 5-fold CV; imbalance-aware metrics (P, R, F1, ROC-AUC,
PR-AUC). Paired t-tests (per-fold F1) test whether class differences are real.

NOTE: rows are the INNER JOIN of the labelled set with graph_topology features
(addresses that have raw transactions), so every feature set is compared on the
SAME accounts.

Outputs: results/feature_class_results.csv, results/significance.csv, results/run_meta.json
Run:  python3 src/run_experiment.py
"""
from __future__ import annotations

import csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import data_loader as dl
import feature_classes as fc

from scipy import stats
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(REPO, "results")
GRAPH_CSV = os.path.join(REPO, "data", "graph_topology_features.csv")
GRAPH_COLS = ["g_pagerank", "g_in_degree", "g_out_degree", "g_total_degree",
              "g_clustering", "g_kcore"]
SCORING = {"precision": "precision", "recall": "recall", "f1": "f1",
           "roc_auc": "roc_auc", "pr_auc": "average_precision"}


def models():
    return {
        "LogReg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(n_estimators=120, class_weight="balanced",
                                               random_state=42, n_jobs=1),
        "HistGBDT": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                   class_weight="balanced", random_state=42),
    }


def main():
    os.makedirs(RESULTS, exist_ok=True)
    import pandas as pd
    df, source = dl.load()
    df["addr"] = df["Address"].astype(str).str.strip().str.lower()
    gt = pd.read_csv(GRAPH_CSV); gt["addr"] = gt["Address"].astype(str).str.strip().str.lower()
    merged = df.merge(gt[["addr"] + GRAPH_COLS], on="addr", how="inner")
    print(f"merged rows: {len(merged)} (have graph-topology features); "
          f"fraud rate {pd.to_numeric(merged['FLAG']).mean():.3f}", flush=True)

    X_all, y, resolved = dl.prepare_xy(merged)
    for c in GRAPH_COLS:
        X_all[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0).values

    feature_sets = {
        "degree_graph": resolved["graph"],
        "graph_topology": GRAPH_COLS,
        "transaction": resolved["transaction"],
        "temporal": resolved["temporal"],
        "all": fc.all_feature_columns(resolved) + GRAPH_COLS,
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    res_csv = os.path.join(RESULTS, "feature_class_results.csv")
    foldf = os.path.join(RESULTS, "_rf_fold_f1.json")
    done = set()
    rows = []
    if os.path.exists(res_csv):
        with open(res_csv) as f:
            for r in csv.DictReader(f):
                done.add((r["feature_set"], r["model"])); rows.append(r)
    fold_f1 = json.load(open(foldf)) if os.path.exists(foldf) else {}
    fold_f1 = {}  # (feature_set) -> per-fold F1 for RandomForest (for significance)
    fieldnames = ["feature_set","n_features","model"] + [m+s2 for m in SCORING for s2 in ("_mean","_std")]
    for fs, cols in feature_sets.items():
        Xs = X_all[cols].values
        for mname, model in models().items():
            if (fs, mname) in done:
                continue
            cvres = cross_validate(model, Xs, y, cv=cv, scoring=SCORING, n_jobs=1)
            row = {"feature_set": fs, "n_features": len(cols), "model": mname}
            for met in SCORING:
                row[met + "_mean"] = round(float(np.mean(cvres["test_" + met])), 4)
                row[met + "_std"] = round(float(np.std(cvres["test_" + met])), 4)
            rows.append(row); done.add((fs, mname))
            if mname == "RandomForest":
                fold_f1[fs] = list(cvres["test_f1"])
            # checkpoint immediately
            with open(res_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
                for rr in rows: w.writerow({k: rr.get(k) for k in fieldnames})
            json.dump(fold_f1, open(foldf, "w"))
            print("{:<14} {:<13} F1={:.3f} PR-AUC={:.3f}".format(fs, mname, row["f1_mean"], row["pr_auc_mean"]), flush=True)

    n_expected = len(feature_sets) * len(models())
    if len(rows) < n_expected:
        print(f"PARTIAL: {len(rows)}/{n_expected} combos done; rerun to continue.", flush=True)
        return 0

    # paired significance tests on RandomForest per-fold F1
    comps = [("graph_topology", "degree_graph"), ("transaction", "graph_topology"),
             ("transaction", "temporal"), ("all", "transaction")]
    sig = []
    for a, b in comps:
        t, p = stats.ttest_rel(fold_f1[a], fold_f1[b])
        sig.append({"comparison": f"{a} vs {b}", "mean_f1_a": round(float(np.mean(fold_f1[a])), 4),
                    "mean_f1_b": round(float(np.mean(fold_f1[b])), 4),
                    "t_stat": round(float(t), 3), "p_value": round(float(p), 4),
                    "significant_0.05": bool(p < 0.05)})
        print("  sig {:<28} dF1={:+.3f} p={:.4f}".format(f"{a} vs {b}",
              float(np.mean(fold_f1[a]) - np.mean(fold_f1[b])), float(p)))
    with open(os.path.join(RESULTS, "significance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sig[0].keys())); w.writeheader(); w.writerows(sig)

    meta = {"data_source": source, "n_rows_with_graph": int(len(merged)),
            "fraud_rate": round(float(np.mean(y)), 4), "cv_folds": 5,
            "models": list(models().keys()),
            "feature_counts": {k: len(v) for k, v in feature_sets.items()},
            "note_significance": "Paired t-test on RandomForest per-fold F1 (n=5 folds; small-sample)."}
    with open(os.path.join(RESULTS, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\nDATA:", source.upper(), "| rows:", len(merged), "| wrote results CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
