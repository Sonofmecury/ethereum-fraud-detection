"""
build_graph_features_timesplit.py -- LEAKAGE AUDIT.

Rebuild graph-topology features using ONLY each account's EARLIEST 70% of
transactions (per-account time-respecting split). This excludes each account's
most recent activity -- the activity most likely to postdate fraud labelling --
while preserving class balance (a global timestamp cutoff collapses the fraud
class, since fraud labels are concentrated in later periods; that observation is
itself reported as a leakage caveat). If PageRank/k-core on early-only history
still separates fraud from benign, the topology signal is not merely a post-hoc
consequence of being a flagged hub.

Output: data/graph_topology_features_timesplit.csv
Run:  python3 src/build_graph_features_timesplit.py
"""
from __future__ import annotations
import csv, io, os, sys, zipfile
csv.field_size_limit(10**9)
import pandas as pd
import igraph as ig

HERE=os.path.dirname(__file__); REPO=os.path.join(HERE,"..")
LABELS=os.path.join(REPO,"data","transaction_dataset.csv")
OUT=os.path.join(REPO,"data","graph_topology_features_timesplit.csv")
ZIP_CANDS=[os.path.join(REPO,"..","Ethereum-Fraud-Detection-main.zip"),
           os.path.join(REPO,"Ethereum-Fraud-Detection-main.zip")]
ROW_CAP=3000
EARLY_FRAC=0.7  # per-account: use only earliest 70% of an account's transactions

def find_zip():
    for p in ZIP_CANDS:
        if os.path.exists(p): return p
    raise SystemExit("raw zip not found")

def main():
    labels=pd.read_csv(LABELS,usecols=["Address","FLAG"])
    labels["addr"]=labels["Address"].str.strip().str.lower()
    lab_flag=dict(zip(labels["addr"],labels["FLAG"])); labset=set(lab_flag)
    z=zipfile.ZipFile(find_zip())
    norm=[n for n in z.namelist() if "Transaction_data/Normal_txn" in n and n.endswith(".csv")]
    addr_of=lambda n: n.split("/")[-1].replace("_normal_txn.csv","").lower()
    files=[n for n in norm if addr_of(n) in labset]
    print(f"per-account early-{EARLY_FRAC:.0%} time-split graph from {len(files)} files", flush=True)
    node_id={}; edges=[]; kept=0; dropped=0
    def nid(a):
        i=node_id.get(a)
        if i is None: i=len(node_id); node_id[a]=i
        return i
    for n in files:
        r=csv.reader(io.TextIOWrapper(io.BytesIO(z.read(n)),encoding="utf-8")); h=next(r,None)
        if not h: continue
        try: fi=h.index("from"); ti=h.index("to"); si=h.index("timeStamp")
        except ValueError: continue
        recs=[]
        for i,row in enumerate(r):
            if i>=ROW_CAP: break
            if len(row)<=max(fi,ti,si): continue
            try: t=int(row[si])
            except: continue
            f=(row[fi] or "").strip().lower(); to=(row[ti] or "").strip().lower()
            if f and to: recs.append((t,f,to))
        if not recs: continue
        recs.sort(key=lambda x:x[0])
        keep_n=max(1,int(len(recs)*EARLY_FRAC))
        for t,f,to in recs[:keep_n]: edges.append((nid(f),nid(to))); kept+=1
        dropped+=len(recs)-keep_n
    print(f"kept {kept} edges, dropped {dropped} post-cutoff; nodes={len(node_id)}", flush=True)
    g=ig.Graph(n=len(node_id),edges=edges,directed=True)
    pr=g.pagerank(directed=True); indeg=g.degree(mode="in"); outdeg=g.degree(mode="out")
    gu=g.as_undirected(mode="collapse"); clus=gu.transitivity_local_undirected(mode="zero"); kc=gu.coreness()
    rows=[]
    for a,i in node_id.items():
        if a in lab_flag:
            rows.append({"Address":a,"FLAG":int(lab_flag[a]),"g_pagerank":pr[i],
                         "g_in_degree":indeg[i],"g_out_degree":outdeg[i],
                         "g_total_degree":indeg[i]+outdeg[i],"g_clustering":clus[i],"g_kcore":kc[i]})
    out=pd.DataFrame(rows); out.to_csv(OUT,index=False)
    print(f"wrote {OUT}: {len(out)} labelled nodes with pre-cutoff edges, fraud rate {out['FLAG'].mean():.3f}")

if __name__=="__main__":
    raise SystemExit(main())
