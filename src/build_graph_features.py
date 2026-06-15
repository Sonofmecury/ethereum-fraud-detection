"""
build_graph_features.py -- construct REAL graph-topology features from the raw
Ethereum transaction files (addresses the critique's point that degree-count
"graph" features are not true graph analytics).

Reads the raw per-address normal-transaction CSVs from the dataset zip, builds a
directed transaction graph (nodes = addresses, edges = from->to), and computes,
for each labelled address, genuine network-topology features:
    pagerank, in_degree, out_degree, total_degree,
    clustering (local transitivity), kcore (coreness)

(Betweenness centrality is intentionally omitted: it is O(VE) and infeasible at
this graph size in-budget; this is stated as a limitation.)

A per-file row cap bounds a few hyper-active addresses; the cap is recorded.

Output: data/graph_topology_features.csv  (Address, FLAG, <topology features>)

Run:  python3 src/build_graph_features.py
"""
from __future__ import annotations

import csv
import io
import os
import sys
import zipfile

import pandas as pd
import igraph as ig

csv.field_size_limit(10**9)

HERE = os.path.dirname(__file__)
REPO = os.path.join(HERE, "..")
LABELS = os.path.join(REPO, "data", "transaction_dataset.csv")
OUT = os.path.join(REPO, "data", "graph_topology_features.csv")
ZIP_CANDIDATES = [
    os.path.join(REPO, "..", "Ethereum-Fraud-Detection-main.zip"),
    os.path.join(REPO, "Ethereum-Fraud-Detection-main.zip"),
]
ROW_CAP = 3000  # max transactions read per address file


def find_zip():
    for p in ZIP_CANDIDATES:
        if os.path.exists(p):
            return p
    raise SystemExit("Raw dataset zip not found; place Ethereum-Fraud-Detection-main.zip "
                     "in the parent folder. (Graph features need the raw transactions.)")


def main():
    labels = pd.read_csv(LABELS, usecols=["Address", "FLAG"])
    labels["addr"] = labels["Address"].str.strip().str.lower()
    lab_flag = dict(zip(labels["addr"], labels["FLAG"]))
    labset = set(lab_flag)

    z = zipfile.ZipFile(find_zip())
    norm = [n for n in z.namelist()
            if "Transaction_data/Normal_txn" in n and n.endswith(".csv")]

    def addr_of(n):
        return n.split("/")[-1].replace("_normal_txn.csv", "").lower()

    files = [n for n in norm if addr_of(n) in labset]
    print(f"building graph from {len(files)} labelled-address files (cap {ROW_CAP}/file)...",
          flush=True)

    node_id = {}
    edges = []

    def nid(a):
        i = node_id.get(a)
        if i is None:
            i = len(node_id); node_id[a] = i
        return i

    for k, n in enumerate(files):
        raw = z.read(n)
        r = csv.reader(io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8"))
        hdr = next(r, None)
        if not hdr:
            continue
        try:
            fi = hdr.index("from"); ti = hdr.index("to")
        except ValueError:
            continue
        for i, row in enumerate(r):
            if i >= ROW_CAP:
                break
            if len(row) <= max(fi, ti):
                continue
            f = (row[fi] or "").strip().lower(); t = (row[ti] or "").strip().lower()
            if f and t:
                edges.append((nid(f), nid(t)))

    print(f"nodes={len(node_id):,} edges={len(edges):,}; computing topology...", flush=True)
    g = ig.Graph(n=len(node_id), edges=edges, directed=True)

    pr = g.pagerank(directed=True)
    indeg = g.degree(mode="in"); outdeg = g.degree(mode="out")
    gu = g.as_undirected(mode="collapse")
    clustering = gu.transitivity_local_undirected(mode="zero")
    kcore = gu.coreness()

    rows = []
    for a, i in node_id.items():
        if a in lab_flag:
            rows.append({"Address": a, "FLAG": int(lab_flag[a]),
                         "g_pagerank": pr[i], "g_in_degree": indeg[i],
                         "g_out_degree": outdeg[i], "g_total_degree": indeg[i] + outdeg[i],
                         "g_clustering": clustering[i], "g_kcore": kcore[i]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(out)} labelled nodes, {out['FLAG'].mean():.3f} fraud rate")
    print("(omitted: betweenness centrality -- O(VE), infeasible at this size)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
