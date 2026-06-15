"""Investigate WHY PageRank dominates: distribution by class, and whether it is a
proxy for transaction volume / degree (Spearman correlations)."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import data_loader as dl
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO=os.path.join(os.path.dirname(__file__),".."); RES=os.path.join(REPO,"results")
GRAPH_COLS=["g_pagerank","g_in_degree","g_out_degree","g_total_degree","g_clustering","g_kcore"]
df,_=dl.load(); df["addr"]=df["Address"].astype(str).str.strip().str.lower()
gt=pd.read_csv(os.path.join(REPO,"data","graph_topology_features.csv"))
gt["addr"]=gt["Address"].astype(str).str.strip().str.lower()
m=df.merge(gt[["addr"]+GRAPH_COLS],on="addr",how="inner")
y=pd.to_numeric(m["FLAG"]).astype(int).values
pr=pd.to_numeric(m["g_pagerank"],errors="coerce").fillna(0).values

# distribution by class
fr=pr[y==1]; be=pr[y==0]
u,pu=stats.mannwhitneyu(fr,be,alternative="two-sided")
out={"pagerank_fraud_median":float(np.median(fr)),"pagerank_benign_median":float(np.median(be)),
     "pagerank_fraud_mean":float(np.mean(fr)),"pagerank_benign_mean":float(np.mean(be)),
     "fraud_higher": bool(np.median(fr)>np.median(be)),
     "mannwhitney_u":float(u),"mannwhitney_p":float(pu)}

# is PageRank a proxy for volume / degree? Spearman vs candidate features
def col(name):
    c=[x for x in m.columns if x.strip()==name]
    return pd.to_numeric(m[c[0]],errors="coerce").fillna(0).values if c else None
cands={"total_transactions":"total transactions (including tnx to create contract",
       "total_ether_received":"total ether received","total_ether_sent":"total Ether sent",
       "received_tnx":"Received Tnx","sent_tnx":"Sent tnx",
       "g_total_degree":"g_total_degree","g_in_degree":"g_in_degree"}
corr={}
for k,name in cands.items():
    v=col(name) if not name.startswith("g_") else pd.to_numeric(m[name],errors="coerce").fillna(0).values
    if v is not None:
        rho,_=stats.spearmanr(pr,v); corr[k]=round(float(rho),3)
out["spearman_pagerank_vs"]=corr
json.dump(out, open(os.path.join(RES,"pagerank_analysis.json"),"w"), indent=2)
print(json.dumps(out, indent=2))

# figure: PageRank distribution by class (log)
fig,ax=plt.subplots(figsize=(7,4.5))
bins=np.logspace(np.log10(max(pr.min(),1e-9)), np.log10(pr.max()+1e-9), 40)
ax.hist(be,bins=bins,alpha=0.6,label="benign",color="#74a9cf",density=True)
ax.hist(fr,bins=bins,alpha=0.6,label="fraud",color="#e6550d",density=True)
ax.set_xscale("log"); ax.set_xlabel("PageRank (log)"); ax.set_ylabel("density")
ax.set_title("Figure 4. PageRank distribution by class  [REAL DATA]"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(RES,"figures","fig4_pagerank.png"),dpi=150); plt.close(fig)
print("wrote fig4_pagerank.png")
