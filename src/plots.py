"""Figures for the (extended) feature-class study. Tags REAL/SYNTHETIC from run_meta.
  fig1_feature_class_f1.png    F1 by feature class x model
  fig2_importance_share.png    permutation-importance share by class
  fig3_top_features.png        top individual features (permutation importance)
"""
import csv, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES=os.path.join(os.path.dirname(__file__),"..","results"); FIG=os.path.join(RES,"figures")
CLASS_COLOR={"graph_topology":"#2b8cbe","degree_graph":"#a6bddb","transaction":"#e6550d",
             "temporal":"#fdae6b","other":"#999999"}

def rows(n):
    with open(os.path.join(RES,n)) as f: return list(csv.DictReader(f))

def tag():
    try: src=json.load(open(os.path.join(RES,"run_meta.json")))["data_source"]
    except Exception: src="?"
    return "[REAL DATA]" if src=="real" else "[SYNTHETIC — illustrative]"

def fig1(t):
    r=rows("feature_class_results.csv")
    sets=["degree_graph","graph_topology","transaction","temporal","all"]
    models=["LogReg","RandomForest","HistGBDT"]
    fig,ax=plt.subplots(figsize=(10,5)); w=0.8/len(models); x=range(len(sets))
    for mi,m in enumerate(models):
        vals=[next((float(z["f1_mean"]) for z in r if z["feature_set"]==s and z["model"]==m),0) for s in sets]
        errs=[next((float(z["f1_std"]) for z in r if z["feature_set"]==s and z["model"]==m),0) for s in sets]
        ax.bar([i+mi*w for i in x],vals,width=w,yerr=errs,capsize=2,label=m)
    ax.set_xticks([i+w for i in x]); ax.set_xticklabels(sets,rotation=12)
    ax.set_ylabel("F1 (fraud class, 5-fold CV mean ± SD)"); ax.set_ylim(0,1.0)
    ax.set_title("Figure 1. Fraud-detection F1 by feature class and model  "+t); ax.legend(fontsize=9)
    fig.tight_layout(); o=os.path.join(FIG,"fig1_feature_class_f1.png"); fig.savefig(o,dpi=150); plt.close(fig); return o

def fig2(t):
    r=rows("feature_importance.csv")
    r=sorted(r,key=lambda z:-float(z["importance_share"]))
    labels=[z["feature_class"] for z in r]; vals=[float(z["importance_share"])*100 for z in r]
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    b=ax.bar(labels,vals,color=[CLASS_COLOR.get(l,"#999") for l in labels])
    ax.set_ylabel("Permutation-importance share (%)")
    ax.set_title("Figure 2. Feature-class share of permutation importance  "+t)
    for bar,v in zip(b,vals): ax.text(bar.get_x()+bar.get_width()/2,v,"{:.1f}%".format(v),ha="center",va="bottom",fontsize=9)
    ax.tick_params(axis="x",rotation=10)
    fig.tight_layout(); o=os.path.join(FIG,"fig2_importance_share.png"); fig.savefig(o,dpi=150); plt.close(fig); return o

def fig3(t):
    r=rows("top_features.csv")[:12][::-1]
    names=[z["feature"].strip()[:34] for z in r]; vals=[float(z["perm_importance"]) for z in r]
    cols=[CLASS_COLOR.get(z["feature_class"],"#999") for z in r]
    fig,ax=plt.subplots(figsize=(9,5.5)); ax.barh(names,vals,color=cols)
    ax.set_xlabel("Permutation importance (drop in average precision)")
    ax.set_title("Figure 3. Top 12 features driving fraud detection  "+t)
    from matplotlib.patches import Patch
    leg=[Patch(color=CLASS_COLOR[c],label=c) for c in ["graph_topology","degree_graph","transaction","temporal"]]
    ax.legend(handles=leg,fontsize=8,loc="lower right")
    fig.tight_layout(); o=os.path.join(FIG,"fig3_top_features.png"); fig.savefig(o,dpi=150); plt.close(fig); return o

def main():
    os.makedirs(FIG,exist_ok=True); t=tag()
    for o in (fig1(t),fig2(t),fig3(t)): print("Wrote",o)

if __name__=="__main__": main()
