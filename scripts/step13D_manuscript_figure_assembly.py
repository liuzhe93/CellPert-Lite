#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step13D: reproducible manuscript figure assembly for CellPert-Lite.
No training, no invented architecture, no cross-cell-line pooled estimates.
Dependencies: numpy, pandas, matplotlib.
"""
import argparse, hashlib, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

MODELS=["gated_direct","no_gate_direct","genept_only_direct","uce_only_direct"]
LABEL={"gated_direct":"Gated fusion","no_gate_direct":"No-gate fusion",
       "genept_only_direct":"GenePT-only","uce_only_direct":"UCE-only"}
COLOR=dict(zip(MODELS,["#2867A5","#27816C","#D59B28","#7653C5"]))
GROUP={"K562_train_seen":"Seen","K562_test_heldout":"Held-out"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "pdf.fonttype":42,"svg.fonttype":"none",
                     "savefig.facecolor":"white"})

def locate(root, override, paths):
    if override:
        q=Path(override)
        q=q if q.is_absolute() else root/q
        if not q.is_file(): raise FileNotFoundError(str(q))
        return q
    for name in paths:
        q=root/name
        if q.is_file(): return q
    return None

def load(root, override, paths, required=False):
    q=locate(root,override,paths)
    if q is None:
        if required: raise FileNotFoundError("Missing input; searched: "+str(paths))
        return None,None
    return pd.read_csv(q),q

def cols(df, fields, name):
    missing=set(fields)-set(df.columns)
    if missing: raise ValueError(f"{name}: missing columns {sorted(missing)}; got {list(df.columns)}")

def save(fig, name, out, pdf):
    fig.tight_layout(pad=1.8)
    for ext in ("png","svg","pdf"):
        fig.savefig(out/f"{name}.{ext}",dpi=240,bbox_inches="tight")
    pdf.savefig(fig,bbox_inches="tight")
    plt.close(fig)

def tidy(ax,ylabel=None):
    ax.grid(axis="y",alpha=.15)
    ax.set_axisbelow(True)
    if ylabel: ax.set_ylabel(ylabel)

def dot_models(ax,df,metric,err=None,title="",ylabel="Mean perturbation PCC"):
    x=np.arange(len(MODELS))
    for i,m in enumerate(MODELS):
        d=df[df.model==m]
        if len(d)!=1: continue
        y=float(d.iloc[0][metric])
        e=float(d.iloc[0][err]) if err and err in d and pd.notna(d.iloc[0][err]) else 0
        ax.errorbar(i,y,yerr=e,fmt="o",color=COLOR[m],capsize=3,ms=7)
    ax.set_xticks(x,[LABEL[m] for m in MODELS],rotation=20,ha="right")
    ax.set_title(title);tidy(ax,ylabel)

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--root",default=".")
    a.add_argument("--output",default=None)
    for key in ("k562","external","paired","filter","representative","crossline","enrichment"):
        a.add_argument("--"+key+"-file",default=None)
    args=a.parse_args()
    root=Path(args.root).resolve()
    out=Path(args.output).resolve() if args.output else root/"results/step13D_manuscript_figure_assembly"
    out.mkdir(parents=True,exist_ok=True)
    paths={
      "k562":["results/step13B_internal_external_validation/K562_input_copy.csv",
              "results/step10A_unseen_perturbation_transfer/metrics_summary.csv"],
      "external":["results/step13B_internal_external_validation/Table_external_seed_mean_sd.csv",
                  "results/step12E_metric_reconciliation/paper_summary.csv"],
      "paired":["results/step13A_core_results_integration/Table2_paired_model_comparisons.csv",
                "results/step12E_metric_reconciliation/paired_model_target_bootstrap.csv"],
      "filter":["results/step12D_response_matrix_quality_and_calibration/filter_sensitivity_summary.csv"],
      "representative":["results/step12C_representative_response_visualization/representative_prediction_metrics.csv"],
      "crossline":["results/step12C_representative_response_visualization/cross_line_observed_comparison.csv"],
      "enrichment":["results/step12B_functional_enrichment/enrichment_significant.csv",
                    "results/step12B_functional_enrichment_fixed/enrichment_significant.csv"]
    }
    data={};sources={};missing=[]
    for key, candidates in paths.items():
        data[key],sources[key]=load(root,getattr(args,key+"_file"),candidates,key in ("k562","external"))
        if sources[key] is None: missing.append(key)
    k=data["k562"];e=data["external"]
    cols(k,["dataset","model","threshold","perturbation_pcc_mean","global_pcc_mean"],"K562/RPE1")
    if "mean_target_pcc_mean" in e:
        cols(e,["dataset","model","group","mean_target_pcc_mean"],"external")
        e=e.rename(columns={"mean_target_pcc_mean":"pcc","mean_target_pcc_sd":"pcc_sd",
            "mean_seed_micro_relative_skill_mean":"micro_skill","mean_target_skill_mean":"macro_skill"})
    elif "mean_target_pcc" in e:
        cols(e,["dataset","model","group","mean_target_pcc"],"external")
        e=e.rename(columns={"mean_target_pcc":"pcc",
                             "mean_micro_relative_skill":"micro_skill",
                             "mean_macro_target_skill":"macro_skill"})
    else: raise ValueError("External table has no recognized mean-target PCC column")
    e=e[e.model.isin(MODELS)&e.dataset.isin(["HepG2","Jurkat"])].copy()
    k=k[k.model.isin(MODELS)].copy()
    if "pcc_sd" not in e: e["pcc_sd"]=np.nan
    for field in ("micro_skill","macro_skill"):
        if field not in e:e[field]=np.nan
    manifest=[]
    for key,path in sources.items():
        manifest.append({"input":key,"path":str(path) if path else "",
                         "status":"found" if path else "missing_optional",
                         "sha256":hashlib.sha256(path.read_bytes()).hexdigest() if path else ""})
    pd.DataFrame(manifest).to_csv(out/"source_manifest.csv",index=False)
    k.to_csv(out/"source_K562_RPE1.csv",index=False)
    e.to_csv(out/"source_HepG2_Jurkat.csv",index=False)
    audit={"missing_optional":missing,"figures":[],"notes":[
       "K562/RPE1 and HepG2/Jurkat values are descriptive, not pooled or directly exchangeable.",
       "RPE1 thresholds use different perturbation subsets.",
       "Seen vs held-out compares different target sets.",
       "Error bars from seed SD do not represent biological replication.",
       "Functional enrichment is exploratory and contains overlapping gene sets.",
       "No architecture diagram is inferred from summary CSVs."
    ]}
    with PdfPages(out/"Step13D_all_panels.pdf") as pdf:
        # Figure 1: evidence-based evaluation design, not model architecture.
        fig,ax=plt.subplots(figsize=(11,3.8));ax.axis("off")
        nodes=[("K562\ntraining / frozen weights",.12),("K562\nheld-out targets",.37),
               ("External transfer\nRPE1",.60),("External transfer\nHepG2 / Jurkat",.85)]
        for label,x in nodes:
            ax.text(x,.58,label,ha="center",va="center",fontsize=11,
                    bbox=dict(boxstyle="round,pad=.65",fc="#EDF3F8",ec="#7891A9"),
                    transform=ax.transAxes)
        for x1,x2 in ((.23,.27),(.48,.51),(.70,.73)):
            ax.annotate("",xy=(x2,.58),xytext=(x1,.58),xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->",lw=1.7))
        ax.text(.5,.14,"Evaluation schematic only; model architecture requires original network specification.",
                ha="center",color="#555",transform=ax.transAxes)
        ax.set_title("Figure 1 | Evaluation contexts (not a network architecture)",pad=18)
        save(fig,"Figure1_evaluation_design",out,pdf);audit["figures"].append("Figure1")
        # Figure 2: K562
        d=k[k.dataset=="K562_unseen"]
        if len(d):
            fig,axes=plt.subplots(1,2,figsize=(11,4.3))
            dot_models(axes[0],d,"perturbation_pcc_mean",title="A | Mean target PCC",ylabel="Mean perturbation PCC")
            dot_models(axes[1],d,"global_pcc_mean","global_pcc_sd" if "global_pcc_sd" in d else None,
                       title="B | Global PCC",ylabel="Global PCC")
            fig.suptitle("Figure 2 | K562 internal unseen-perturbation evaluation")
            save(fig,"Figure2_K562_internal",out,pdf);audit["figures"].append("Figure2")
        # Figure 3: RPE1
        d=k[k.dataset.astype(str).str.startswith("RPE1")]
        if len(d):
            fig,ax=plt.subplots(figsize=(8.7,4.6))
            for m in MODELS:
                sub=d[d.model==m].sort_values("threshold")
                ax.plot(sub.threshold,sub.perturbation_pcc_mean,"o-",label=LABEL[m],color=COLOR[m],lw=1.8)
            ax.set_xticks(sorted(d.threshold.unique()))
            ax.set_xlabel("RPE1 minimum batch-perturbation cell threshold")
            tidy(ax,"Mean perturbation PCC")
            ax.legend(frameon=False,ncol=2,fontsize=9)
            ax.set_title("Figure 3 | RPE1 external transfer (different target sets)")
            save(fig,"Figure3_RPE1_threshold_sensitivity",out,pdf);audit["figures"].append("Figure3")
        # Figure 4: external seen vs held-out, connected dots
        fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
        for ax,ds in zip(axes,["HepG2","Jurkat"]):
            for j,m in enumerate(MODELS):
                sub=e[(e.dataset==ds)&(e.model==m)]
                v=[]
                for g in ("K562_train_seen","K562_test_heldout"):
                    q=sub[sub.group==g]
                    v.append(float(q.iloc[0].pcc) if len(q) else np.nan)
                ax.plot([0,1],v,"o-",color=COLOR[m],lw=1.8,ms=6,label=LABEL[m])
            ax.set_xticks([0,1],["Seen","Held-out"]);ax.set_title(ds)
            tidy(ax,"Mean perturbation PCC" if ds=="HepG2" else None)
        axes[1].legend(loc="upper right",frameon=False,fontsize=9)
        fig.suptitle("Figure 4 | External generalization gap (non-identical target sets)")
        save(fig,"Figure4_external_seen_heldout",out,pdf);audit["figures"].append("Figure4")
        # Figure 5: model paired effects, show CI; no significance inference if absent
        paired=data["paired"]
        if paired is not None:
            cols(paired,["dataset","group","comparison","metric","mean_difference","ci95_low","ci95_high"],"paired")
            d=paired[(paired.metric=="pcc") &
              (paired.comparison.str.contains("gated_direct - genept_only_direct",regex=False))].copy()
            if len(d):
                d=d.sort_values(["dataset","group"])
                fig,ax=plt.subplots(figsize=(9,4.5))
                yy=np.arange(len(d))
                vals=d.mean_difference.to_numpy(float)
                lo=d.ci95_low.to_numpy(float);hi=d.ci95_high.to_numpy(float)
                ax.errorbar(vals,yy,xerr=np.vstack([vals-lo,hi-vals]),fmt="o",
                            color="#2867A5",capsize=3)
                ax.axvline(0,color="#444",ls="--",lw=1)
                ax.set_yticks(yy,[f"{r.dataset} | {r.group}" for r in d.itertuples()])
                ax.set_xlabel("Paired ΔPCC (Gated − GenePT), 95% bootstrap CI")
                ax.set_title("Figure 5 | Matched-target model comparison")
                ax.invert_yaxis();ax.grid(axis="x",alpha=.15)
                save(fig,"Figure5_paired_ablation",out,pdf);audit["figures"].append("Figure5")
        # Figure 6: response strength filtering (truth-dependent diagnostic)
        filt=data["filter"]
        if filt is not None:
            cols(filt,["dataset","model","group","filter","mean_pcc","median_amplitude_ratio"],"filter")
            d=filt[(filt.model=="gated_direct") &
                   (filt.group.astype(str).str.lower().isin(["heldout","k562_test_heldout"]))].copy()
            d=d[d["filter"].str.startswith("abs_observed_ge_")].copy()
            d["threshold"]=pd.to_numeric(d["filter"].str.replace("abs_observed_ge_","",regex=False),errors="coerce")
            d=d.dropna(subset=["threshold"])
            if len(d):
                fig,axes=plt.subplots(1,2,figsize=(11,4.5))
                for ds in ("HepG2","Jurkat"):
                    q=d[d.dataset==ds].sort_values("threshold")
                    axes[0].plot(q.threshold,q.mean_pcc,"o-",label=ds)
                    axes[1].plot(q.threshold,q.median_amplitude_ratio,"o-",label=ds)
                axes[0].set_ylabel("Mean perturbation PCC")
                axes[1].set_ylabel("Median predicted / observed RMS")
                for ax in axes:
                    ax.set_xlabel("Absolute observed-response threshold")
                    ax.legend(frameon=False);tidy(ax)
                fig.suptitle("Figure 6 | Truth-dependent signal filtering (diagnostic only)")
                save(fig,"Figure6_signal_strength_and_amplitude",out,pdf);audit["figures"].append("Figure6")
        # Figure 7: representative target comparisons
        rep=data["representative"]
        if rep is not None:
            cols(rep,["dataset","perturbation","mean_seed_pcc","observed_rms","predicted_rms"],"representative")
            fig,axes=plt.subplots(1,2,figsize=(12,5))
            piv=rep.pivot_table(index="perturbation",columns="dataset",values="mean_seed_pcc")
            piv=piv.reindex(sorted(piv.index))
            x=np.arange(len(piv));w=.37
            for j,ds in enumerate(["HepG2","Jurkat"]):
                if ds in piv:
                    axes[0].bar(x+(j-.5)*w,piv[ds],w,label=ds)
            axes[0].set_xticks(x,piv.index,rotation=35,ha="right")
            axes[0].axhline(0,color="#555",lw=.8);axes[0].legend(frameon=False)
            tidy(axes[0],"Mean seed PCC")
            for ds in ["HepG2","Jurkat"]:
                q=rep[rep.dataset==ds]
                axes[1].scatter(q.observed_rms,q.predicted_rms,label=ds,s=55,alpha=.85)
            maxv=max(rep.observed_rms.max(),rep.predicted_rms.max())*1.05
            axes[1].plot([0,maxv],[0,maxv],"--",color="#666",lw=1)
            axes[1].set(xlabel="Observed RMS",ylabel="Predicted RMS")
            axes[1].legend(frameon=False)
            fig.suptitle("Figure 7 | Selected perturbations: context and amplitude")
            save(fig,"Figure7_representative_perturbations",out,pdf);audit["figures"].append("Figure7")
        # Figure 8: exploratory enrichment; avoid presenting overlapping terms as independent
        enr=data["enrichment"]
        if enr is not None:
            cols(enr,["dataset","group","category","database","term","fdr_bh","overlap"],"enrichment")
            d=enr[(enr.fdr_bh>0)&(enr.fdr_bh<=.05)].copy()
            if len(d):
                # one most significant term per dataset/group/category, up to 10
                d=d.sort_values("fdr_bh").drop_duplicates(["dataset","group","category"]).head(10)
                d=d.iloc[::-1]
                fig,ax=plt.subplots(figsize=(11,max(4.5,.45*len(d)+1.8)))
                lab=[f"{r.dataset} | {r.group} | {str(r.term)[:58]}" for r in d.itertuples()]
                ax.barh(np.arange(len(d)),-np.log10(d.fdr_bh.clip(lower=1e-300)),color="#548B9A")
                ax.set_yticks(np.arange(len(d)),lab,fontsize=8)
                ax.set_xlabel("−log10(BH-adjusted p), exploratory enrichment")
                ax.set_title("Figure 8 | Selected enriched terms (overlapping gene sets)")
                ax.grid(axis="x",alpha=.15)
                save(fig,"Figure8_exploratory_enrichment",out,pdf);audit["figures"].append("Figure8")
    (out/"validation_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    lines=["# Step13D manuscript figure assembly","",
           "This is an evidence-based assembly, NOT a reconstruction of missing model architecture.",
           "Figures use source-specific summaries. K562/RPE1 versus HepG2/Jurkat are not pooled.",
           "RPE1 threshold panels compare non-identical target sets.",
           "Seen/held-out targets differ; no causal attribution is made.",
           "Step12D response filters use observed truth and are retrospective diagnostics.",
           "Bootstrap CI is target-level (seed-averaged), not biological replication.",
           "Enrichment terms overlap and should not be counted as independent pathways.","",
           "## Generated panels"]+[f"- {x}" for x in audit["figures"]]+[
           "","## Missing optional sources"]+[f"- {x}" for x in missing]+[
           "","## Before submission",
           "- Supply the actual model architecture and data processing schematic for Figure 1.",
           "- Confirm all group sizes and response-gene sets in Methods.",
           "- Check PDF/SVG typography and layout in a vector editor.",
           "- Select main versus supplementary panels after inspecting the complete manuscript."]
    (out/"README_figure_assembly.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"[DONE] {out} | {len(audit['figures'])} figures | missing optional: {missing}")
if __name__=="__main__":
    main()
