#!/usr/bin/env python3
from __future__ import annotations
import argparse,base64,csv,html,json,math
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
FAMS=["add32f","mul32f","fma32f","rot32"]
LABEL={"add32f":"FP32 addition","mul32f":"FP32 multiplication","fma32f":"FP32 FMA","rot32":"UInt32 rotation"}
COL=dict(zip(FAMS,["#2878b5","#d95f02","#2a9d55","#88419d","#d73027"]))
LABEL_DY={"add32f":0,"mul32f":-13,"fma32f":13,"rot32":0}
BASES=["raw_fp32","fp32_to_fp64","raw_fp64"]
def q(v): return np.quantile(v,[.25,.5,.75]).tolist()
def plot(rows,kernel,path):
 groups=defaultdict(list)
 for r in rows:
  if r["kernel"]==kernel: groups[(r["stage"],r["family"],int(r["k"]))].append(float(r["ms"]))
 base_stage="rerun" if ("rerun","u32_base",0) in groups else "initial"
 fig,ax=plt.subplots(figsize=(10.5,6.2));base0=q(groups[(base_stage,"u32_base",0)])[1]
 for fam in FAMS:
  points={k:q(v) for (stage,f,k),v in groups.items() if f==fam and stage in (base_stage,"extension")};xs=[0]+sorted(points); vals=[q(groups[(base_stage,"u32_base",0)])]+[points[x] for x in xs[1:]];ys=[v[1] for v in vals]
  ax.plot(xs,ys,"o-",lw=2,color=COL[fam]);ax.fill_between(xs,[v[0] for v in vals],[v[2] for v in vals],color=COL[fam],alpha=.12)
  ax.annotate(LABEL[fam],(xs[-1],ys[-1]),xytext=(18,LABEL_DY[fam]),textcoords="offset points",va="center",color=COL[fam],arrowprops=dict(arrowstyle="-",linestyle=":",color=COL[fam]))
 for b,style in zip(BASES,["--","-.",":"]):
  qq=q(groups[(base_stage,b,0)]);ax.axhspan(qq[0],qq[2],alpha=.06,color="black");ax.axhline(qq[1],ls=style,color="#333",lw=1.3,label=b)
 ax.set(xlabel="Retained target-family instructions per decoded value",ylabel="Total kernel time (ms)",title=f"{kernel.upper()} instruction-cost sweep")
 ax.grid(alpha=.2);ax.legend(loc="upper left",frameon=False);ax.text(.01,.01,f"X=0 shared UInt32-to-FP32-to-FP64 anchor: {base0:.3f} ms",transform=ax.transAxes,fontsize=9);fig.tight_layout(rect=(0,0,.80,1));fig.savefig(path.with_suffix(".png"),dpi=180);fig.savefig(path.with_suffix(".svg"));plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument("--samples",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--manifest",required=True);p.add_argument("--audit",required=True);a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 with open(a.samples,newline="") as f: rows=list(csv.DictReader(f))
 if not rows or any(r["mode"]!="full" for r in rows): raise ValueError("report accepts full rows only")
 initial=[r for r in rows if r["stage"]=="initial"]
 if len(initial)!=3600: raise ValueError(f"expected 3600 initial rows, got {len(initial)}")
 keys=defaultdict(list)
 for r in rows:
  ms=float(r["ms"])
  if not math.isfinite(ms) or ms<=0 or r["valid"]!="1":raise ValueError("nonpositive, nonfinite, or invalid timing row")
  keys[(r["stage"],r["kernel"],r["family"],int(r["k"]))].append(ms)
 initial_keys=[k for k in keys if k[0]=="initial"]
 if len(initial_keys)!=72 or any(len(v)!=50 for v in keys.values()):raise ValueError("group inventory/count mismatch")
 expected={("initial",kernel,fam,k) for kernel in ("dot","gemv") for fam in FAMS for k in (1,2,4,8,12,16,24,32)}|{("initial",kernel,b,0) for kernel in ("dot","gemv") for b in ("raw_fp32","fp32_to_fp64","raw_fp64","u32_base")}
 if set(initial_keys)!=expected:raise ValueError("exact initial case inventory mismatch")
 rerun_keys={k for k in keys if k[0]=="rerun"}
 expected_rerun={("rerun",)+k[1:] for k in expected}
 if rerun_keys and rerun_keys!=expected_rerun:raise ValueError("incomplete rerun inventory")
 stage_round=defaultdict(list)
 for r in rows:stage_round[(r["stage"],int(r["round"]))].append(int(r["order"]))
 if any(sorted(v)!=list(range(len(v))) for v in stage_round.values()):raise ValueError("duplicate or missing execution-order index")
 if any({int(r["round"]) for r in rows if (r["stage"],r["kernel"],r["family"],int(r["k"]))==key}!=set(range(50)) for key in keys):raise ValueError("round inventory mismatch")
 for key in [k for k in keys if k[0]=="extension"]:
  if key[3] not in (0,48,64):raise ValueError("bad extension K")
 extension_keys={k for k in keys if k[0]=="extension"}
 for kernel in ("dot","gemv"):
  present={k for k in extension_keys if k[1]==kernel};official="rerun" if rerun_keys else "initial";raw=np.median(keys[(official,kernel,"raw_fp64",0)])
  eligible={fam for fam in FAMS if np.median(keys[(official,kernel,fam,32)])<raw}
  wanted={("extension",kernel,b,0) for b in ("raw_fp32","fp32_to_fp64","raw_fp64","u32_base")}|{("extension",kernel,fam,k) for fam in eligible for k in (48,64)} if eligible else set()
  if present!=wanted:raise ValueError(f"extension inventory mismatch for {kernel}")
 audit=json.loads(Path(a.audit).read_text());passed={(x["kernel"],x["family"],int(x["k"])) for x in audit if x["status"]=="pass"}
 plotted={(r["kernel"],r["family"],int(r["k"])) for r in rows if r["family"] in FAMS}
 if not plotted<=passed:raise ValueError(f"plotted cases lack passing assembly audit: {sorted(plotted-passed)}")
 summary=[]
 for key,v in sorted(keys.items()):
  a25,med,a75=q(v);stage,kernel,fam,k=key;base_stage=stage if (stage,kernel,"fp32_to_fp64",0) in keys else "initial";fp=q(keys[(base_stage,kernel,"fp32_to_fp64",0)])[1];raw=q(keys[(base_stage,kernel,"raw_fp64",0)])[1];summary.append(dict(stage=stage,kernel=kernel,family=fam,k=k,n=len(v),q1_ms=a25,median_ms=med,q3_ms=a75,ratio_fp32_to_fp64=med/fp,ratio_raw_fp64=med/raw))
 with (out/"timing_summary.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
 plot(rows,"dot",out/"dot_instruction_cost");plot(rows,"gemv",out/"gemv_instruction_cost")
 images=[plt.imread(out/"dot_instruction_cost.png"),plt.imread(out/"gemv_instruction_cost.png")]
 fig,axs=plt.subplots(2,1,figsize=(12,13));
 for ax,image in zip(axs,images):ax.imshow(image);ax.axis("off")
 fig.tight_layout();fig.savefig(out/"combined_instruction_cost.png",dpi=160);plt.close(fig)
 findings=[]
 for kernel in ("dot","gemv"):
  official="rerun" if any(x["stage"]=="rerun" and x["kernel"]==kernel for x in summary) else "initial"
  fp=next(x for x in summary if x["stage"]==official and x["kernel"]==kernel and x["family"]=="fp32_to_fp64")["median_ms"]
  raw=next(x for x in summary if x["stage"]==official and x["kernel"]==kernel and x["family"]=="raw_fp64")["median_ms"]
  for fam in FAMS:
   pts=[x for x in summary if x["kernel"]==kernel and x["family"]==fam and x["stage"] in (official,"extension")];within=[x["k"] for x in pts if x["ratio_fp32_to_fp64"]<=1.05];cross=[x["k"] for x in pts if x["ratio_raw_fp64"]>=1.0];findings.append(f"{kernel} {LABEL[fam]}: largest tested K within +5% of stage-matched FP32-to-FP64 = {max(within) if within else 'none'}; first measured K at/above stage-matched raw FP64 = {min(cross) if cross else 'not reached through '+str(max(x['k'] for x in pts))}.")
 drift=[]
 for stage in sorted({r["stage"] for r in rows if r["stage"] in ("initial","rerun")}):
  for kernel in ("dot","gemv"):
   for base in ("raw_fp32","fp32_to_fp64","raw_fp64","u32_base"):
    vals=[float(r["ms"]) for r in sorted(rows,key=lambda x:int(x["round"])) if r["stage"]==stage and r["kernel"]==kernel and r["family"]==base];early=np.median(vals[:25]);late=np.median(vals[25:]);drift.append(f"{stage} {kernel} {base}: {abs(late-early)/early:.2%} early/late median drift")
 manifest=Path(a.manifest).read_text() if a.manifest else "not supplied"
 def datauri(p):return "data:image/png;base64,"+base64.b64encode(p.read_bytes()).decode()
 body=f"""<!doctype html><meta charset=utf-8><title>Instruction cost sweep</title><style>body{{font:16px system-ui;max-width:1150px;margin:40px auto;color:#17202a}}img{{width:100%;height:auto}}code,pre{{background:#f4f5f6;padding:8px;white-space:pre-wrap}}li{{margin:.35em}}</style><h1>Dependent instruction cost inside DOT and GEMV</h1><p>H200 measurements with scalar x1 access, 32-bit encoded storage and FP64 accumulation. The shared X=0 decoder rounds UInt32 to FP32 and widens to FP64, so it does not preserve every UInt32 exactly. Floating curves execute FP32 operations before widening. X counts retained target-family SASS instructions per decoded operand.</p><h2>DOT</h2><img src='{datauri(out/'dot_instruction_cost.png')}'><h2>GEMV</h2><img src='{datauri(out/'gemv_instruction_cost.png')}'><h2>Threshold observations</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in findings)}</ul><h2>Baseline drift</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in drift)}</ul><p>This is a synthetic retained-instruction test, not a universal instruction-cost model.</p><h2>Run manifest</h2><pre>{html.escape(manifest)}</pre>"""
 (out/"report.html").write_text(body)
 print(f"analysis passed: {len(rows)} rows, {len(keys)} groups")
if __name__=="__main__":main()
