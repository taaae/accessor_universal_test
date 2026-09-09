#!/usr/bin/env python3
from __future__ import annotations
import argparse,base64,csv,hashlib,html,json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from validate_block_scale_results import validate
COL={(16,"fp32"):"#1f77b4",(16,"fp64"):"#17becf",(32,"fp32"):"#d95f02",(32,"fp64"):"#ff9f1c",(128,"fp32"):"#2ca02c",(128,"fp64"):"#8bc34a"}
BASECOL={"raw_fp32":"#222222","fp32_to_fp64":"#666666","raw_fp64":"#999999"}
def quant(v):return np.quantile(v,[.25,.5,.75]).tolist()
def label(vid):
 if vid=="raw_fp32":return "raw FP32"
 if vid=="fp32_to_fp64":return "FP32 to FP64"
 if vid=="raw_fp64":return "raw FP64"
 p,b,s=vid.rsplit("_",2);return f"B={b[1:]}, {s.upper()}, {'per element' if p=='per_element' else 'deferred'}"
def spread_labels(ax,ends,min_px=24):
 ax.figure.canvas.draw();inv=ax.transData.inverted();ordered=sorted(ends,key=lambda x:ax.transData.transform((x[1],x[2]))[1]);ys=[];last=ax.bbox.y0+5
 for item in ordered:
  py=max(ax.transData.transform((item[1],item[2]))[1],last);ys.append(py);last=py+min_px
 if ys[-1]>ax.bbox.y1-5:
  shift=ys[-1]-(ax.bbox.y1-5);ys=[y-shift for y in ys]
 for i in range(len(ys)-2,-1,-1):ys[i]=min(ys[i],ys[i+1]-min_px)
 if ys[0]<ax.bbox.y0+5:
  shift=ax.bbox.y0+5-ys[0];ys=[y+shift for y in ys]
 for item,py in zip(ordered,ys):item[3]=inv.transform((ax.bbox.x1,py))[1]
def main():
 p=argparse.ArgumentParser();p.add_argument("--samples",required=True);p.add_argument("--manifest",required=True);p.add_argument("--audit",required=True);p.add_argument("--correctness",required=True);p.add_argument("--output-dir",required=True);a=p.parse_args();rows,manifest=validate(a.samples,a.manifest,"full");audit_bytes=Path(a.audit).read_bytes();audit=json.loads(audit_bytes)
 expected_audit={(p,b,s) for p in ("per_element","deferred") for b in (16,32,128) for s in ("fp32","fp64")}|{(x,0,"none") for x in ("raw_fp32","fp32_to_fp64","raw_fp64")}|{(x,0,"none") for x in ("reduce32","reduce64")}
 got_audit={(x["kind"],int(x["B"]),x["scale_type"]) for x in audit}
 if got_audit!=expected_audit or any(x["status"]!="pass" for x in audit):raise ValueError("assembly audit inventory failed")
 if hashlib.sha256(audit_bytes).hexdigest()!=manifest["audit_sha256"]:raise ValueError("audit digest does not match manifest")
 checks=Path(a.correctness).read_text();
 if "all_passed=1" not in checks or "validated_case_count=240" not in checks:raise ValueError("correctness inventory failed")
 groups=defaultdict(list)
 for r in rows:groups[(r["stage"],int(r["N"]),r["variant_id"])].append(float(r["time_ms"]))
 official={n:("rerun" if any(k[0]=="rerun" and k[1]==n for k in groups) else "initial") for n in manifest["sizes"]};summary=[]
 for n in manifest["sizes"]:
  for vid in sorted({r["variant_id"] for r in rows}):q1,med,q3=quant(groups[(official[n],n,vid)]);base=quant(groups[(official[n],n,"fp32_to_fp64")])[1];summary.append(dict(N=n,stage=official[n],variant_id=vid,n=50,q1_ms=q1,median_ms=med,q3_ms=q3,ratio_fp32_to_fp64=med/base))
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 with (out/"timing_summary.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
 fig,ax=plt.subplots(figsize=(15,10));ends=[]
 for vid in sorted({x["variant_id"] for x in summary}):
  pts=[x for x in summary if x["variant_id"]==vid];xs=[x["N"] for x in pts];med=[x["median_ms"] for x in pts];q1=[x["q1_ms"] for x in pts];q3=[x["q3_ms"] for x in pts]
  if vid in BASECOL:color=BASECOL[vid];ls={"raw_fp32":"-","fp32_to_fp64":"--","raw_fp64":":"}[vid];marker="s"
  else:pfx,b,s=vid.rsplit("_",2);color=COL[(int(b[1:]),"fp32" if s=="s32" else "fp64")];ls="-" if pfx=="per_element" else "--";marker="o" if pfx=="per_element" else "^"
  ax.plot(xs,med,ls=ls,marker=marker,lw=2,color=color);ax.fill_between(xs,q1,q3,color=color,alpha=.08);ends.append([vid,xs[-1],med[-1],None,color])
 ax.set_xscale("log",base=2);ax.set_yscale("log");ax.set_xticks(manifest["sizes"],[f"2^{int(np.log2(n))}" for n in manifest["sizes"]]);ax.set_xlabel("DOT length N (log2 spacing)");ax.set_ylabel("Two-kernel time (ms, log scale)");ax.set_title("Block-scale FP32 payload DOT on H200");ax.grid(True,which="both",alpha=.2);spread_labels(ax,ends)
 for vid,x,y,ly,color in ends:ax.annotate("",(x,y),xytext=(x*1.50,ly),textcoords="data",xycoords="data",arrowprops=dict(arrowstyle="-",linestyle=":",color=color),annotation_clip=False);ax.text(x*1.55,ly,label(vid),color=color,fontsize=9,va="center")
 ax.set_xlim(manifest["sizes"][0]/1.25,manifest["sizes"][-1]*3.8);fig.tight_layout();fig.savefig(out/"dot_block_scale_all.png",dpi=180);fig.savefig(out/"dot_block_scale_all.svg");plt.close(fig)
 def uri(path):return "data:image/png;base64,"+base64.b64encode(path.read_bytes()).decode()
 fastest=[];comparisons=[];drift=[]
 for n in manifest["sizes"]:
  pts=[x for x in summary if x["N"]==n];best=min(pts,key=lambda x:x["median_ms"]);fastest.append(f"N={n}: {label(best['variant_id'])} at {best['median_ms']:.6g} ms")
  for x in pts:comparisons.append(f"N={n}, {label(x['variant_id'])}: median {x['median_ms']:.6g} ms, IQR {x['q1_ms']:.6g} to {x['q3_ms']:.6g} ms, {x['ratio_fp32_to_fp64']:.3f}x FP32-to-FP64")
  for stage in ("initial","rerun"):
   if not any(r["stage"]==stage and int(r["N"])==n for r in rows):continue
   for vid in ("raw_fp32","fp32_to_fp64","raw_fp64"):
    vals=[float(r["time_ms"]) for r in sorted(rows,key=lambda z:int(z["round"])) if r["stage"]==stage and int(r["N"])==n and r["variant_id"]==vid];ratio=abs(float(np.median(vals[25:]))-float(np.median(vals[:25])))/float(np.median(vals[:25]));drift.append(f"N={n}, {stage}, {label(vid)}: {ratio:.2%}"+(" (still above 5%)" if stage=="rerun" and ratio>.05 else ""))
 body=f"""<!doctype html><meta charset=utf-8><title>Block-scale DOT</title><style>body{{font:16px system-ui;max-width:1450px;margin:32px auto;color:#17202a}}img{{width:100%;height:auto}}pre{{background:#f4f5f6;padding:12px;white-space:pre-wrap}}li{{margin:.3em}}</style><h1>One-level shared scales inside DOT</h1><p>The full inventory contains 75 initial cases and 3,750 samples: all 15 configurations at N=2^16, 2^20, 2^24, 2^26, and 2^28, each measured in 50 shuffled rounds. Points are medians and bands are interquartile ranges. Inputs repeat without explicit cache flushing.</p><img src='{uri(out/'dot_block_scale_all.png')}'><h2>What was measured</h2><p>Payloads are FP32. Both operands use independent scales. Per-element variants reconstruct each operand in FP64 before FP64 FMA. Deferred variants reduce payload products within each storage block, multiply the completed subtotal by the two scales, and then accumulate. Raw FP32 alone accumulates in FP32. Scale storage costs 32+scale_bits/B bits per value.</p><h2>Fastest measured configurations</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in fastest)}</ul><h2>Baseline drift</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in drift)}</ul><h2>All measured medians, quartiles, and baseline ratios</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in comparisons)}</ul><h2>Correctness evidence</h2><pre>{html.escape(checks)}</pre><h2>Run manifest</h2><pre>{html.escape(json.dumps(manifest,indent=2))}</pre><p>These timings compare whole DOT implementations. The deferred reduction changes summation order and is DOT-specific, so it does not establish a generic accessor speedup or an accuracy ranking.</p>"""
 (out/"report.html").write_text(body);(out/"screenshot.html").write_text(f"<!doctype html><meta charset=utf-8><title>Block-scale DOT</title><style>body{{font-family:system-ui;margin:24px;background:#fff}}h1,p{{max-width:1400px;margin-left:auto;margin-right:auto}}img{{display:block;width:min(1600px,100%);margin:auto}}</style><h1>Block-scale FP32 payload DOT on H200</h1><p>N = 2^16, 2^20, 2^24, 2^26, 2^28. All 12 scaled variants and three native baselines.</p><img src='{uri(out/'dot_block_scale_all.png')}'>")
 print(f"analysis passed: {len(rows)} rows, {len(summary)} official summaries")
if __name__=="__main__":main()
