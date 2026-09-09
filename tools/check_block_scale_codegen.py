#!/usr/bin/env python3
"""Fail-closed structural audit of actual block-scale timed-kernel SASS."""
from __future__ import annotations
import argparse,csv,json,re
from collections import Counter
from pathlib import Path
FUN=re.compile(r"(?:Function\s*:\s*|\.section\s+\.text\.)(.+?)\s*$")
INS=re.compile(r"/\*([0-9a-fA-F]+)\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)(?:\s+([^;]*))?;")
CASE=re.compile(r"(per_element|deferred)_kernelILi(16|32|128)E([fd])E")
BASE={"raw32_kernel":"raw_fp32","widened32_kernel":"fp32_to_fp64","raw64_kernel":"raw_fp64"}
REDUCE={"reduce32_kernel":"reduce32","reduce64_kernel":"reduce64"}
REG=re.compile(r"\b(?:U?R(?:Z|\d+))(?:\.reuse)?\b")
def parse(text):
 out={};cur=None
 for line in text.splitlines():
  m=FUN.search(line)
  if m:cur=m.group(1).strip();out.setdefault(cur,[]);continue
  m=INS.search(line)
  if m and cur:out[cur].append((int(m.group(1),16),m.group(2),m.group(3) or ""))
 return out
def resources(text):
 out={};cur=None
 for line in text.splitlines():
  m=re.search(r"Function properties for (\S+)",line)
  if m:cur=m.group(1);out.setdefault(cur,{})
  if cur:
   m=re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads",line)
   if m:out[cur].update(spill_stores=int(m.group(1)),spill_loads=int(m.group(2)))
   m=re.search(r"Used (\d+) registers.*?(\d+) bytes smem",line)
   if m:out[cur].update(registers=int(m.group(1)),shared_bytes=int(m.group(2)))
 return out
def lineage(seq):
 prov={};loads=[];dmuls=[];dfmas=[]
 for pos,(_,op,args) in enumerate(seq):
  regs=[x.replace(".reuse","") for x in REG.findall(args)]
  if not regs:continue
  dest,sources=regs[0],regs[1:]
  if op.startswith("LDG"):
   tag=f"load{len(loads)}";loads.append((pos,op,dest,args));prov[dest]={tag};continue
  source_sets=[set(prov.get(x,set())) for x in sources]
  if op.startswith(("F2F","DMUL","DFMA","DADD","SHFL")):
   merged=set().union(*source_sets);prov[dest]=merged
   if op.startswith("DMUL"):dmuls.append((pos,source_sets,merged,dest))
   if op.startswith("DFMA"):dfmas.append((pos,source_sets,merged,dest))
  else:
   prov.pop(dest,None)
 return loads,dmuls,dfmas
def row(sym,kind,b,scale,seq,res):
 ops=[x[1] for x in seq];hist=Counter(ops);reasons=[]
 if any(op.startswith(("LDL","STL","CALL")) for op in ops):reasons.append("spill_or_call")
 if any("DIV" in op for op in ops):reasons.append("division_opcode")
 if any(op.startswith(("LDG.128","LDG.E.128")) for op in ops):reasons.append("vectorized_load")
 loads=[op for op in ops if op.startswith("LDG")];loads64=[op for op in loads if ".64" in op]
 if not loads:reasons.append("missing_global_load")
 dmul=sum(op.startswith("DMUL") for op in ops);dfma=sum(op.startswith("DFMA") for op in ops);shfl=sum(op.startswith("SHFL") for op in ops)
 traced_loads,traced_dmuls,traced_dfmas=lineage(seq)
 if kind=="per_element" and (dmul!=2 or dfma!=1 or shfl!=0):reasons.append("generic_reconstruction_shape")
 if kind=="deferred" and (dmul!=1 or dfma!=2 or shfl!=(4 if b==16 else 5)):reasons.append("deferred_reduction_shape")
 if kind in ("per_element","deferred"):
  if len(loads)!=4 or len(loads64)!=(0 if scale=="fp32" else 2):reasons.append("scaled_load_width_or_count")
  expected_widens=4 if scale=="fp32" else 2
  if sum(op.startswith("F2F.F64.F32") for op in ops)!=expected_widens:reasons.append("widen_count")
 if kind=="per_element":
  products=[x for x in traced_dmuls if len(x[2])==2]
  if len(products)!=2 or products[0][2]&products[1][2] or len(products[0][2]|products[1][2])!=4:reasons.append("reconstruction_load_lineage")
  elif not any(len(x[1])>=2 and x[1][0]==products[0][2] and x[1][1]==products[1][2] for x in traced_dfmas):reasons.append("dot_fma_does_not_consume_reconstructed_operands")
 if kind=="deferred":
  payload=[x for x in traced_dfmas if len(x[1])>=2 and len(x[1][0])==1 and len(x[1][1])==1]
  scale_products=[x for x in traced_dmuls if len(x[2])==2]
  shfl_pos=[i for i,(_,op,_) in enumerate(seq) if op.startswith("SHFL")]
  if not payload or not scale_products or not shfl_pos or not(payload[0][0]<min(shfl_pos)<scale_products[-1][0]):reasons.append("deferred_operation_order_or_lineage")
  elif not any(x[0]>scale_products[-1][0] and scale_products[-1][2].issubset(x[2]) for x in traced_dfmas):reasons.append("deferred_final_fma_lineage")
 if kind=="raw_fp32" and (sum(op.startswith("FFMA") for op in ops)!=1 or len(loads)!=2 or loads64):reasons.append("raw_fp32_shape")
 if kind=="fp32_to_fp64" and (dfma!=1 or len(loads)!=2 or loads64 or sum(op.startswith("F2F.F64.F32") for op in ops)!=2):reasons.append("widened_fp32_shape")
 if kind=="raw_fp64" and (dfma!=1 or len(loads)!=2 or len(loads64)!=2):reasons.append("raw_fp64_shape")
 if kind=="reduce32" and (len(loads)!=1 or not any(op.startswith("FADD") for op in ops)):reasons.append("reduce32_shape")
 if kind=="reduce64" and (len(loads)!=1 or not any(op.startswith("DADD") for op in ops)):reasons.append("reduce64_shape")
 for k in ("registers","spill_loads","spill_stores","shared_bytes"):
  if k not in res:reasons.append("missing_resource_"+k)
 if res.get("spill_loads",0) or res.get("spill_stores",0):reasons.append("ptxas_spills")
 return dict(symbol=sym,kind=kind,B=b,scale_type=scale,dmul=dmul,dfma=dfma,shuffle=shfl,opcode_histogram=dict(hist),registers=res.get("registers"),shared_bytes=res.get("shared_bytes"),spill_loads=res.get("spill_loads"),spill_stores=res.get("spill_stores"),status="pass" if not reasons else "fail",reason=";".join(reasons))
def run(sass,build):
 funcs=parse(sass);rr=resources(build);rows=[]
 for sym,seq in funcs.items():
  m=CASE.search(sym)
  if m:rows.append(row(sym,m.group(1),int(m.group(2)),"fp32" if m.group(3)=="f" else "fp64",seq,rr.get(sym,{})));continue
  for needle,name in BASE.items():
   if needle in sym:rows.append(row(sym,name,0,"none",seq,rr.get(sym,{})));break
  else:
   for needle,name in REDUCE.items():
    if needle in sym:rows.append(row(sym,name,0,"none",seq,rr.get(sym,{})));break
 expected={(p,b,s) for p in ("per_element","deferred") for b in (16,32,128) for s in ("fp32","fp64")}|{(x,0,"none") for x in BASE.values()}|{(x,0,"none") for x in REDUCE.values()};got={(x["kind"],x["B"],x["scale_type"]) for x in rows}
 if got!=expected:raise ValueError(f"timed-kernel inventory mismatch missing={expected-got} extra={got-expected}")
 bad=[x for x in rows if x["status"]!="pass"]
 if bad:raise ValueError(f"SASS audit failed: {bad}")
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument("--sass",required=True);p.add_argument("--build-log",required=True);p.add_argument("--output-dir",required=True);a=p.parse_args();rows=run(Path(a.sass).read_text(errors="replace"),Path(a.build_log).read_text(errors="replace"));out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);(out/"assembly_audit.json").write_text(json.dumps(rows,indent=2));
 with (out/"assembly_audit.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(f"SASS audit passed: {len(rows)} timed first-stage kernels")
if __name__=="__main__":main()
