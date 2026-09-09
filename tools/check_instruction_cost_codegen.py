#!/usr/bin/env python3
"""Fail-closed dataflow audit of actual timed-kernel SASS."""
from __future__ import annotations
import argparse,csv,json,re
from collections import Counter
from pathlib import Path
FUN=re.compile(r"(?:Function\s*:\s*|\.section\s+\.text\.)(.+?)\s*$")
INS=re.compile(r"/\*([0-9a-fA-F]+)\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)(?:\s+([^;]*))?;")
REG=re.compile(r"\b(?:U?R(?:Z|\d+))(?:\.reuse)?\b"); ADDR=re.compile(r"(?:0x)?([0-9a-fA-F]+)\s*$")
CASE=re.compile(r"(dot|gemv)_timed_kernel.*familyE([0-3])E.*Li(0|1|2|4|8|12|16|24|32|48|64)E")
FAMS=["add32f","mul32f","fma32f","rot32"]
def parse(text):
 out={};cur=None
 for line in text.splitlines():
  m=FUN.search(line)
  if m:cur=m.group(1).strip();out.setdefault(cur,[]);continue
  m=INS.search(line)
  if m and cur:out[cur].append((int(m.group(1),16),m.group(2),m.group(3) or ""))
 return out
def hot_loop(seq):
 positions={x[0]:i for i,x in enumerate(seq)};found=[]
 for end,x in enumerate(seq):
  if not x[1].startswith(("BRA","JMP")):continue
  m=ADDR.search(x[2]); target=int(m.group(1),16) if m else -1
  if target in positions and positions[target]<end:
   region=seq[positions[target]:end+1]
   if sum(op.startswith("LDG") for _,op,_ in region)==2 and any(op.startswith("DFMA") for _,op,_ in region):found.append(region)
 if len(found)!=1:raise ValueError(f"expected one two-load hot loop, found {len(found)}")
 return found[0]
def is_target(op,fam):
 return (fam=="add32f" and op.startswith("FADD")) or (fam=="mul32f" and op.startswith("FMUL")) or (fam=="fma32f" and op.startswith("FFMA")) or (fam=="rot32" and op.startswith("SHF.L.W.U32"))
def audit_symbol(symbol,seq,kernel,fam,k,resource=None):
 resource=resource or {};reasons=[]
 try:loop=hot_loop(seq)
 except ValueError as e:return {"symbol":symbol,"kernel":kernel,"family":fam,"k":k,"status":"fail","reason":str(e)}
 if any(op.startswith(("LDL","STL","CALL")) for _,op,_ in loop):reasons.append("spill_or_call_in_hot_loop")
 loads=[]
 for pos,(_,op,args) in enumerate(loop):
  regs=REG.findall(args)
  if op.startswith("LDG") and regs:loads.append((pos,regs[0]))
 states=[]
 for load_pos,reg in loads:
  current=reg;count=0;c1=c2=False;alive=True
  for _,op,args in loop[load_pos+1:]:
   regs=REG.findall(args)
   if not regs:continue
   dest,sources=regs[0],regs[1:];uses=current in sources
   if uses and op.startswith("I2F.F32.U32"):current=dest;c1=True;continue
   if uses and is_target(op,fam):
    if fam=="rot32" and sources.count(current)<2:reasons.append("rotation_not_wrap_same_source")
    current=dest;count+=1;continue
   if uses and op.startswith("F2F.F64"):current=dest;c2=True;continue
   if dest==current and not uses:alive=False;break
  states.append((current,count,c1,c2,alive))
 if len(states)!=2:reasons.append("not_two_source_loads")
 else:
  for index,(_,count,c1,c2,alive) in enumerate(states):
   if count!=k:reasons.append(f"chain_{index}_length_{count}_expected_{k}")
   if not c1 or not c2:reasons.append(f"chain_{index}_missing_conversion")
   if not alive:reasons.append(f"chain_{index}_overwritten")
  acc=[x for x in loop if x[1].startswith("DFMA") and states[0][0] in REG.findall(x[2])[1:] and states[1][0] in REG.findall(x[2])[1:]]
  if len(acc)!=1:reasons.append(f"accumulation_consumption_count_{len(acc)}")
 total=sum(is_target(op,fam) for _,op,_ in loop)
 if total!=2*k:reasons.append(f"target_total_{total}_expected_{2*k}")
 if resource.get("spill_loads",0) or resource.get("spill_stores",0):reasons.append("ptxas_spills")
 return {"symbol":symbol,"kernel":kernel,"family":fam,"k":k,"chain_a":states[0][1] if states else -1,"chain_b":states[1][1] if len(states)>1 else -1,"target_opcode_count":total,"opcode_histogram":dict(Counter(op for _,op,_ in loop)),"registers":resource.get("registers"),"spill_loads":resource.get("spill_loads"),"spill_stores":resource.get("spill_stores"),"shared_bytes":resource.get("shared_bytes"),"status":"pass" if not reasons else "fail","reason":";".join(dict.fromkeys(reasons))}
def resources(text):
 out={};current=None
 for line in text.splitlines():
  m=re.search(r"Function properties for (\S+)",line)
  if m:current=m.group(1);out.setdefault(current,{})
  if current:
   m=re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads",line)
   if m:out[current].update(spill_stores=int(m.group(1)),spill_loads=int(m.group(2)))
   m=re.search(r"Used (\d+) registers.*?(\d+) bytes smem",line)
   if m:out[current].update(registers=int(m.group(1)),shared_bytes=int(m.group(2)))
 return out
def run(text,build=""):
 funcs=parse(text);res=resources(build);rows=[]
 for sym,seq in funcs.items():
  m=CASE.search(sym)
  if m:rows.append(audit_symbol(sym,seq,m.group(1),FAMS[int(m.group(2))],int(m.group(3)),res.get(sym,{})))
 expected={(q,f,k) for q in ("dot","gemv") for f in FAMS for k in (0,1,2,4,8,12,16,24,32,48,64)};got={(r["kernel"],r["family"],r["k"]) for r in rows}
 if got!=expected:raise ValueError(f"inventory mismatch missing={sorted(expected-got)[:8]} extra={sorted(got-expected)[:8]}")
 bad=[r for r in rows if r["status"]!="pass"]
 if bad:raise ValueError(f"assembly audit failed: {bad[:3]}")
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument("--sass",required=True);p.add_argument("--build-log",required=True);p.add_argument("--output-dir",required=True);a=p.parse_args();rows=run(Path(a.sass).read_text(errors="replace"),Path(a.build_log).read_text(errors="replace"));out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);(out/"assembly_audit.json").write_text(json.dumps(rows,indent=2));
 with (out/"assembly_audit.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(f"assembly audit passed: {len(rows)} timed specializations")
if __name__=="__main__":main()
