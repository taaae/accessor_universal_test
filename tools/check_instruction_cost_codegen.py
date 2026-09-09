#!/usr/bin/env python3
"""Fail-closed SASS audit for actual timed instruction-cost kernels."""
from __future__ import annotations
import argparse,csv,json,re
from collections import Counter
from pathlib import Path
FUN=re.compile(r"(?:Function\s*:\s*|\.section\s+\.text\.)(.+?)\s*$")
INS=re.compile(r"/\*([0-9a-fA-F]+)\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)(?:\s+([^;]*))?;")
REG=re.compile(r"\b(?:U?R(?:Z|\d+))\b")
CASE=re.compile(r"(dot|gemv)_timed_kernel.*family(?:IL)?([0-4]).*Li(0|1|2|4|8|12|16|24|32|48|64)")
FAMS=["add32","xor32","rot32","mul32","fma64"]
def parse(text):
 out={};cur=None
 for line in text.splitlines():
  m=FUN.search(line)
  if m: cur=m.group(1).strip();out.setdefault(cur,[]);continue
  m=INS.search(line)
  if m and cur: out[cur].append((int(m.group(1),16),m.group(2),m.group(3) or ""))
 return out
def target(op,fam):
 if fam=="add32": return op.startswith(("IADD3","UIADD3")) or op=="IMAD.IADD"
 if fam=="xor32": return op.startswith("LOP3")
 if fam=="rot32": return op.startswith("SHF")
 if fam=="mul32": return op=="IMAD" or op.startswith("XMAD")
 return op.startswith("DFMA") or op.startswith("FFMA.D2")
def audit_symbol(symbol,seq,kernel,fam,k):
 hist=Counter(x[1] for x in seq); reasons=[]
 if any(x[1].startswith(("LDL","STL","CALL")) for x in seq): reasons.append("spill_or_call")
 loads=[]
 for pos,x in enumerate(seq):
  regs=REG.findall(x[2])
  if x[1].startswith("LDG") and regs: loads.append((pos,regs[0]))
 chains=[]
 for start,reg in loads:
  length=0;current=reg;converted=False
  for _,op,args in seq[start+1:]:
   regs=REG.findall(args)
   if not regs: continue
   dest,src=regs[0],regs[1:]
   if current in src and target(op,fam): current=dest;length+=1
   elif current in src and (op.startswith("I2F") or op.startswith("F2F")): current=dest;converted=True
   elif current in src and op.startswith(("MOV","PRMT")): current=dest
  if length==k and converted: chains.append(length)
 if len(chains)<2: reasons.append(f"need_two_exact_dependent_chains_found_{chains}")
 total=sum(target(op,fam) for _,op,_ in seq)
 if total < (2*k+1 if fam=="fma64" else 2*k): reasons.append(f"too_few_target_ops_{total}")
 return {"symbol":symbol,"kernel":kernel,"family":fam,"k":k,"chain_a":chains[0] if chains else -1,"chain_b":chains[1] if len(chains)>1 else -1,"target_opcode_count":total,"opcode_histogram":dict(hist),"status":"pass" if not reasons else "fail","reason":";".join(reasons),"local_bytes":0,"shared_bytes":"unknown","setup_extra":"runtime scalar operands"}
def run(text):
 rows=[]
 for sym,seq in parse(text).items():
  m=CASE.search(sym)
  if m: rows.append(audit_symbol(sym,seq,m.group(1),FAMS[int(m.group(2))],int(m.group(3))))
 expected={(q,f,k) for q in ("dot","gemv") for f in FAMS for k in (0,1,2,4,8,12,16,24,32,48,64)}
 got={(r["kernel"],r["family"],r["k"]) for r in rows}
 if got!=expected: raise ValueError(f"specialization inventory mismatch missing={sorted(expected-got)[:8]} extra={sorted(got-expected)[:8]}")
 bad=[r for r in rows if r["status"]!="pass"]
 if bad: raise ValueError(f"assembly audit failed: {bad[:3]}")
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument("--sass",required=True);p.add_argument("--output-dir",required=True);a=p.parse_args();rows=run(Path(a.sass).read_text(errors="replace"));out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);(out/"assembly_audit.json").write_text(json.dumps(rows,indent=2));
 with (out/"assembly_audit.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(f"assembly audit passed: {len(rows)} timed specializations")
if __name__=="__main__":main()
