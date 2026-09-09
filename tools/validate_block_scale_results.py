#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
FULL_N=[65536,1048576,16777216,67108864,268435456]
SMOKE_N=[16384,1048576]
BLOCKS=[16,32,128]
SCALES=["fp32","fp64"]
PATHS=["per_element","deferred"]
BASE=["raw_fp32","fp32_to_fp64","raw_fp64"]
FIELDS="mode,stage,N,B,scale_type,path,variant_id,payload_type,arithmetic,grid,threads,round,order,time_ms,result,valid,job_id,node,seed_set_id,source_commit,binary_sha,sass_sha,audit_sha".split(",")
def variants():
 return {(f"{p}_b{b}_s{32 if s=='fp32' else 64}",b,s,p,"fp32","fp64") for p in PATHS for b in BLOCKS for s in SCALES}|{("raw_fp32",0,"none","baseline","fp32","fp32"),("fp32_to_fp64",0,"none","baseline","fp32","fp64"),("raw_fp64",0,"none","baseline","fp64","fp64")}
def fail(msg):raise ValueError(msg)
def validate(samples,manifest_path,mode=None):
 manifest=json.loads(Path(manifest_path).read_text());rows=list(csv.DictReader(open(samples,newline="")))
 if not rows or list(rows[0].keys())!=FIELDS:fail("CSV header mismatch")
 observed_mode={r["mode"] for r in rows};mode=mode or manifest.get("mode")
 if observed_mode!={mode} or mode not in ("full","smoke"):fail("mode mismatch")
 ns=FULL_N if mode=="full" else SMOKE_N;samples_per=50 if mode=="full" else 3;allowed=variants();groups=defaultdict(list);round_orders=defaultdict(list)
 required={"source_commit":"commit","binary_sha":"binary_sha256","sass_sha":"sass_sha256","audit_sha":"audit_sha256","job_id":"job_id","node":"node"}
 for r in rows:
  try:key=(r["variant_id"],int(r["B"]),r["scale_type"],r["path"],r["payload_type"],r["arithmetic"]);n=int(r["N"]);rd=int(r["round"]);order=int(r["order"]);ms=float(r["time_ms"]);value=float(r["result"])
  except Exception as e:fail(f"malformed row: {e}")
  if key not in allowed:fail(f"unknown or inconsistent variant {key}")
  if n not in ns or r["stage"] not in ("initial","rerun"):fail("bad N or stage")
  if r["grid"]!="512" or r["threads"]!="256" or r["seed_set_id"]!="block-scale-v1" or r["valid"]!="1":fail("identity mismatch")
  if not math.isfinite(ms) or ms<=0 or not math.isfinite(value):fail("invalid numeric value")
  for col,mkey in required.items():
   if r[col]!=str(manifest.get(mkey,"")):fail(f"manifest binding mismatch {col}")
  g=(r["stage"],n,r["variant_id"]);groups[g].append(r);round_orders[(r["stage"],n,rd)].append(order)
 initial={("initial",n,v[0]) for n in ns for v in allowed}
 if {g for g in groups if g[0]=="initial"}!=initial:fail("exact initial inventory mismatch")
 if mode=="full" and len([r for r in rows if r["stage"]=="initial"])!=3750:fail("expected 3750 initial rows")
 if mode=="smoke" and len(rows)!=90:fail("expected exactly 90 smoke rows")
 rerun_ns={n for stage,n,_ in groups if stage=="rerun"}
 for n in rerun_ns:
  if {(s,x,v) for s,x,v in groups if s=="rerun" and x==n}!={("rerun",n,v[0]) for v in allowed}:fail("incomplete rerun block")
 for g,items in groups.items():
  if len(items)!=samples_per or {int(x["round"]) for x in items}!=set(range(samples_per)):fail(f"case sample inventory mismatch {g}")
  if len({x["result"] for x in items})!=1:fail(f"nondeterministic result {g}")
 for key,orders in round_orders.items():
  if sorted(orders)!=list(range(15)):fail(f"execution order mismatch {key}")
 for stage,n,_ in groups:
  for p in PATHS:
   for b in BLOCKS:
    a=groups[(stage,n,f"{p}_b{b}_s32")][0]["result"];z=groups[(stage,n,f"{p}_b{b}_s64")][0]["result"]
    if a!=z:fail(f"scale-width result identity failed {stage} {n} {p} B{b}")
 expected_manifest=dict(mode=mode,sizes=ns,variants=15,samples=samples_per,warmups=10 if mode=="full" else 1,grid=512,threads=256,payload="fp32",scaled_arithmetic="fp64",access="scalar_x1",cache_protocol="repeat_input_no_explicit_flush",seed_set_id="block-scale-v1")
 if any(manifest.get(k)!=v for k,v in expected_manifest.items()):fail("manifest experiment identity mismatch")
 print(f"validation passed: {len(rows)} rows, {len(groups)} groups, mode={mode}")
 return rows,manifest
def main():
 p=argparse.ArgumentParser();p.add_argument("--samples",required=True);p.add_argument("--manifest",required=True);p.add_argument("--mode",choices=["full","smoke"]);a=p.parse_args();validate(a.samples,a.manifest,a.mode)
if __name__=="__main__":main()
