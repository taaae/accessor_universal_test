import csv,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/"tools"))
from validate_block_scale_results import validate,variants,FULL_N
FIELDS="mode,stage,N,B,scale_type,path,variant_id,payload_type,arithmetic,grid,threads,round,order,time_ms,result,valid,job_id,node,seed_set_id,source_commit,binary_sha,sass_sha,audit_sha".split(",")
def fixture(tmp_path):
 m=dict(mode="full",sizes=FULL_N,variants=15,samples=50,warmups=10,grid=512,threads=256,payload="fp32",scaled_arithmetic="fp64",access="scalar_x1",cache_protocol="repeat_input_no_explicit_flush",seed_set_id="block-scale-v1",commit="c",binary_sha256="b",sass_sha256="s",audit_sha256="a",job_id="j",node="n");mp=tmp_path/"m.json";mp.write_text(json.dumps(m));p=tmp_path/"x.csv"
 with p.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
  for n in FULL_N:
   vv=sorted(variants())
   for rd in range(50):
    for order,(vid,b,scale,path,payload,arith) in enumerate(vv):w.writerow(dict(mode="full",stage="initial",N=n,B=b,scale_type=scale,path=path,variant_id=vid,payload_type=payload,arithmetic=arith,grid=512,threads=256,round=rd,order=order,time_ms=1,result=2,valid=1,job_id="j",node="n",seed_set_id="block-scale-v1",source_commit="c",binary_sha="b",sass_sha="s",audit_sha="a"))
 return p,mp
def rewrite(p,rows):
 with p.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
def load(p):return list(csv.DictReader(p.open()))
def test_valid(tmp_path):p,m=fixture(tmp_path);validate(p,m,"full")
@pytest.mark.parametrize("mutation",["missing","duplicate","bad_n","bad_time","bad_digest","bad_order","bad_variant","nondeterministic"])
def test_rejects(tmp_path,mutation):
 p,m=fixture(tmp_path);r=load(p)
 if mutation=="missing":r.pop()
 elif mutation=="duplicate":r.append(r[-1].copy())
 elif mutation=="bad_n":r[0]["N"]="7"
 elif mutation=="bad_time":r[0]["time_ms"]="nan"
 elif mutation=="bad_digest":r[0]["binary_sha"]="wrong"
 elif mutation=="bad_order":r[0]["order"]="14"
 elif mutation=="bad_variant":r[0]["variant_id"]="unknown"
 else:r[0]["result"]="3"
 rewrite(p,r)
 with pytest.raises(ValueError):validate(p,m,"full")
