import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/"tools"))
from check_block_scale_codegen import row
RES=dict(registers=20,shared_bytes=2048,spill_loads=0,spill_stores=0)
def seq(ops):return [(i*16,op,"R1, R2, R3") for i,op in enumerate(ops)]
def test_generic_positive():
 s=[(0,"LDG.E","R1, desc"),(1,"LDG.E","R2, desc"),(2,"LDG.E","R3, desc"),(3,"LDG.E","R4, desc"),(4,"F2F.F64.F32","R5, R1"),(5,"F2F.F64.F32","R6, R2"),(6,"F2F.F64.F32","R7, R3"),(7,"F2F.F64.F32","R8, R4"),(8,"DMUL","R9, R5, R6"),(9,"DMUL","R10, R7, R8"),(10,"DFMA","R11, R9, R10, R12")]
 assert row("x","per_element",16,"fp32",s,RES)["status"]=="pass"
def test_generic_rejects_missing_multiply():assert row("x","per_element",16,"fp32",seq(["LDG.E","F2F.F64.F32","DMUL","DFMA"]),RES)["status"]=="fail"
def test_deferred_rejects_no_shuffle():assert row("x","deferred",32,"fp64",seq(["LDG.E","DMUL","DFMA","DFMA"]),RES)["status"]=="fail"
def test_rejects_spill():
 r=dict(RES,spill_loads=8);assert row("x","raw_fp64",0,"none",seq(["LDG.E","DFMA"]),r)["status"]=="fail"
