import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/"tools"))
from check_block_scale_codegen import row
RES=dict(registers=20,shared_bytes=2048,spill_loads=0,spill_stores=0)
def seq(ops):return [(i*16,op,"R1, R2, R3") for i,op in enumerate(ops)]
def test_generic_positive():
 s=[(0,"LDG.E","R1, desc"),(1,"LDG.E","R2, desc"),(2,"LDG.E","R3, desc"),(3,"LDG.E","R4, desc"),(4,"F2F.F64.F32","R5, R1"),(5,"F2F.F64.F32","R6, R2"),(6,"F2F.F64.F32","R7, R3"),(7,"F2F.F64.F32","R8, R4"),(8,"DMUL","R9, R5, R7"),(9,"DMUL","R10, R6, R8"),(10,"DFMA","R11, R9, R10, R12")]
 assert row("x","per_element",16,"fp32",s,RES)["status"]=="pass"
def test_generic_rejects_missing_multiply():assert row("x","per_element",16,"fp32",seq(["LDG.E","F2F.F64.F32","DMUL","DFMA"]),RES)["status"]=="fail"
def test_deferred_rejects_no_shuffle():assert row("x","deferred",32,"fp64",seq(["LDG.E","DMUL","DFMA","DFMA"]),RES)["status"]=="fail"
def test_rejects_spill():
 r=dict(RES,spill_loads=8);assert row("x","raw_fp64",0,"none",seq(["LDG.E","DFMA"]),r)["status"]=="fail"
def generic_sequence(mispair=False,intervene=False):
 s=[(0,"LDG.E","R1, desc"),(1,"LDG.E","R2, desc"),(2,"LDG.E","R3, desc"),(3,"LDG.E","R4, desc"),(4,"F2F.F64.F32","R5, R1"),(5,"F2F.F64.F32","R6, R2"),(6,"F2F.F64.F32","R7, R3"),(7,"F2F.F64.F32","R8, R4")]
 s += [(8,"DMUL","R9, R5, R6" if mispair else "R9, R5, R7"),(9,"DMUL","R10, R7, R8" if mispair else "R10, R6, R8")]
 if intervene:s.append((10,"DADD","R9, R9, R20"))
 s.append((11,"DFMA","R11, R9, R10, R12"));return s
def test_rejects_payload_payload_scale_scale_pairing():assert row("x","per_element",16,"fp32",generic_sequence(mispair=True),RES)["status"]=="fail"
def test_rejects_intervening_reconstruction_change():assert row("x","per_element",16,"fp32",generic_sequence(intervene=True),RES)["status"]=="fail"
def test_rejects_unrelated_raw64_fma():assert row("x","raw_fp64",0,"none",[(0,"LDG.E.64","R1, desc"),(1,"LDG.E.64","R2, desc"),(2,"DFMA","R3, R8, R10, R3")],RES)["status"]=="fail"
def test_rejects_deferred_before_complete_shuffle():
 s=[(0,"LDG.E","@!P0 R1, desc"),(1,"LDG.E","@!P0 R2, desc"),(2,"LDG.E","@!P1 R3, desc"),(3,"LDG.E","@!P1 R4, desc"),(4,"F2F.F64.F32","R5, R1"),(5,"F2F.F64.F32","R6, R2"),(6,"F2F.F64.F32","R7, R3"),(7,"F2F.F64.F32","R8, R4"),(8,"DFMA","R9, R5, R6, RZ"),(9,"SHFL.DOWN","R10, R9, 0x10, 0x1f"),(10,"DMUL","@!P1 R11, R7, R8"),(11,"DFMA","@!P1 R12, R9, R11, R12")]
 s.extend((12+i,"SHFL.DOWN",f"R20, R9, {off}, 0x1f") for i,off in enumerate(("0x10","0x8","0x8","0x4","0x4","0x2","0x2","0x1","0x1")));assert row("x","deferred",32,"fp32",s,RES)["status"]=="fail"
