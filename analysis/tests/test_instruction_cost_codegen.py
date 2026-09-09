import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location("audit",Path(__file__).parents[2]/"tools/check_instruction_cost_codegen.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def audit(*args):return module.audit_symbol(*args,{"registers":20,"spill_loads":0,"spill_stores":0,"shared_bytes":2048})
def i(a,op,args):return(a,op,args)
def trace(op="FADD",k=1,overwrite=False):
 s=[i(0x100,"LDG.E","R2,[R8]"),i(0x110,"LDG.E","R3,[R9]"),i(0x120,"I2F.F32.U32","R4,R2"),i(0x130,"I2F.F32.U32","R5,R3")]
 if overwrite:s.append(i(0x135,"MOV","R4,R20"))
 for n in range(k):s += [i(0x140+n*0x20,op,"R4,R4,UR4"),i(0x150+n*0x20,op,"R5,R5,UR4")]
 end=0x140+k*0x20;s += [i(end,"F2F.F64.F32","R6,R4"),i(end+0x10,"F2F.F64.F32","R8,R5"),i(end+0x20,"DFMA","R10,R6,R8,R10"),i(end+0x30,"BRA","0x100")];return s
def test_accepts_exact_add_chain():assert audit("x",trace(),"dot","add32f",1)["status"]=="pass"
def test_rejects_folded_chain():assert audit("x",trace(k=1),"dot","add32f",2)["status"]=="fail"
def test_rejects_overwritten_chain():assert audit("x",trace(overwrite=True),"dot","add32f",1)["status"]=="fail"
def test_dfma_accumulation_not_fp32_fma():assert audit("x",trace(op="FADD"),"dot","fma32f",1)["status"]=="fail"
def test_rejects_extra_dependent_family():
 s=trace();s.insert(-4,i(0x155,"FMUL","R4,R4,UR6"));assert audit("x",s,"dot","add32f",1)["status"]=="fail"
def test_rejects_merged_operands():
 s=trace();s.insert(-3,i(0x155,"MOV","R5,R4"));assert audit("x",s,"dot","add32f",1)["status"]=="fail"
