import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location("audit",Path(__file__).parents[2]/"tools/check_instruction_cost_codegen.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
audit_symbol=module.audit_symbol
def i(op,args): return (0,op,args)
def test_rejects_folded_chain():
 s=[i("LDG.E","R2,[R8]"),i("LDG.E","R3,[R9]"),i("I2F.F64.U32","R4,R2"),i("I2F.F64.U32","R6,R3")]
 assert audit_symbol("x",s,"dot","add32",2)["status"]=="fail"
def test_rejects_independent_and_pointer_add():
 s=[i("LDG.E","R2,[R8]"),i("LDG.E","R3,[R9]"),i("IADD3","R10,R8,4"),i("IADD3","R4,R2,R20"),i("IADD3","R5,R2,R20"),i("I2F.F64.U32","R6,R4"),i("I2F.F64.U32","R8,R3")]
 assert audit_symbol("x",s,"dot","add32",2)["status"]=="fail"
def test_rejects_missing_operand_chain():
 s=[i("LDG.E","R2,[R8]"),i("LDG.E","R3,[R9]"),i("IADD3","R2,R2,R20"),i("I2F.F64.U32","R4,R2"),i("I2F.F64.U32","R6,R3")]
 assert audit_symbol("x",s,"dot","add32",1)["status"]=="fail"
def test_fma_accumulation_is_not_reconstruction():
 s=[i("LDG.E","R2,[R8]"),i("LDG.E","R3,[R9]"),i("I2F.F64.U32","R4,R2"),i("I2F.F64.U32","R6,R3"),i("DFMA","R8,R4,R6,R8")]
 assert audit_symbol("x",s,"dot","fma64",1)["status"]=="fail"
