from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from conversion_calibration.analysis import analyze, load_timings
from conversion_calibration.manifest import CASES
from conversion_calibration.report import build


FEATURE_FIELDS = [
    "case_id","split","group","function","loop_instruction_count","integer_alu",
    "integer_multiply","integer_divide","conversion","fp32","fp64","special",
    "global_loads","shared_loads","constant_loads","local_loads","local_stores",
    "branch_count","predicated_instruction_count","expected_true_warp_path",
    "expected_false_warp_path","critical_dependency_depth","lut_loads_metadata",
    "lut_footprint_bytes","lut_expected_sectors_per_warp","branch_probability",
    "registers_per_thread","static_shared_bytes","dynamic_shared_bytes",
    "local_bytes_per_thread","estimated_occupancy",
]


class AnalysisTest(unittest.TestCase):
    def make_fixture(self, root: Path):
        feature_path=root/"features.csv"
        with feature_path.open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=FEATURE_FIELDS);writer.writeheader()
            for index,case in enumerate(CASES):
                writer.writerow({"case_id":case.case_id,"split":case.split,"group":case.group,"function":f"k{index}",
                    "loop_instruction_count":10+index%7,"integer_alu":index%5,"integer_multiply":0,"integer_divide":0,
                    "conversion":index%3,"fp32":0,"fp64":index%11,"special":0,"global_loads":case.lut_loads,
                    "shared_loads":0,"constant_loads":0,"local_loads":0,"local_stores":0,"branch_count":int(case.kind=="branch"),
                    "predicated_instruction_count":index%4,"expected_true_warp_path":0,"expected_false_warp_path":0,
                    "critical_dependency_depth":2+index%9,"lut_loads_metadata":case.lut_loads,"lut_footprint_bytes":0,
                    "lut_expected_sectors_per_warp":case.lut_loads*4,"branch_probability":case.branch_probability or 0,
                    "registers_per_thread":32,"static_shared_bytes":0,"dynamic_shared_bytes":2048,"local_bytes_per_thread":0,"estimated_occupancy":1})
        timing_path=root/"timing_samples.csv"
        with timing_path.open("w",newline="") as handle:
            fields=["case_id","split","group","round","position","sample","stage","repeats","n","elapsed_ms","per_dot_ms","result_bits","status"]
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
            for case_index,case in enumerate(CASES):
                value=0.4+0.003*(case_index%7)+0.001*(case_index%5)
                for round_index in range(3):
                    for sample in range(10):
                        writer.writerow({"case_id":case.case_id,"split":case.split,"group":case.group,"round":round_index,"position":case_index,"sample":sample,"stage":"measurement","repeats":1,"n":1<<27,"elapsed_ms":value,"per_dot_ms":value*(1+(sample-4.5)*0.0001),"result_bits":"0x0","status":"ok"})
            anchor=CASES[1]
            for round_index in range(3):
                for stage,value in (("round_start_anchor",0.4),("round_end_anchor",0.401)):
                    writer.writerow({"case_id":anchor.case_id,"split":anchor.split,"group":anchor.group,"round":round_index,"position":-1,"sample":0,"stage":stage,"repeats":1,"n":1<<27,"elapsed_ms":value,"per_dot_ms":value,"result_bits":"0x0","status":"ok"})
        return timing_path,feature_path

    def test_fit_freeze_evaluate_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); timing,features=self.make_fixture(root); output=root/"analysis"
            result=analyze(timing,features,output)
            model=json.loads((output/"model.json").read_text())
            self.assertEqual(len(model["fit_case_ids"]),112)
            self.assertTrue(model["frozen_before_final_evaluation"])
            self.assertFalse(set(model["fit_case_ids"]) & {case.case_id for case in CASES if case.split=="final"})
            self.assertTrue(result["quality_control"]["timing_pass"])
            run=root/"run";run.mkdir();(run/"analysis").mkdir()
            for name in ("analysis.json","model.json","predictions.csv","timing_summary.csv"):
                (run/"analysis"/name).write_bytes((output/name).read_bytes())
            (run/"features.csv").write_bytes(features.read_bytes())
            for name in ("timing_samples.csv","sass.txt","environment.txt"):(run/name).write_text("fixture\n")
            build(run,run/"analysis"/"report.html")
            report=(run/"analysis"/"report.html").read_text()
            self.assertIn("H200 conversion-cost calibration",report)
            self.assertIn("final_lns32_r23_reference_exp2",report)

    def test_missing_anchors_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); timing,_=self.make_fixture(root)
            rows=[]
            with timing.open(newline="") as handle:
                rows=list(csv.DictReader(handle)); fields=list(rows[0])
            with timing.open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
                writer.writerows(row for row in rows if "anchor" not in row["stage"])
            with self.assertRaisesRegex(ValueError,"anchor"):
                load_timings(timing)


if __name__=="__main__":unittest.main()
