from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from conversion_calibration.codegen import render, write
from conversion_calibration.manifest import (
    BLOCKS, CASES, FULL_N, ITERATIONS_PER_THREAD, KEY_XOR, LEFT_SEED,
    RIGHT_SEED, THREADS, manifest_document,
)


class ManifestTest(unittest.TestCase):
    def test_exact_inventory(self):
        self.assertEqual(Counter(case.split for case in CASES), {
            "train": 112, "synthetic_validation": 12,
            "development": 5, "final": 6,
        })
        self.assertEqual(Counter(case.group for case in CASES if case.split == "train"), {
            "anchors": 3, "integer_bit": 24, "expensive_integer": 6,
            "numeric_conversion": 11, "floating_point": 14,
            "mixed_pipeline": 10, "branches": 16, "luts": 17,
            "special_operations": 5, "latency_occupancy": 6,
        })
        self.assertEqual(len({case.case_id for case in CASES}), 135)

    def test_fixed_geometry_and_seeds(self):
        self.assertEqual(FULL_N, 1 << 27)
        self.assertEqual((BLOCKS, THREADS, ITERATIONS_PER_THREAD), (512, 256, 1024))
        self.assertEqual(RIGHT_SEED, LEFT_SEED ^ KEY_XOR)
        fixed = manifest_document()["fixed"]
        self.assertEqual(fixed["generator"], "Philox4x32-10")
        self.assertFalse(fixed["fast_math"])

    def test_branch_and_lut_metadata_are_explicit(self):
        for case in CASES:
            if case.kind == "branch":
                self.assertIsNotNone(case.branch_probability)
                self.assertAlmostEqual(case.branch_probability,
                                       1.0 / case.params["denominator"])
            if case.kind == "lut":
                self.assertEqual(case.lut_memory, case.params["memory"])
                self.assertEqual(case.lut_index_bits, case.params["bits"])
                self.assertEqual(case.lut_loads, case.params["loads"])

    def test_final_cases_never_enter_fit_split(self):
        train_ids = {case.case_id for case in CASES if case.split == "train"}
        self.assertFalse(train_ids & {case.case_id for case in CASES if case.split == "final"})
        for case in CASES:
            if case.kind == "real":
                self.assertGreaterEqual(len(case.provenance), 2)

    def test_codegen_is_deterministic_and_complete(self):
        first, second = render(), render()
        self.assertEqual(hashlib.sha256(first.encode()).digest(),
                         hashlib.sha256(second.encode()).digest())
        for index, case in enumerate(CASES):
            self.assertIn(f"template <> struct decoder<{index}>", first)
            self.assertIn(f'{{"{case.case_id}", "{case.split}"', first)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); write(root / "cases.cuh", root / "manifest.json")
            doc = json.loads((root / "manifest.json").read_text())
            self.assertEqual(len(doc["cases"]), 135)
            self.assertEqual(doc["generated_header_sha256"],
                             hashlib.sha256((root / "cases.cuh").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
