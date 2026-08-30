from __future__ import annotations

import unittest

from conversion_calibration.manifest import CASES
from conversion_calibration.sass import critical_depth, feature_row, main_loop, parse_functions


FIXTURE = r'''
Function : void aut::calibration::dot_case_kernel<0>(unsigned int const*, unsigned int const*, unsigned long, aut::calibration::device_tables, double*)
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   IADD3 R4, R2, 0x1, RZ ; /* 0x0000000102047810 */
        /*0020*/                   LOP3.LUT R5, R4, 0xff, RZ, 0xc0, !PT ; /* 0x000fe200078e00ff */
        /*0030*/              @P0  IADD3 R6, R5, 0x2, RZ ;
        /*0040*/                   BRA 0x10 ;
        /*0050*/                   EXIT ;
'''


class SassTest(unittest.TestCase):
    def test_parse_loop_and_dependency(self):
        functions = parse_functions(FIXTURE)
        self.assertEqual(len(functions), 1)
        instructions = next(iter(functions.values()))
        loop = main_loop(instructions)
        self.assertEqual([item.address for item in loop], [0x10, 0x20, 0x30, 0x40])
        self.assertEqual(sum(item.predicate for item in loop), 1)
        self.assertGreaterEqual(critical_depth(loop), 3)

    def test_global_lut_sector_work_counts_both_operands(self):
        index = next(i for i, case in enumerate(CASES)
                     if case.case_id == "lut_global_1x_4")
        row = feature_row(index, [])
        sectors = 4  # 16 doubles occupy four 32-byte sectors.
        one_lookup = sectors * (1.0 - (1.0 - 1.0 / sectors) ** 32)
        self.assertAlmostEqual(row["lut_expected_sectors_per_warp"],
                               2.0 * one_lookup)

    def test_hopper_imad_iadd_is_add_not_multiply(self):
        fixture = r'''
Function : void aut::calibration::dot_case_kernel<3>()
        /*0010*/ IMAD.IADD R5, R4, 0x1, R3 ;
        /*0020*/ IADD3 R6, R5, R2, RZ ;
        /*0030*/ BRA 0x10 ;
'''
        instructions = next(iter(parse_functions(fixture).values()))
        row = feature_row(3, instructions)
        self.assertEqual(row["iadd_sass"], 2)
        self.assertEqual(row["iadd3_sass"], 1)
        self.assertEqual(row["imad_sass"], 0)
        self.assertEqual(row["integer_multiply"], 0)


if __name__ == "__main__":
    unittest.main()
