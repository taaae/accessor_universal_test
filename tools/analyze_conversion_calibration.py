#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from conversion_calibration.analysis import analyze

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("timing",type=Path); p.add_argument("features",type=Path); p.add_argument("output",type=Path); a=p.parse_args()
    result=analyze(a.timing,a.features,a.output)
    print(f"timing_qc={result['quality_control']['timing_pass']} acceptance={result['acceptance']['all_pass']} model={result['model_sha256']}")

if __name__=="__main__": main()
