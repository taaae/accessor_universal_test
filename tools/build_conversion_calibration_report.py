#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from conversion_calibration.report import build
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("run_dir",type=Path);p.add_argument("output",type=Path);a=p.parse_args();build(a.run_dir,a.output);print(a.output)
if __name__=="__main__":main()
