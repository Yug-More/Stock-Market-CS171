"""Run or independently audit the delayed adaptive uncertainty study."""

import argparse
from pathlib import Path

from marketpulse.experiments.adaptive_study import run, verify

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["run", "verify"])
    args = parser.parse_args()
    (run if args.step == "run" else verify)(Path(__file__).resolve().parents[1])
