"""Build or audit the evidence-backed final research synthesis."""

import argparse
from pathlib import Path

from marketpulse.experiments.final_synthesis import build, verify

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["build", "verify"])
    args = parser.parse_args()
    (build if args.step == "build" else verify)(Path(__file__).resolve().parents[1])
