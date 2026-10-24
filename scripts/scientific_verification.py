"""Execute a separate-arithmetic review; preserve previously saved review reports."""

import argparse
import json
from pathlib import Path

from marketpulse.scientific_review import run

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
if args.output and args.output.exists():
    raise ValueError("choose a fresh review report path")
report = run(root)
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "inputs"}))
