"""Run the frozen prospective protocol only on complete, unexposed acquired data."""

import argparse
from pathlib import Path

from marketpulse.experiments.prospective_intake import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.manifest, args.output)
