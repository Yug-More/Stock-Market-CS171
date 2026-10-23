"""Execute Phase 10 full-budget expansion or separately aggregate/audit it."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["train", "analyze", "verify"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.step == "train":
        from marketpulse.experiments.expanded_seeds import run_expansion

        run_expansion(root, root / "results/phase10/multiseed")
    else:
        from marketpulse.experiments.expanded_analysis import analyze, verify

        (analyze if args.step == "analyze" else verify)(root)


if __name__ == "__main__":
    main()
