"""Run the separately versioned Phase 6–8 research steps from repository root."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step",
        choices=[
            "audit",
            "matched",
            "seeds",
            "statistics",
            "calibration",
            "figures",
            "report",
            "verify",
        ],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.step == "seeds":
        from marketpulse.experiments.multiseed import run_multiseed

        output = args.output or root / "results/robustness/multiseed"
        run_multiseed(root, output, smoke=args.smoke)
    else:
        from marketpulse.experiments.research_cycle import run_step

        run_step(root, args.step, output=args.output)


if __name__ == "__main__":
    main()
