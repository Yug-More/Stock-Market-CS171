"""Execute the explicitly retrospective four-block infrastructure rehearsal."""

from pathlib import Path

from marketpulse.experiments.independent_validation import run_rehearsal

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run_rehearsal(root, root / "results/phase9/rehearsal")
