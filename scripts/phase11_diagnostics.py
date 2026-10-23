"""Audit adaptive saturation and interval nesting; diagnostics do not select a winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from marketpulse.experiments.recovery import file_hash
from marketpulse.experiments.robustness_data import inventory, write_json


def calculate(root: Path) -> pd.DataFrame:
    base = root / "results/phase11"
    manifest = json.loads((base / "manifest.json").read_text())
    rows = []
    for entry in manifest["files"]:
        part = pd.read_csv(base / entry["path"])
        chosen = part[part.eligible]
        crossing = (chosen.lower_80 < chosen.lower_90) | (chosen.upper_80 > chosen.upper_90)
        fields = {k: entry[k] for k in ["dataset", "horizon", "model", "variant"]}
        rows.append(
            {
                **fields,
                "eligible": len(chosen),
                "nesting_violations": int(crossing.sum()),
                "nesting_fraction": float(crossing.mean()),
                "mature_interval_feedback": int(part.feedback_n.iloc[-1]),
                "clipped_80": int(part.clipped_80.iloc[-1]),
                "clipped_90": int(part.clipped_90.iloc[-1]),
                "mean_alpha_80": float(chosen.alpha_80.mean()),
                "mean_alpha_90": float(chosen.alpha_90.mean()),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "results/phase11"
    calculated = calculate(root)
    path = output / "diagnostics.csv"
    context = output / "diagnostics_manifest.json"
    if args.verify:
        saved = pd.read_csv(path)
        assert saved.shape == calculated.shape
        np.testing.assert_allclose(
            saved.select_dtypes("number"), calculated.select_dtypes("number"), rtol=1e-10
        )
        assert saved[["dataset", "model", "variant"]].equals(
            calculated[["dataset", "model", "variant"]]
        )
        manifest = json.loads(context.read_text())
        assert manifest["source_sha256"] == file_hash(Path(__file__))
        assert manifest["study_manifest_sha256"] == file_hash(output / "manifest.json")
    else:
        if path.exists():
            raise ValueError("preserve previous diagnostics")
        calculated.to_csv(path, index=False)
        write_json(
            context,
            {
                "classification": "exploratory failure diagnostics; no method retuned",
                "source_sha256": file_hash(Path(__file__)),
                "study_manifest_sha256": file_hash(output / "manifest.json"),
            },
        )
        write_json(output / "checksums.json", inventory(output))
    print(
        json.dumps(
            {
                "diagnostic_rows": len(calculated),
                "nesting_violations": int(calculated.nesting_violations.sum()),
            }
        )
    )


if __name__ == "__main__":
    main()
