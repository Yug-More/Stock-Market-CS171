"""Independently recompute every Phase 9 rehearsal score and actual date boundary."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from marketpulse.data.catalog import load_catalog
from marketpulse.data.loader import load_market_csv
from marketpulse.experiments.recovery import file_hash
from marketpulse.experiments.robustness_data import inventory, write_json


def audit(root: Path) -> dict:
    directory = root / "results/phase9"
    output = directory / "rehearsal"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["independent_validation_completed"] is False
    assert manifest["protocol_sha256"] == file_hash(root / "configs/independent_validation.json")
    metrics = pd.read_csv(output / "metrics.csv")
    count = 0
    for entry in load_catalog(root):
        market = load_market_csv(root / entry.path, dataset_name=entry.name)
        dates = pd.to_datetime(market.date).dt.strftime("%Y-%m-%d").to_numpy()
        positions = {date: i for i, date in enumerate(dates)}
        close = market.close.to_numpy()
        for horizon in [1, 5, 20]:
            plan = next(
                p
                for p in manifest["plans"]
                if p["dataset"] == entry.name and p["horizon"] == horizon
            )
            for a, b in zip(plan["blocks"], plan["blocks"][1:], strict=False):
                assert a["last_endpoint"] < b["first"]
                assert positions[a["last"]] + horizon == positions[a["last_endpoint"]]
            name = f"{entry.name.replace(' ', '_')}_h{horizon}"
            frame = pd.read_csv(output / f"{name}.csv.gz")
            pos = frame.date.map(positions).to_numpy(int)
            np.testing.assert_allclose(
                frame.y_true, np.log(close[pos + horizon] / close[pos]), atol=1e-14
            )
            assert len(frame) == 252 * 2 * 5
            for model, part in frame.groupby("model"):
                cal = part[part.partition == "calibration"]
                test = part[part.partition == "test"]
                assert positions[cal.date.max()] + horizon < positions[test.date.min()]
                assert len(cal) == len(test) == 252
                residuals = np.abs(cal.y_true - cal.y_pred).to_numpy()
                for level in [0.8, 0.9]:
                    rank = math.ceil((len(cal) + 1) * level)
                    q = float(np.partition(residuals, rank - 1)[rank - 1])
                    lower, upper = test.y_pred.to_numpy() - q, test.y_pred.to_numpy() + q
                    actual = test.y_true.to_numpy()
                    score = (
                        upper
                        - lower
                        + 2
                        / (1 - level)
                        * np.where(
                            actual < lower,
                            lower - actual,
                            np.where(actual > upper, actual - upper, 0),
                        )
                    )
                    expected = [
                        np.sqrt(np.mean((actual - test.y_pred) ** 2)),
                        np.mean((actual >= lower) & (actual <= upper)),
                        2 * q,
                        np.mean(score),
                    ]
                    row = metrics[
                        (metrics.dataset == entry.name)
                        & (metrics.horizon == horizon)
                        & (metrics.model == model)
                        & (metrics.nominal == level)
                    ].iloc[0]
                    np.testing.assert_allclose(
                        row[["rmse", "coverage", "mean_width", "interval_score"]].to_numpy(float),
                        expected,
                        rtol=1e-10,
                        atol=1e-13,
                    )
                    count += 1
    assert count == len(metrics) == 60
    return {
        "status": "passed",
        "rehearsal_model_fits": 30,
        "interval_metric_rows": count,
        "independent_confirmation": False,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = audit(root)
    directory = root / "results/phase9"
    checksum = directory / "checksums.json"
    if checksum.exists():
        assert json.loads(checksum.read_text()) == inventory(directory)
    else:
        write_json(directory / "audit.json", result)
        write_json(checksum, inventory(directory))
    print(json.dumps(result))
