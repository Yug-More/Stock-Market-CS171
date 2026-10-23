"""Shared, auditable populations for extensions of the saved walk-forward study."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from marketpulse.data.catalog import load_catalog
from marketpulse.data.loader import load_market_csv
from marketpulse.evaluation.regimes import stress_periods
from marketpulse.experiments.benchmark import ALL_MODELS
from marketpulse.experiments.recovery import file_hash


def load_design(root: Path) -> dict:
    with (root / "configs/robustness.toml").open("rb") as stream:
        return tomllib.load(stream)


def enrich_predictions(root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    """Recover actual target realization dates and raw session positions."""
    pieces = []
    for entry in load_catalog(root):
        market = load_market_csv(root / entry.path, dataset_name=entry.name)
        dates = pd.to_datetime(market["date"]).dt.strftime("%Y-%m-%d").to_numpy()
        lookup = {date: index for index, date in enumerate(dates)}
        close = market["close"].to_numpy()
        part = frame.loc[frame.dataset == entry.name].copy()
        if part.empty:
            continue
        positions = part["date"].map(lookup)
        if positions.isna().any():
            raise ValueError("forecast origin is absent from its raw dataset")
        positions = positions.to_numpy(dtype=int)
        endpoints = positions + part.horizon.to_numpy(dtype=int)
        if np.any(endpoints >= len(dates)):
            raise ValueError("forecast target is unavailable")
        np.testing.assert_allclose(
            part.y_true, np.log(close[endpoints] / close[positions]), rtol=1e-10, atol=1e-14
        )
        part["origin_position"] = positions
        part["target_date"] = dates[endpoints]
        part["stress"] = "outside_stress"
        for period in stress_periods():
            mask = part.date.between(str(period.start.date()), str(period.end.date()))
            part.loc[mask, "stress"] = period.key
        pieces.append(part)
    if not pieces:
        raise ValueError("no recognized index predictions")
    result = pd.concat(pieces, ignore_index=True)
    if result.duplicated(["dataset", "horizon", "model", "date"]).any():
        raise ValueError("duplicate forecast identifier")
    if not np.isfinite(result[["y_true", "y_pred"]].to_numpy()).all():
        raise ValueError("nonfinite forecast")
    return result


def load_predictions(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "results/walkforward/full/predictions.csv.gz")
    if set(frame.model) != set(ALL_MODELS):
        raise ValueError("canonical models missing")
    return enrich_predictions(root, frame)


def matched_membership(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep shared origins, explicitly distinguish endpoint agreement."""
    base = predictions[predictions.model == "zero_return"]
    names = sorted(base.dataset.unique())
    if len(names) != 2:
        raise ValueError("matching requires exactly two indexes")
    columns = ["horizon", "date", "target_date"]
    left = base[base.dataset == names[0]][columns].rename(columns={"target_date": "endpoint_a"})
    right = base[base.dataset == names[1]][columns].rename(columns={"target_date": "endpoint_b"})
    joined = left.merge(
        right, on=["horizon", "date"], how="outer", indicator=True, validate="one_to_one"
    )
    joined["origin_match"] = joined["_merge"].eq("both")
    joined["endpoint_match"] = joined.origin_match & joined.endpoint_a.eq(joined.endpoint_b)
    return joined.drop(columns="_merge").sort_values(["horizon", "date"]).reset_index(drop=True)


def select_matched(
    frame: pd.DataFrame, membership: pd.DataFrame, *, endpoints: bool
) -> pd.DataFrame:
    key = "endpoint_match" if endpoints else "origin_match"
    keys = membership.loc[membership[key], ["horizon", "date"]]
    return frame.merge(keys, on=["horizon", "date"], how="inner", validate="many_to_one")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def inventory(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): file_hash(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "checksums.json" and not path.name.endswith(".tmp")
    }
