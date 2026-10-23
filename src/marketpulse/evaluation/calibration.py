"""Prequential intervals from strictly mature, genuinely out-of-sample residuals."""

from __future__ import annotations

import bisect
import math
from collections import deque

import numpy as np
import pandas as pd


def residual_quantile(sorted_scores: list[float], coverage: float) -> float:
    """Conformal-style ceil((n+1)*coverage) order statistic; never clip k>n."""
    if not 0 < coverage < 1 or not sorted_scores:
        raise ValueError("coverage and nonempty calibration scores are required")
    if not np.isfinite(sorted_scores).all() or np.any(np.asarray(sorted_scores) < 0):
        raise ValueError("calibration scores must be nonnegative and finite")
    if any(a > b for a, b in zip(sorted_scores, sorted_scores[1:], strict=False)):
        raise ValueError("quantile input must be sorted")
    k = math.ceil((len(sorted_scores) + 1) * coverage)
    return float(sorted_scores[k - 1]) if k <= len(sorted_scores) else math.inf


def interval_score(
    actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, coverage: float
) -> np.ndarray:
    actual, lower, upper = map(lambda a: np.asarray(a, float), (actual, lower, upper))
    if actual.shape != lower.shape or lower.shape != upper.shape or not 0 < coverage < 1:
        raise ValueError("interval vectors must align and coverage must lie in (0,1)")
    if not np.isfinite(np.r_[actual, lower, upper]).all() or np.any(lower > upper):
        raise ValueError("intervals must be finite and ordered")
    alpha = 1 - coverage
    return (
        upper - lower + 2 / alpha * (np.maximum(lower - actual, 0) + np.maximum(actual - upper, 0))
    )


def chronological_intervals(
    frame: pd.DataFrame, *, method: str, levels: list[float], minimum: int = 252, window: int = 504
) -> pd.DataFrame:
    """Calibrate one index/horizon/model group without looking beyond each origin.

    Windows count mature residuals. Actual endpoint dates control eligibility;
    targets touching the current decision date are conservatively excluded.
    """
    if method not in ("trailing", "expanding", "volatility_scaled"):
        raise ValueError("unknown calibration method")
    if minimum < 1 or window < minimum or not levels or any(not 0 < p < 1 for p in levels):
        raise ValueError("invalid calibration history or nominal levels")
    ordered = frame.sort_values("date").reset_index(drop=True)
    if ordered.empty or ordered.date.duplicated().any():
        raise ValueError("calibration requires unique chronological origins")
    dates = pd.to_datetime(ordered.date).to_numpy()
    endpoints = pd.to_datetime(ordered.target_date).to_numpy()
    if np.any(endpoints <= dates) or np.any(endpoints[1:] <= endpoints[:-1]):
        raise ValueError("target endpoints must follow origins and be strictly increasing")
    residuals = np.abs(ordered.y_true.to_numpy() - ordered.y_pred.to_numpy())
    volatility = ordered.volatility.to_numpy()
    if not np.isfinite(residuals).all():
        raise ValueError("nonfinite residuals")
    scaled = method == "volatility_scaled"
    if scaled and (not np.isfinite(volatility).all() or np.any(volatility <= 0)):
        raise ValueError("volatility scaling requires causal finite positive volatility")
    scores = residuals / volatility if scaled else residuals
    history: list[float] = []
    queue: deque[tuple[float, int]] = deque()
    cursor = 0
    rows = []
    for i in range(len(ordered)):
        while cursor < i and endpoints[cursor] < dates[i]:
            value = float(scores[cursor])
            bisect.insort(history, value)
            queue.append((value, cursor))
            cursor += 1
            if method != "expanding" and len(queue) > window:
                old, _ = queue.popleft()
                history.pop(bisect.bisect_left(history, old))
        row = {
            "date": ordered.date.iloc[i],
            "target_date": ordered.target_date.iloc[i],
            "origin_position": int(ordered.origin_position.iloc[i]),
            "y_true": float(ordered.y_true.iloc[i]),
            "y_pred": float(ordered.y_pred.iloc[i]),
            "volatility": float(volatility[i]),
            "regime": ordered.regime.iloc[i],
            "stress": ordered.stress.iloc[i],
            "calibration_n": len(history),
            "eligible": len(history) >= minimum,
            "status": "warmup",
        }
        if queue:
            first, last = queue[0][1], queue[-1][1]
            row.update(
                calibration_first_origin=ordered.date.iloc[first],
                calibration_last_origin=ordered.date.iloc[last],
                residual_cutoff_date=ordered.target_date.iloc[last],
            )
        for level in levels:
            suffix = str(round(100 * level))
            # history is maintained sorted; validate quantile inputs in the public
            # helper/tests, avoid rescanning a growing history at every origin.
            k = math.ceil((len(history) + 1) * level)
            q = history[k - 1] if row["eligible"] and k <= len(history) else math.nan
            radius = q * volatility[i] if scaled else q
            row[f"lower_{suffix}"] = row["y_pred"] - radius
            row[f"upper_{suffix}"] = row["y_pred"] + radius
            if row["eligible"] and k > len(history):
                row["eligible"] = False
                row["status"] = "insufficient_quantile"
        if row["eligible"]:
            row["status"] = "eligible"
        rows.append(row)
    return pd.DataFrame(rows)


def interval_metrics(records: pd.DataFrame, level: float) -> dict:
    """Coverage and proper score on eligible finite intervals only."""
    suffix = str(round(level * 100))
    part = records.loc[records.eligible]
    base = {
        "n_origins": len(records),
        "n": len(part),
        "excluded": len(records) - len(part),
        "nominal": level,
    }
    if part.empty:
        return base
    actual = part.y_true.to_numpy()
    lower, upper = part[f"lower_{suffix}"].to_numpy(), part[f"upper_{suffix}"].to_numpy()
    score = interval_score(actual, lower, upper, level)
    empirical = float(np.mean((actual >= lower) & (actual <= upper)))
    return {
        **base,
        "coverage": empirical,
        "coverage_gap": empirical - level,
        "mean_width": float(np.mean(upper - lower)),
        "median_width": float(np.median(upper - lower)),
        "interval_score": float(np.mean(score)),
    }
