"""Bounded ACI-inspired feedback; both residual and coverage feedback must mature."""

from __future__ import annotations

import bisect
import math
from collections import deque

import numpy as np
import pandas as pd


def adaptive_intervals(
    frame: pd.DataFrame,
    *,
    window: int,
    gamma: float,
    minimum: int = 252,
    levels: tuple[float, ...] = (0.8, 0.9),
    alpha_bounds: tuple[float, float] = (0.02, 0.5),
) -> pd.DataFrame:
    low, high = alpha_bounds
    if (
        minimum < 1
        or window < minimum
        or gamma < 0
        or not levels
        or any(not 0 < p < 1 for p in levels)
        or not 1 / (minimum + 1) <= low < high < 1
        or any(not low <= 1 - p <= high for p in levels)
    ):
        raise ValueError("invalid history, feedback or finite-quantile bounds")
    source = frame.sort_values("date").reset_index(drop=True)
    if source.empty or source.date.duplicated().any():
        raise ValueError("unique nonempty origins required")
    dates = pd.to_datetime(source.date).to_numpy()
    ends = pd.to_datetime(source.target_date).to_numpy()
    if np.any(ends <= dates) or np.any(ends[1:] <= ends[:-1]):
        raise ValueError("actual endpoints must be ordered and later than origins")
    actual, prediction = source.y_true.to_numpy(float), source.y_pred.to_numpy(float)
    if not np.isfinite(np.r_[actual, prediction]).all():
        raise ValueError("finite forecasts and outcomes required")
    n, k = len(source), len(levels)
    lower, upper = np.full((n, k), np.nan), np.full((n, k), np.nan)
    alphas = np.empty((n, k))
    alpha = 1 - np.array(levels)
    eligible = np.zeros(n, bool)
    clipped = np.zeros(k, int)
    counts, feedbacks, clips = [], [], []
    cutoffs = []
    history: list[float] = []
    queue: deque[float] = deque()
    cursor = feedback = 0
    for i in range(n):
        while cursor < i and ends[cursor] < dates[i]:
            value = float(abs(actual[cursor] - prediction[cursor]))
            bisect.insort(history, value)
            queue.append(value)
            if len(queue) > window:
                old = queue.popleft()
                history.pop(bisect.bisect_left(history, old))
            if eligible[cursor]:
                miss = (actual[cursor] < lower[cursor]) | (actual[cursor] > upper[cursor])
                candidate = alpha + gamma * ((1 - np.array(levels)) - miss)
                clipped += (candidate < low) | (candidate > high)
                alpha = np.clip(candidate, low, high)
                feedback += 1
            cursor += 1
        alphas[i] = alpha
        counts.append(len(history))
        feedbacks.append(feedback)
        clips.append(clipped.copy())
        cutoffs.append(source.target_date.iloc[cursor - 1] if cursor else None)
        eligible[i] = len(history) >= minimum
        if eligible[i]:
            for j in range(k):
                rank = math.ceil((len(history) + 1) * (1 - alpha[j]))
                if rank > len(history):
                    raise AssertionError("configured alpha bound did not ensure finite rank")
                radius = history[rank - 1]
                lower[i, j], upper[i, j] = prediction[i] - radius, prediction[i] + radius
    result = source[["date", "target_date", "origin_position"]].copy()
    result["calibration_n"], result["eligible"] = counts, eligible
    result["feedback_n"], result["residual_cutoff_date"] = feedbacks, cutoffs
    for j, level in enumerate(levels):
        suffix = round(100 * level)
        result[f"alpha_{suffix}"] = alphas[:, j]
        result[f"lower_{suffix}"], result[f"upper_{suffix}"] = lower[:, j], upper[:, j]
        result[f"clipped_{suffix}"] = np.asarray(clips)[:, j]
    return result
