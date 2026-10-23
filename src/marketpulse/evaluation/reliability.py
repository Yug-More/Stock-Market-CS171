"""Paired forecast inference with session-preserving moving blocks and Bartlett HAC."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd


def paired_losses(frame: pd.DataFrame, first: str, second: str, *, absolute: bool = False):
    """Require identical unique origins, target endpoints and observed labels."""
    left = frame[frame.model == first].sort_values("origin_position").reset_index(drop=True)
    right = frame[frame.model == second].sort_values("origin_position").reset_index(drop=True)
    if left.empty or left["date"].duplicated().any() or right["date"].duplicated().any():
        raise ValueError("paired forecasts require nonempty unique origins")
    for column in ("date", "target_date", "origin_position"):
        if not left[column].equals(right[column]):
            raise ValueError(f"unpaired {column}")
    if not np.array_equal(left.y_true.to_numpy(), right.y_true.to_numpy()):
        raise ValueError("paired outcomes differ")
    residual_a = left.y_pred.to_numpy() - left.y_true.to_numpy()
    residual_b = right.y_pred.to_numpy() - right.y_true.to_numpy()
    if not np.isfinite(np.r_[residual_a, residual_b]).all():
        raise ValueError("nonfinite paired predictions")
    operation = np.abs if absolute else np.square
    return left, operation(residual_a), operation(residual_b)


def valid_block_starts(positions: np.ndarray, length: int) -> np.ndarray:
    """Blocks never span a structural session gap."""
    positions = np.asarray(positions, dtype=int)
    if length < 1 or positions.ndim != 1 or np.any(np.diff(positions) <= 0):
        raise ValueError("invalid block length or session positions")
    if len(positions) < length:
        return np.array([], dtype=int)
    starts = np.arange(len(positions) - length + 1)
    return starts[positions[starts + length - 1] - positions[starts] == length - 1]


def moving_block_means(
    values: np.ndarray, positions: np.ndarray, length: int, replicates: int, seed: int
) -> np.ndarray:
    """Resample chronological block sums, preserving paired columns and exact n.

    Prefix sums avoid materializing replicates x n x columns. The final block
    is truncated to n modulo length, with starts drawn from valid shorter blocks.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or len(values) != len(positions):
        raise ValueError("resampling values must be finite aligned columns")
    starts = valid_block_starts(positions, length)
    if len(starts) == 0 or replicates < 2:
        raise ValueError("insufficient contiguous history or replicates")
    prefix = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    sums = prefix[starts + length] - prefix[starts]
    rng = np.random.default_rng(seed)
    count, remainder = divmod(len(values), length)
    draws = rng.integers(0, len(starts), size=(replicates, count))
    result = sums[draws].sum(axis=1)
    if remainder:
        short = valid_block_starts(positions, remainder)
        picked = rng.choice(short, size=replicates)
        result += prefix[picked + remainder] - prefix[picked]
    return result / len(values)


def hac_variance(influence: np.ndarray, positions: np.ndarray, bandwidth: int) -> float:
    """Variance of the sample mean using Bartlett weights within contiguous runs."""
    values = np.asarray(influence, dtype=float)
    if values.ndim != 1 or len(values) < 2 or len(values) != len(positions):
        raise ValueError("HAC needs aligned vectors with at least two observations")
    if not np.isfinite(values).all() or not 0 <= bandwidth < len(values):
        raise ValueError("invalid HAC values or bandwidth")
    if np.any(np.diff(positions) <= 0):
        raise ValueError("HAC positions must be strictly increasing")
    centered = values - np.mean(values)
    n = len(values)
    segments = np.r_[0, np.cumsum(np.diff(positions) != 1)]
    long_run = float(centered @ centered / n)
    for lag in range(1, bandwidth + 1):
        same_run = segments[lag:] == segments[:-lag]
        gamma = np.sum(centered[lag:][same_run] * centered[:-lag][same_run]) / n
        long_run += 2 * (1 - lag / (bandwidth + 1)) * gamma
    return max(float(long_run / n), 0.0)


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Holm stepdown over the entire declared family; undefined tests count as p=1."""
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or np.any(
        (values[np.isfinite(values)] < 0) | (values[np.isfinite(values)] > 1)
    ):
        raise ValueError("p-values must lie in [0, 1]")
    values = np.where(np.isfinite(values), values, 1.0)
    order = np.argsort(values, kind="stable")
    adjusted = np.maximum.accumulate(values[order] * (len(values) - np.arange(len(values))))
    result = np.empty_like(values)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def compare_losses(
    a: np.ndarray,
    b: np.ndarray,
    positions: np.ndarray,
    *,
    length: int,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
    minimum_blocks: int = 10,
    mask: np.ndarray | None = None,
) -> dict:
    """Conditional paired loss effects without compressing regime-selected dates."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) != len(positions):
        raise ValueError("loss vectors must be aligned")
    if not np.isfinite(np.r_[a, b]).all() or np.any(np.r_[a, b] < 0):
        raise ValueError("losses must be nonnegative and finite")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    membership = np.ones(len(a), bool) if mask is None else np.asarray(mask, bool)
    if membership.shape != a.shape:
        raise ValueError("regime membership must align with the full timeline")
    n = int(membership.sum())
    result = {
        "n": n,
        "timeline_n": len(a),
        "block_length": length,
        "bandwidth": length - 1,
        "nominal_block_equivalents": n / length,
        "confidence": confidence,
        "bootstrap_replicates": replicates,
        "status": "descriptive_small_sample",
    }
    if n == 0:
        return {**result, "status": "unavailable"}
    mean_a, mean_b = float(a[membership].mean()), float(b[membership].mean())
    difference = mean_a - mean_b
    result.update(
        loss_a=mean_a,
        loss_b=mean_b,
        mean_loss_difference=difference,
        skill_vs_comparator=(1 - mean_a / mean_b) if mean_b > 0 else None,
    )
    if n < length * minimum_blocks or len(valid_block_starts(positions, length)) == 0:
        return result
    columns = np.column_stack([a * membership, b * membership, membership.astype(float)])
    means = moving_block_means(columns, positions, length, replicates, seed)
    valid = means[:, 2] > 0
    means = means[valid]
    result["valid_bootstrap_replicates"] = int(valid.sum())
    if valid.sum() < 0.95 * replicates:
        return {**result, "status": "descriptive_unstable_membership"}
    averaged = means[:, :2] / means[:, 2, None]
    alpha = (1 - confidence) / 2
    bounds = np.quantile(averaged[:, 0] - averaged[:, 1], [alpha, 1 - alpha])
    result.update(ci_lower=float(bounds[0]), ci_upper=float(bounds[1]))
    if np.all(averaged[:, 1] > 0):
        skills = np.quantile(1 - averaged[:, 0] / averaged[:, 1], [alpha, 1 - alpha])
        result.update(skill_ci_lower=float(skills[0]), skill_ci_upper=float(skills[1]))
    influence = membership * (a - b - difference) * len(a) / n
    variance = hac_variance(influence, positions, length - 1)
    result["hac_standard_error"] = math.sqrt(variance)
    if variance <= 0:
        return {**result, "status": "descriptive_degenerate_variance"}
    statistic = difference / math.sqrt(variance)
    critical = NormalDist().inv_cdf((1 + confidence) / 2)
    result.update(
        status="approximate_inference",
        statistic=statistic,
        p_value=math.erfc(abs(statistic) / math.sqrt(2)),
        hac_ci_lower=difference - critical * math.sqrt(variance),
        hac_ci_upper=difference + critical * math.sqrt(variance),
    )
    return result
