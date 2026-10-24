"""Independent arithmetic checks for the final MarketPulse evidence audit.

This deliberately uses direct CSV/JSON parsing and NumPy calculations rather
than the package's publication, metrics, or scientific-review helpers. It
checks what can be reconstructed from the repository; it cannot authenticate
the original data acquisition chain or recover missing historical weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("results/final_evidence_authenticity_audit")
DATASETS = {
    "NASDAQ-100": ("data/nasdaq100.csv", "^NDX"),
    "S&P 500": ("data/SPX.csv", "^GSPC"),
}
MODELS = ["zero_return", "historical_mean", "linear_regression", "random_forest", "lstm"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(name: str) -> pd.DataFrame:
    path, _ = DATASETS[name]
    if name == "NASDAQ-100":
        raw = pd.read_csv(ROOT / path, header=[0, 1, 2])
        frame = raw.iloc[:, :6].copy()
        frame.columns = ["date", "close", "high", "low", "open", "volume"]
    else:
        raw = pd.read_csv(ROOT / path)
        frame = raw.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in frame.columns:
        if column != "date":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame.date.notna()].reset_index(drop=True)
    return frame


def dataset_audit(name: str) -> tuple[pd.DataFrame, dict]:
    frame = load_raw(name)
    numeric = frame.select_dtypes(include=[np.number])
    values_finite = bool(np.isfinite(numeric.to_numpy()).all())
    date_diff = frame.date.diff().dt.days
    returns = np.log(frame.close).diff()
    close_run = frame.close.eq(frame.close.shift()).fillna(False)
    if "adj_close" in frame:
        adjusted_equal = bool(np.array_equal(frame.adj_close.to_numpy(), frame.close.to_numpy()))
    else:
        adjusted_equal = None
    checks = {
        "path": DATASETS[name][0],
        "sha256": sha256(ROOT / DATASETS[name][0]),
        "rows": len(frame),
        "columns": list(frame.columns),
        "dtypes": {key: str(value) for key, value in frame.dtypes.items()},
        "first_date": frame.date.iloc[0].date().isoformat(),
        "last_date": frame.date.iloc[-1].date().isoformat(),
        "missing_by_column": {key: int(value) for key, value in frame.isna().sum().items()},
        "duplicate_dates": int(frame.date.duplicated().sum()),
        "dates_strictly_increasing": bool(frame.date.is_monotonic_increasing),
        "numeric_values_finite": values_finite,
        "nonpositive_ohlc_values": {
            key: int((frame[key] <= 0).sum()) for key in ["open", "high", "low", "close"]
        },
        "ohlc_relationship_violations": int(
            (
                (frame.high < frame[["open", "close"]].max(axis=1))
                | (frame.low > frame[["open", "close"]].min(axis=1))
                | (frame.high < frame.low)
            ).sum()
        ),
        "duplicate_full_rows": int(frame.duplicated().sum()),
        "consecutive_identical_closes": int(close_run.sum()),
        "consecutive_identical_close_examples": [
            {
                "date": frame.date.iloc[i].date().isoformat(),
                "previous_date": frame.date.iloc[i - 1].date().isoformat(),
                "close": float(frame.close.iloc[i]),
            }
            for i in np.flatnonzero(close_run.to_numpy())[:12]
        ],
        "zero_volume": int((frame.volume == 0).sum()),
        "negative_volume": int((frame.volume < 0).sum()),
        "adj_close_equals_close": adjusted_equal,
        "max_calendar_gap_days": int(date_diff.max()),
        "gaps_over_four_days": [
            {
                "previous": frame.date.iloc[i - 1].date().isoformat(),
                "next": frame.date.iloc[i].date().isoformat(),
                "days": int(date_diff.iloc[i]),
            }
            for i in np.flatnonzero((date_diff > 4).fillna(False).to_numpy())
        ],
        "largest_absolute_log_return": [
            {
                "date": frame.date.iloc[i].date().isoformat(),
                "log_return": float(returns.iloc[i]),
            }
            for i in returns.abs().nlargest(8).index
        ],
    }
    return frame, checks


def holm(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    raw = np.where(np.isfinite(raw), raw, 1.0)
    order = np.argsort(raw, kind="stable")
    adjusted_sorted = np.maximum.accumulate(raw[order] * (len(raw) - np.arange(len(raw))))
    adjusted = np.empty_like(raw)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def hac_and_pvalue(
    difference: np.ndarray, positions: np.ndarray, bandwidth: int
) -> tuple[float, float]:
    """Bartlett HAC standard error and normal-reference two-sided p value."""
    d = np.asarray(difference, dtype=float)
    pos = np.asarray(positions, dtype=int)
    n = len(d)
    centered = d - np.mean(d)
    long_run = float(centered @ centered / n)
    for lag in range(1, bandwidth + 1):
        valid = pos[lag:] - pos[:-lag] == lag
        covariance_sum = np.sum(centered[lag:][valid] * centered[:-lag][valid]) / n
        long_run += 2 * (1 - lag / (bandwidth + 1)) * covariance_sum
    variance = max(long_run / n, 0.0)
    se = math.sqrt(variance)
    p = math.erfc(abs(float(np.mean(d)) / se) / math.sqrt(2)) if se else 1.0
    return se, p


def block_bootstrap_ci(
    difference: np.ndarray, positions: np.ndarray, length: int, replicates: int, seed: int
) -> list[float]:
    """Moving-block interval using explicit block slices, not prefix sums."""
    d = np.asarray(difference, dtype=float)
    pos = np.asarray(positions, dtype=int)
    starts = np.array(
        [i for i in range(len(d) - length + 1) if pos[i + length - 1] - pos[i] == length - 1]
    )
    if not len(starts):
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    count, remainder = divmod(len(d), length)
    chosen = rng.integers(0, len(starts), size=(replicates, count))
    cumulative = np.r_[0.0, np.cumsum(d)]
    block_starts = starts[chosen]
    boot = np.sum(cumulative[block_starts + length] - cumulative[block_starts], axis=1)
    if remainder:
        short_starts = np.array(
            [
                i
                for i in range(len(d) - remainder + 1)
                if pos[i + remainder - 1] - pos[i] == remainder - 1
            ]
        )
        chosen_short = rng.choice(short_starts, size=replicates)
        boot += cumulative[chosen_short + remainder] - cumulative[chosen_short]
    boot /= len(d)
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def load_primary_metric_series(
    predictions: pd.DataFrame,
    price_frames: dict[str, pd.DataFrame],
    *,
    dataset: str,
    horizon: int,
    model: str,
    field: str = "y_pred",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group = predictions.loc[
        predictions.dataset.eq(dataset)
        & predictions.horizon.eq(horizon)
        & predictions.model.eq(model)
    ].copy()
    prices = price_frames[dataset]
    indices = pd.Series(np.arange(len(prices)), index=prices.date)
    group["position"] = group.date.map(indices)
    group = group.sort_values("position")
    y = group.y_true.to_numpy(float)
    prediction = group[field].to_numpy(float)
    return y, prediction, group.position.to_numpy(int)


def test_family(
    family_name: str,
    table_path: str,
    *,
    records: list[dict],
    expected_n: int,
    seed: int,
) -> dict:
    stored = pd.read_csv(ROOT / table_path)
    flag = "primary_family" if "primary_family" in stored.columns else "primary"
    stored = stored.loc[stored[flag].astype(bool)].copy()
    calculated = pd.DataFrame(records)
    keys = [
        key for key in calculated.columns if key in stored.columns and key not in {"difference"}
    ]
    keys = [
        key for key in keys if key in {"dataset", "horizon", "first", "second", "model", "nominal"}
    ]
    stored_keyed = stored.set_index(keys).sort_index()
    calculated_keyed = calculated.set_index(keys).sort_index()
    if len(stored) != expected_n or len(calculated) != expected_n:
        raise AssertionError(
            f"{family_name}: expected {expected_n}, got {len(stored)}/{len(calculated)}"
        )
    p_delta = float(np.max(np.abs(stored_keyed.p_value - calculated_keyed.p_value)))
    se_delta = float(
        np.max(np.abs(stored_keyed.hac_standard_error - calculated_keyed.hac_standard_error))
    )
    effect_delta = float(
        np.max(np.abs(stored_keyed.mean_loss_difference - calculated_keyed.mean_loss_difference))
    )
    holm_col = "holm_p_value" if "holm_p_value" in stored.columns else "holm_p"
    corrected = holm(calculated_keyed.p_value.to_numpy())
    holm_delta = float(np.max(np.abs(stored_keyed[holm_col].to_numpy() - corrected)))
    ci_delta = None
    if "ci_lower" in stored.columns:
        intervals = [
            block_bootstrap_ci(
                row.difference.iloc[0],
                row.positions.iloc[0],
                20,
                2000,
                seed,
            )
            for _, row in calculated.groupby(keys, sort=True)
        ]
        endpoints = np.asarray(intervals)
        ci_delta = float(
            max(
                np.max(np.abs(stored_keyed.ci_lower.to_numpy() - endpoints[:, 0])),
                np.max(np.abs(stored_keyed.ci_upper.to_numpy() - endpoints[:, 1])),
            )
        )
    stored_sig = stored.loc[stored.mean_loss_difference.lt(0) & stored[holm_col].lt(0.05)]
    calc_sig_count = int(
        np.sum((calculated_keyed.mean_loss_difference.to_numpy() < 0) & (corrected < 0.05))
    )
    return {
        "name": family_name,
        "source": table_path,
        "family_size": expected_n,
        "unique_comparison_keys": len(stored_keyed),
        "max_abs_effect_difference": effect_delta,
        "max_abs_hac_se_difference": se_delta,
        "max_abs_nominal_p_difference": p_delta,
        "max_abs_holm_p_difference": holm_delta,
        "max_abs_moving_block_ci_endpoint_difference": ci_delta,
        "negative_effect_holm_significant_count": calc_sig_count,
        "saved_significant_count": int(len(stored_sig)),
        "saved_significant_cells": stored_sig[keys].to_dict("records"),
        "status": "matched"
        if max(effect_delta, se_delta, p_delta, holm_delta) < 1e-10
        and (ci_delta is None or ci_delta < 1e-10)
        else "discrepancy",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "independent_recalculation.json")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"preserve existing audit output: {output}")

    price_frames: dict[str, pd.DataFrame] = {}
    dataset_reports = {}
    for name in DATASETS:
        price_frames[name], dataset_reports[name] = dataset_audit(name)

    prediction_path = ROOT / "results/walkforward/full/predictions.csv.gz"
    predictions = pd.read_csv(prediction_path)
    predictions["date"] = pd.to_datetime(predictions.date)
    target_errors: list[float] = []
    target_date_errors = 0
    predictions_by_market = {}
    for dataset, frame in price_frames.items():
        part = predictions.loc[predictions.dataset.eq(dataset)].copy()
        positions_by_date = pd.Series(np.arange(len(frame)), index=frame.date)
        origin = part.date.map(positions_by_date)
        invalid_origin = origin.isna().to_numpy()
        if invalid_origin.any():
            raise AssertionError(f"{dataset}: prediction dates absent from input")
        origin = origin.to_numpy(dtype=int)
        horizon = part.horizon.to_numpy(dtype=int)
        endpoint = origin + horizon
        valid = endpoint < len(frame)
        if not valid.all():
            raise AssertionError(f"{dataset}: forecast endpoint beyond input")
        expected_target = np.log(frame.close.to_numpy()[endpoint] / frame.close.to_numpy()[origin])
        difference = np.abs(expected_target - part.y_true.to_numpy(float))
        target_errors.extend(difference.tolist())
        endpoints = frame.date.to_numpy()[endpoint]
        # There is no endpoint column in the core prediction CSV; each expected
        # endpoint is checked against the horizon's row-step construction.
        target_date_errors += int(np.sum(np.asarray(endpoints) <= part.date.to_numpy()))
        part["position"] = origin
        predictions_by_market[dataset] = part

    predictions["position"] = predictions.apply(
        lambda row: int(price_frames[row.dataset].date.searchsorted(row.date)), axis=1
    )
    key = ["dataset", "horizon", "model", "fold", "date"]
    duplicate_prediction_keys = int(predictions.duplicated(key).sum())
    common_sample_failures = 0
    for _, block in predictions.groupby(["dataset", "horizon", "fold"]):
        reference = block.loc[block.model.eq(MODELS[0])].sort_values("date")
        for model in MODELS[1:]:
            candidate = block.loc[block.model.eq(model)].sort_values("date")
            if not reference.date.reset_index(drop=True).equals(
                candidate.date.reset_index(drop=True)
            ):
                common_sample_failures += 1
            if not np.allclose(
                reference.y_true.to_numpy(), candidate.y_true.to_numpy(), rtol=0, atol=1e-12
            ):
                common_sample_failures += 1

    fold_markers = sorted((ROOT / "results/walkforward/full/folds").rglob("fold_*.json"))
    boundary_failures = 0
    per_cell_folds = Counter()
    fold_execution = Counter()
    for marker in fold_markers:
        info = json.loads(marker.read_text())
        sample = info["sample"]
        market = price_frames[info["dataset"]]
        idx = pd.Series(np.arange(len(market)), index=market.date)
        train_end = int(idx[pd.Timestamp(sample["train"]["end_date"])])
        validation_start = int(idx[pd.Timestamp(sample["validation"]["start_date"])])
        validation_end = int(idx[pd.Timestamp(sample["validation"]["end_date"])])
        test_start = int(idx[pd.Timestamp(sample["test"]["start_date"])])
        h = int(info["horizon"])
        if train_end + h >= validation_start or validation_end + h >= test_start:
            boundary_failures += 1
        actual_test_positions = predictions.loc[
            predictions.dataset.eq(info["dataset"])
            & predictions.horizon.eq(h)
            & predictions.fold.eq(info["fold"])
            & predictions.model.eq("zero_return"),
            "position",
        ].sort_values()
        expected_test_positions = np.arange(test_start, test_start + int(sample["test"]["n"]))
        if not np.array_equal(actual_test_positions.to_numpy(dtype=int), expected_test_positions):
            boundary_failures += 1
        per_cell_folds[(info["dataset"], h)] += 1
        fold_execution[
            "execution_metadata" if "execution" in info else "recovered_without_execution_metadata"
        ] += 1

    metrics = (
        predictions.assign(
            error=lambda x: x.y_pred - x.y_true,
            squared_error=lambda x: (x.y_pred - x.y_true) ** 2,
            absolute_error=lambda x: (x.y_pred - x.y_true).abs(),
        )
        .groupby(["dataset", "horizon", "model"], sort=True)
        .agg(
            n=("y_true", "size"),
            mse=("squared_error", "mean"),
            mae=("absolute_error", "mean"),
            bias=("error", "mean"),
            directional_accuracy=("error", lambda errors: np.nan),
        )
        .reset_index()
    )
    # Recompute directional accuracy from prediction and target signs explicitly.
    direction = (
        predictions.assign(
            directional_match=lambda x: np.where(
                x.y_true.ne(0), np.sign(x.y_pred) == np.sign(x.y_true), np.nan
            )
        )
        .groupby(["dataset", "horizon", "model"], sort=True)
        .directional_match.mean()
    )
    metrics["directional_accuracy"] = [
        direction.loc[(row.dataset, row.horizon, row.model)] for row in metrics.itertuples()
    ]
    metrics["rmse"] = np.sqrt(metrics.mse)
    for (dataset, horizon), indices in metrics.groupby(["dataset", "horizon"]).groups.items():
        cell = metrics.loc[indices].set_index("model")
        zero_mse = float(cell.loc["zero_return", "mse"])
        mean_mse = float(cell.loc["historical_mean", "mse"])
        metrics.loc[indices, "skill_vs_zero"] = 1 - metrics.loc[indices, "mse"] / zero_mse
        metrics.loc[indices, "skill_vs_historical_mean"] = (
            1 - metrics.loc[indices, "mse"] / mean_mse
        )
        order = cell.rmse.sort_values(kind="stable").index.tolist()
        for rank, model in enumerate(order, start=1):
            metrics.loc[
                (metrics.dataset.eq(dataset))
                & (metrics.horizon.eq(horizon))
                & (metrics.model.eq(model)),
                "rank",
            ] = rank
    saved_metrics = pd.read_csv(ROOT / "results/walkforward/full/pooled_metrics.csv")
    metrics_join = metrics.merge(
        saved_metrics,
        on=["dataset", "horizon", "model"],
        suffixes=("_audit", "_saved"),
        validate="one_to_one",
    )
    metric_max_error = {}
    for metric in [
        "n",
        "rmse",
        "mae",
        "bias",
        "directional_accuracy",
        "skill_vs_zero",
        "skill_vs_historical_mean",
        "rank",
    ]:
        metric_max_error[metric] = float(
            np.nanmax(np.abs(metrics_join[f"{metric}_audit"] - metrics_join[f"{metric}_saved"]))
        )

    original_config = tomllib.loads((ROOT / "configs/robustness.toml").read_text())
    original_records = []
    for dataset in sorted(predictions.dataset.unique()):
        for horizon in [1, 5, 20]:
            base = predictions.loc[
                predictions.dataset.eq(dataset) & predictions.horizon.eq(horizon)
            ]
            base = base.drop_duplicates("date").sort_values("date")
            positions = base.position.to_numpy(int)
            actual = base.y_true.to_numpy(float)
            models_by_name = {
                model: predictions.loc[
                    predictions.dataset.eq(dataset)
                    & predictions.horizon.eq(horizon)
                    & predictions.model.eq(model)
                ].sort_values("position")
                for model in MODELS
            }
            for first, second in original_config["primary_pairs"]:
                a = models_by_name[first].sort_values("position")
                b = models_by_name[second].sort_values("position")
                diff = (a.y_pred.to_numpy() - a.y_true.to_numpy()) ** 2 - (
                    b.y_pred.to_numpy() - b.y_true.to_numpy()
                ) ** 2
                if len(diff) != len(positions):
                    raise AssertionError("original primary pair has nonmatching population")
                se, pvalue = hac_and_pvalue(diff, positions, 19)
                ci = block_bootstrap_ci(
                    diff, positions, 20, 2000, int(original_config["bootstrap_seed"])
                )
                original_records.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "first": first,
                        "second": second,
                        "difference": diff,
                        "positions": positions,
                        "mean_loss_difference": float(diff.mean()),
                        "hac_standard_error": se,
                        "p_value": pvalue,
                        "ci_lower": ci[0],
                        "ci_upper": ci[1],
                    }
                )
    original_family = test_family(
        "original native MSE",
        "results/statistical_reliability/comparisons.csv",
        records=original_records,
        expected_n=18,
        seed=int(original_config["bootstrap_seed"]),
    )

    expanded_records = []
    seed_losses = pd.read_csv(ROOT / "results/phase10/matched_losses.csv.gz")
    seed_losses["date"] = pd.to_datetime(seed_losses.date)
    config10 = json.loads((ROOT / "configs/phase10.json").read_text())
    for dataset in sorted(seed_losses.dataset.unique()):
        for horizon in [1, 5, 20]:
            cell = seed_losses.loc[
                seed_losses.dataset.eq(dataset) & seed_losses.horizon.eq(horizon)
            ]
            models_by_name = {
                m: cell.loc[cell.model.eq(m)].sort_values("date")
                for m in seed_losses.model.unique()
            }
            for first, second in config10["primary_pairs"]:
                a, b = models_by_name[first], models_by_name[second]
                if not a.date.reset_index(drop=True).equals(b.date.reset_index(drop=True)):
                    raise AssertionError("expanded paired loss dates differ")
                diff = a.mean_seed_loss.to_numpy(float) - b.mean_seed_loss.to_numpy(float)
                positions = a.origin_position.to_numpy(int)
                se, pvalue = hac_and_pvalue(diff, positions, 19)
                ci = block_bootstrap_ci(diff, positions, 20, 2000, int(config10["bootstrap_seed"]))
                expanded_records.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "first": first,
                        "second": second,
                        "difference": diff,
                        "positions": positions,
                        "mean_loss_difference": float(diff.mean()),
                        "hac_standard_error": se,
                        "p_value": pvalue,
                        "ci_lower": ci[0],
                        "ci_upper": ci[1],
                    }
                )
    expanded_family = test_family(
        "expanded five-seed MSE sensitivity",
        "results/phase10/comparisons.csv",
        records=expanded_records,
        expected_n=30,
        seed=int(config10["bootstrap_seed"]),
    )

    adaptive_config = json.loads((ROOT / "configs/phase11.json").read_text())
    trailing_root = ROOT / "results/calibration/records"
    adaptive_root = ROOT / "results/phase11/records"
    adaptive_records = []
    for dataset in sorted(predictions.dataset.unique()):
        slug = "nasdaq-100" if dataset == "NASDAQ-100" else "sandp_500"
        for horizon in [1, 5, 20]:
            for model in MODELS:
                stem = f"{slug}_h{horizon}_{model}"
                adap = pd.read_csv(
                    adaptive_root / f"{stem}_{adaptive_config['primary_variant']}.csv.gz"
                )
                ref = pd.read_csv(trailing_root / f"{stem}_trailing.csv.gz")
                if not adap.date.equals(ref.date) or not np.array_equal(
                    adap.eligible, ref.eligible
                ):
                    raise AssertionError("adaptive/trailing eligibility or origins differ")
                eligible = adap.eligible.to_numpy(bool)
                for level in adaptive_config["levels"]:
                    suffix = str(round(100 * level))
                    actual = ref.y_true.to_numpy(float)[eligible]
                    lo_a, hi_a = (
                        adap[f"lower_{suffix}"].to_numpy(float)[eligible],
                        adap[f"upper_{suffix}"].to_numpy(float)[eligible],
                    )
                    lo_b, hi_b = (
                        ref[f"lower_{suffix}"].to_numpy(float)[eligible],
                        ref[f"upper_{suffix}"].to_numpy(float)[eligible],
                    )
                    alpha = 1 - level
                    score_a = (
                        hi_a
                        - lo_a
                        + 2 / alpha * (np.maximum(lo_a - actual, 0) + np.maximum(actual - hi_a, 0))
                    )
                    score_b = (
                        hi_b
                        - lo_b
                        + 2 / alpha * (np.maximum(lo_b - actual, 0) + np.maximum(actual - hi_b, 0))
                    )
                    diff = score_a - score_b
                    positions = adap.origin_position.to_numpy(int)[eligible]
                    se, pvalue = hac_and_pvalue(diff, positions, 19)
                    ci = block_bootstrap_ci(
                        diff, positions, 20, 2000, int(adaptive_config["bootstrap_seed"])
                    )
                    adaptive_records.append(
                        {
                            "dataset": dataset,
                            "horizon": horizon,
                            "model": model,
                            "nominal": level,
                            "difference": diff,
                            "positions": positions,
                            "mean_loss_difference": float(diff.mean()),
                            "hac_standard_error": se,
                            "p_value": pvalue,
                            "ci_lower": ci[0],
                            "ci_upper": ci[1],
                        }
                    )
    adaptive_family = test_family(
        "adaptive interval score",
        "results/phase11/score_comparisons.csv",
        records=adaptive_records,
        expected_n=60,
        seed=int(adaptive_config["bootstrap_seed"]),
    )

    nesting = pd.read_csv(ROOT / "results/phase11/diagnostics.csv")
    nesting_rows = []
    selected_eligible_total = selected_violation_total = 0
    first_violation_schema = None
    for row in nesting.itertuples(index=False):
        slug = "nasdaq-100" if row.dataset == "NASDAQ-100" else "sandp_500"
        path = adaptive_root / f"{slug}_h{row.horizon}_{row.model}_{row.variant}.csv.gz"
        frame = pd.read_csv(path)
        ok = frame.eligible.to_numpy(bool)
        lower80, upper80 = frame.lower_80.to_numpy(float), frame.upper_80.to_numpy(float)
        lower90, upper90 = frame.lower_90.to_numpy(float), frame.upper_90.to_numpy(float)
        violation = ok & ((lower80 < lower90) | (upper80 > upper90))
        count = int(violation.sum())
        denominator = int(ok.sum())
        if count != int(row.nesting_violations) or denominator != int(row.eligible):
            raise AssertionError("nesting diagnostics disagree with stored bounds")
        nesting_rows.append(
            {
                "dataset": row.dataset,
                "horizon": int(row.horizon),
                "model": row.model,
                "variant": row.variant,
                "eligible": denominator,
                "violations": count,
            }
        )
        if row.variant == adaptive_config["primary_variant"]:
            selected_eligible_total += denominator
            selected_violation_total += count
        if count and first_violation_schema is None:
            first_violation_schema = {
                "dataset": row.dataset,
                "horizon": int(row.horizon),
                "model": row.model,
                "variant": row.variant,
                "eligible": denominator,
                "violations": count,
                "first_date": frame.loc[np.flatnonzero(violation)[0], "date"],
            }

    seed_files = []
    seed_targets_valid = 0
    seed_rows = 0
    task_counts = {}
    seed_set = set()
    for directory in [
        ROOT / "results/robustness/multiseed/runs",
        ROOT / "results/phase10/multiseed/runs",
    ]:
        files = sorted(directory.glob("*.csv.gz"))
        task_counts[str(directory.relative_to(ROOT))] = len(files)
        for path in files:
            frame = pd.read_csv(path)
            if len(frame) != 504 or frame.model.nunique() != 2 or frame.date.nunique() != 252:
                raise AssertionError(f"unexpected seed-run shape: {path}")
            seed = int(frame.seed.iloc[0])
            if frame.seed.nunique() != 1:
                raise AssertionError(f"mixed seed in {path}")
            seed_set.add(seed)
            market = price_frames[frame.dataset.iloc[0]]
            date_index = pd.Series(np.arange(len(market)), index=market.date)
            starts = frame.date.map(date_index).to_numpy(dtype=int)
            end = starts + int(frame.horizon.iloc[0])
            expected = np.log(market.close.to_numpy()[end] / market.close.to_numpy()[starts])
            err = np.abs(expected - frame.y_true.to_numpy(float))
            if err.max(initial=0) > 1e-12:
                raise AssertionError(f"seed output target mismatch: {path}")
            seed_targets_valid += len(frame)
            seed_rows += len(frame)
            seed_files.append(path)

    rerun_path = ROOT / "results/final_evidence_authenticity_audit/reruns/predictions.csv"
    rerun = pd.read_csv(rerun_path)
    rerun["date"] = pd.to_datetime(rerun.date)
    historic = predictions.copy()
    compare_keys = ["dataset", "horizon", "model", "date"]
    rerun_merge = rerun.merge(
        historic, on=compare_keys, suffixes=("_new", "_old"), validate="one_to_one"
    )
    rerun_merge = rerun_merge.loc[rerun_merge.fold_new.eq(rerun_merge.fold_old)]
    rerun_by_model = (
        rerun_merge.groupby(["dataset", "horizon", "model"], sort=True)
        .agg(
            n=("y_pred_new", "size"),
            max_abs_prediction_difference=(
                "y_pred_new",
                lambda x: float(
                    np.max(np.abs(x.to_numpy() - rerun_merge.loc[x.index, "y_pred_old"].to_numpy()))
                ),
            ),
            new_rmse=(
                "y_true_new",
                lambda x: float(
                    np.sqrt(
                        np.mean(
                            (rerun_merge.loc[x.index, "y_pred_new"].to_numpy() - x.to_numpy()) ** 2
                        )
                    )
                ),
            ),
        )
        .reset_index()
    )

    report = {
        "scope": (
            "Independent arithmetic and structural checks implemented with direct pandas/NumPy; "
            "this report does not prove original data acquisition or "
            "historical execution provenance."
        ),
        "repository_commit": "60e77222a467e54d3fa28cbfda54970a5252079f",
        "datasets": dataset_reports,
        "forecast_records": {
            "source": "results/walkforward/full/predictions.csv.gz",
            "sha256": sha256(prediction_path),
            "rows": int(len(predictions)),
            "target_rows_checked": int(len(target_errors)),
            "target_endpoint_date_order_errors": target_date_errors,
            "maximum_absolute_target_recalculation_error": float(max(target_errors)),
            "duplicate_model_fold_origin_keys": duplicate_prediction_keys,
            "common_population_failures": common_sample_failures,
        },
        "folds": {
            "marker_count": len(fold_markers),
            "folds_by_dataset_horizon": [
                {"dataset": key[0], "horizon": key[1], "folds": value}
                for key, value in sorted(per_cell_folds.items())
            ],
            "execution_marker_classes": dict(fold_execution),
            "chronology_or_test_boundary_failures": boundary_failures,
        },
        "native_metrics": {
            "rows_recalculated": int(len(metrics)),
            "maximum_absolute_differences_from_saved_pooled_metrics": metric_max_error,
        },
        "statistical_families": [original_family, expanded_family, adaptive_family],
        "adaptive_nesting": {
            "diagnostic_rows_recalculated_from_saved_bounds": len(nesting_rows),
            "designated_variant_eligible_pairs": selected_eligible_total,
            "designated_variant_crossings": selected_violation_total,
            "primary_variant": adaptive_config["primary_variant"],
            "first_any_variant_example": first_violation_schema,
        },
        "seed_artifacts": {
            "run_file_counts_by_directory": task_counts,
            "seed_values": sorted(seed_set),
            "prediction_rows_checked_against_raw_price_targets": seed_targets_valid,
            "model_run_files": len(seed_files),
        },
        "current_code_rerun": {
            "fold_tasks": int(rerun[["dataset", "horizon", "fold"]].drop_duplicates().shape[0]),
            "forecast_records": int(len(rerun)),
            "matched_historical_records": int(len(rerun_merge)),
            "by_dataset_horizon_model": rerun_by_model.to_dict("records"),
            "execution_metadata": json.loads(
                (
                    ROOT / "results/final_evidence_authenticity_audit/reruns/manifest.json"
                ).read_text()
            )["recovery_environment"],
        },
    }

    if (
        boundary_failures
        or common_sample_failures
        or duplicate_prediction_keys
        or target_date_errors
    ):
        raise AssertionError(
            "structural or alignment check failed: "
            f"boundaries={boundary_failures}, common_samples={common_sample_failures}, "
            f"duplicate_keys={duplicate_prediction_keys}, target_dates={target_date_errors}"
        )
    if max(target_errors) > 1e-12 or max(metric_max_error.values()) > 1e-9:
        raise AssertionError("target or metric discrepancy exceeded tolerance")
    if selected_violation_total != 1494 or selected_eligible_total != 347500:
        raise AssertionError("designated nesting totals differ from the reported denominator/count")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "targets": len(target_errors),
                "metrics": len(metrics),
                "folds": len(fold_markers),
                "families": [18, 30, 60],
                "designated_nesting": [selected_violation_total, selected_eligible_total],
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
