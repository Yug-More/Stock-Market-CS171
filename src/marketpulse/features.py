"""Causal features known at the forecast origin.

The forecast is assumed to be made after the closing price at origin t is
known. Every window ends at t. Volume is not a feature: the S&P 500 file
contains thousands of leading zeros that are not documented as real trading
volume, so a volume feature would not mean the same thing on both indexes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "ret_0",
    "ret_1",
    "ret_5",
    "mom_5",
    "mom_20",
    "roll_mean_5",
    "roll_mean_20",
    "roll_vol_20",
    "ma_distance_20",
    "rsi_14",
)

_RSI_WINDOW = 14
_ROLLING_STD_DDOF = 1


@dataclass(frozen=True)
class FeatureBuild:
    """Feature rows aligned to the input market frame, including warmup gaps."""

    frame: pd.DataFrame
    columns: tuple[str, ...]


def build_features(market: pd.DataFrame) -> FeatureBuild:
    """Build trailing features. Rows that lack history stay missing.

    ``market`` must contain ``close`` and keep trading-session order. The
    function does not drop or fill rows and does not read future prices.
    """

    if "close" not in market.columns:
        raise ValueError("market frame must contain a close column")
    close = pd.to_numeric(market["close"], errors="coerce")
    if bool((close.notna() & (close <= 0)).any()):
        raise ValueError("close must be positive wherever it is observed")
    log_return = np.log(close / close.shift(1))
    features = pd.DataFrame(index=market.index)
    features["ret_0"] = log_return
    features["ret_1"] = log_return.shift(1)
    features["ret_5"] = log_return.shift(5)
    features["mom_5"] = np.log(close / close.shift(5))
    features["mom_20"] = np.log(close / close.shift(20))
    features["roll_mean_5"] = log_return.rolling(5, min_periods=5, center=False).mean()
    features["roll_mean_20"] = log_return.rolling(20, min_periods=20, center=False).mean()
    features["roll_vol_20"] = log_return.rolling(20, min_periods=20, center=False).std(
        ddof=_ROLLING_STD_DDOF
    )
    trailing_mean = close.rolling(20, min_periods=20, center=False).mean()
    features["ma_distance_20"] = np.log(close / trailing_mean)
    features["rsi_14"] = _relative_strength_index(close, _RSI_WINDOW)
    ordered = features.loc[:, list(FEATURE_COLUMNS)]
    return FeatureBuild(frame=ordered, columns=FEATURE_COLUMNS)


def _relative_strength_index(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(window, min_periods=window, center=False).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window, min_periods=window, center=False).mean()
    rsi = pd.Series(np.nan, index=close.index, dtype=float)
    both_zero = gain.notna() & loss.notna() & (gain == 0.0) & (loss == 0.0)
    gain_only = gain.notna() & loss.notna() & (loss == 0.0) & (gain > 0.0)
    defined = gain.notna() & loss.notna() & (loss > 0.0)
    relative_strength = gain.loc[defined] / loss.loc[defined]
    rsi.loc[defined] = 100.0 - (100.0 / (1.0 + relative_strength))
    rsi.loc[gain_only] = 100.0
    rsi.loc[both_zero] = np.nan
    rsi.name = f"rsi_{window}"
    return rsi
