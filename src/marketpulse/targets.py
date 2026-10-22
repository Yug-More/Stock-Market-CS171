"""Forecast targets aligned to a trading-day origin.

The primary target is the forward log return

    r(t, h) = log(C(t + h) / C(t))

where t is the forecast origin and h is a number of rows in the supplied
series. Those rows are treated as trading days already, not calendar days.
The value is stored on row t. The last h rows are missing because the future
close is unknown. This function does not fill or drop those rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """Return r(t, h) aligned to each forecast origin.

    ``horizon`` must be a positive integer. Nonpositive closes raise, because
    a log return is undefined for them. Missing closes stay missing and are
    not interpolated.
    """

    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer number of trading-day steps")
    values = pd.to_numeric(close, errors="coerce")
    non_numeric = close.notna() & values.isna()
    if bool(non_numeric.any()):
        raise ValueError("close contains non-numeric values")
    if bool((values.notna() & (values <= 0)).any()):
        raise ValueError("close must be positive wherever it is observed")
    result = np.log(values.shift(-horizon) / values)
    result.name = f"forward_log_return_h{horizon}"
    return result
