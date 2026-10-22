"""Load the machine-readable forecasting protocol.

The file records initial research choices. It does not mean those experiments
have been run.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

_PROTOCOL_FILE = Path("configs") / "forecasting.toml"
_REQUIRED_KEYS = {
    "schema_version",
    "primary_target",
    "horizons",
    "seed",
    "minimum_training_origins",
    "validation_window",
    "test_window",
    "step",
    "nominal_coverage",
}


@dataclass(frozen=True)
class ForecastingProtocol:
    """Initial walk-forward settings for later experiments."""

    schema_version: int
    primary_target: str
    horizons: tuple[int, ...]
    seed: int
    minimum_training_origins: int
    validation_window: int
    test_window: int
    step: int
    nominal_coverage: float


def load_forecasting_protocol(root: Path) -> ForecastingProtocol:
    """Read ``configs/forecasting.toml`` relative to ``root``."""

    path = root / _PROTOCOL_FILE
    if not path.is_file():
        raise ValueError(f"Forecasting protocol does not exist: {path}")
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    unknown = set(payload) - _REQUIRED_KEYS
    missing = _REQUIRED_KEYS - set(payload)
    if unknown or missing:
        raise ValueError(
            f"Forecasting protocol keys do not match the schema. "
            f"Missing: {sorted(missing)}. Unknown: {sorted(unknown)}."
        )
    raw_horizons = payload["horizons"]
    if not isinstance(raw_horizons, list) or not raw_horizons:
        raise ValueError("horizons must be a non-empty list")
    horizons = tuple(_positive_int(value, "horizon") for value in raw_horizons)
    if len(horizons) != len(set(horizons)):
        raise ValueError("horizons must be unique")
    coverage = payload["nominal_coverage"]
    if isinstance(coverage, bool) or not isinstance(coverage, int | float):
        raise ValueError("nominal_coverage must be a number strictly between 0 and 1")
    coverage_value = float(coverage)
    if not 0.0 < coverage_value < 1.0:
        raise ValueError("nominal_coverage must be strictly between 0 and 1")
    return ForecastingProtocol(
        schema_version=_positive_int(payload["schema_version"], "schema_version"),
        primary_target=_text(payload["primary_target"], "primary_target"),
        horizons=horizons,
        seed=_nonnegative_int(payload["seed"], "seed"),
        minimum_training_origins=_positive_int(
            payload["minimum_training_origins"], "minimum_training_origins"
        ),
        validation_window=_positive_int(payload["validation_window"], "validation_window"),
        test_window=_positive_int(payload["test_window"], "test_window"),
        step=_positive_int(payload["step"], "step"),
        nominal_coverage=coverage_value,
    )


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value
