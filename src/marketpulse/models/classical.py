"""Linear regression and random forest fit only on training origins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from marketpulse.evaluation.metrics import rmse


@dataclass(frozen=True)
class ForestSelection:
    """The depth chosen by validation RMSE, with every candidate recorded."""

    model: Pipeline
    selected_max_depth: int
    validation_rmse_by_depth: dict[int, float]


def fit_linear_regression(x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Fit a standardized linear regression on the training partition."""

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline


def select_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    depths: tuple[int, ...],
    n_estimators: int,
    min_samples_leaf: int,
    seed: int,
) -> ForestSelection:
    """Fit each predeclared depth on training rows and keep the best validation RMSE.

    The test partition is not an argument. Scaling is fit inside the training
    pipeline. Ties resolve to the smaller depth.
    """

    if len(depths) == 0:
        raise ValueError("depths must contain at least one candidate")
    scores: dict[int, float] = {}
    fitted: dict[int, Pipeline] = {}
    for depth in depths:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=depth,
                        min_samples_leaf=min_samples_leaf,
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        scores[depth] = rmse(y_validation, model.predict(x_validation))
        fitted[depth] = model
    selected = min(scores, key=lambda depth: (scores[depth], depth))
    return ForestSelection(
        model=fitted[selected],
        selected_max_depth=selected,
        validation_rmse_by_depth=scores,
    )
