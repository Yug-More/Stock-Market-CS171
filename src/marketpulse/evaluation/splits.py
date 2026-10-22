"""Index arithmetic that keeps multi-horizon labels out of later windows.

A label at origin t uses the close at index t + horizon. If an evaluation
window starts at ``evaluation_start``, any origin with t + horizon >=
evaluation_start reads a price inside that window. Those origins are not
valid training rows.
"""

from __future__ import annotations


def exclusive_training_end(evaluation_start: int, horizon: int) -> int:
    """Return the exclusive end index of leakage-safe training origins.

    Training origins are ``range(exclusive_training_end)``. The last included
    origin uses a future close that is still strictly before
    ``evaluation_start``.
    """

    if isinstance(evaluation_start, bool) or not isinstance(evaluation_start, int):
        raise ValueError("evaluation_start must be an integer index")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if evaluation_start <= horizon:
        raise ValueError(
            "evaluation_start must be greater than the horizon so at least one "
            "training label is free of the evaluation window"
        )
    return evaluation_start - horizon
