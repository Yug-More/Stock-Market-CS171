"""Independent population and seed-selection invariants."""

import pandas as pd
import pytest

from marketpulse.experiments.multiseed import selected_fold_indices
from marketpulse.experiments.robustness_data import matched_membership, select_matched


def test_balanced_fold_selection_is_score_independent():
    assert selected_fold_indices(13, [0, 0.5, 1]) == [0, 6, 12]
    assert selected_fold_indices(81, [0, 0.5, 1]) == [0, 40, 80]
    with pytest.raises(ValueError):
        selected_fold_indices(0, [0])


def test_matching_distinguishes_origin_and_endpoint_populations():
    frame = pd.DataFrame(
        [
            {
                "dataset": "a",
                "horizon": 5,
                "model": "zero_return",
                "date": "2020-01-01",
                "target_date": "2020-01-08",
            },
            {
                "dataset": "b",
                "horizon": 5,
                "model": "zero_return",
                "date": "2020-01-01",
                "target_date": "2020-01-09",
            },
            {
                "dataset": "a",
                "horizon": 5,
                "model": "zero_return",
                "date": "2020-01-02",
                "target_date": "2020-01-09",
            },
        ]
    )
    membership = matched_membership(frame)
    assert membership.origin_match.sum() == 1
    assert membership.endpoint_match.sum() == 0
    assert len(select_matched(frame, membership, endpoints=False)) == 2
    assert select_matched(frame, membership, endpoints=True).empty
