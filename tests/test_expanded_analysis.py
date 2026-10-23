import numpy as np
import pandas as pd
import pytest

from marketpulse.experiments.expanded_analysis import matched_seed_losses
from marketpulse.experiments.multiseed import selected_fold_indices


def test_expansion_folds_do_not_reuse_original_selection():
    for count, expected in [(13, [3, 9]), (81, [20, 60])]:
        added = selected_fold_indices(count, [0.25, 0.75])
        assert added == expected
        assert not set(added) & set(selected_fold_indices(count, [0, 0.5, 1]))


def test_loss_averaging_is_not_ensemble_and_keeps_temporal_n():
    a = pd.DataFrame(
        {
            "dataset": ["index"],
            "horizon": [1],
            "model": ["lstm"],
            "date": ["2020-01-01"],
            "target_date": ["2020-01-02"],
            "origin_position": [1],
            "y_true": [0.0],
            "y_pred": [-2.0],
        }
    )
    b = a.copy()
    b["y_pred"] = 2.0
    result = matched_seed_losses([a, b])
    assert len(result) == 1
    assert result.mean_seed_loss.iloc[0] == 4.0
    assert np.mean([a.y_pred.iloc[0], b.y_pred.iloc[0]]) ** 2 == 0.0
    b["target_date"] = "2020-01-03"
    with pytest.raises(ValueError, match="endpoints"):
        matched_seed_losses([a, b])
