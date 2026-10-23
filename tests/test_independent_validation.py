import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marketpulse.experiments.independent_validation import four_blocks, prospective_gate


@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_four_disjoint_mature_blocks_with_internal_gaps(horizon):
    origins = np.delete(np.arange(6000), np.arange(1100, 1160))
    split, cal, test = four_blocks(origins, horizon)
    blocks = [split.train, split.validation, cal, test]
    assert len(split.train) >= 2520
    assert [len(x) for x in blocks[1:]] == [252, 252, 252]
    for a, b in zip(blocks, blocks[1:], strict=False):
        assert max(a + horizon) < min(b)
        assert not set(a) & set(b)
    assert np.array_equal(split.test, np.r_[cal, test])


def test_insufficient_history_and_reused_holdout_rejected():
    with pytest.raises(ValueError):
        four_blocks(np.arange(3000), 20)
    protocol = json.loads(
        (Path(__file__).parents[1] / "configs/independent_validation.json").read_text()
    )
    manifest = dict.fromkeys(
        ["source_url", "version", "retrieved_utc", "license", "adjustments", "sha256"], "recorded"
    )
    manifest.update(retrieved_utc="2026-10-23T00:00:00Z", prior_outcome_exposure="none declared")
    dates = pd.Series(pd.date_range("2024-01-01", periods=400))
    with pytest.raises(ValueError, match="historical"):
        prospective_gate(dates, np.arange(252), manifest, protocol)
    manifest.pop("license")
    with pytest.raises(ValueError, match="provenance"):
        prospective_gate(dates, np.arange(252), manifest, protocol)
