from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marketpulse.experiments.prospective_intake import frozen_window, prepare


def test_later_acquisition_cannot_slide_the_frozen_test_window():
    dates = pd.Series(pd.bdate_range("2000-01-03", periods=8000))
    origins = np.arange(40, len(dates) - 20)
    a, cal, test = frozen_window(origins, dates, 20, "2026-10-26")
    prefix = origins[origins <= test[-1] + 100]
    _, _, again = frozen_window(prefix, dates, 20, "2026-10-26")
    np.testing.assert_array_equal(test, again)
    assert dates.iloc[test[0]] == pd.Timestamp("2026-10-26")
    assert cal[-1] + 20 < test[0]
    assert a.validation[-1] + 20 < cal[0]
    with pytest.raises(ValueError, match="cannot shift"):
        frozen_window(origins[origins != test[0]], dates, 20, "2026-10-26")
    with pytest.raises(ValueError, match="not yet mature"):
        frozen_window(origins[:4000], dates, 20, "2026-10-26")


def test_incomplete_or_changed_freeze_rejected_before_loading_or_fitting():
    root = Path(__file__).parents[1]
    with pytest.raises(ValueError, match="original freeze"):
        prepare(root, {"protocol_sha256": "changed", "datasets": []})
