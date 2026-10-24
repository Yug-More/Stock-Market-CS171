import math

import numpy as np
import pytest

from marketpulse.scientific_review import holm, inference


def test_review_holm_stepdown_retains_running_maximum():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


@pytest.mark.parametrize(
    "positions,expected_se", [([0, 1, 2, 3], 0.25), ([0, 1, 5, 6], math.sqrt(0.125))]
)
def test_review_hac_excludes_products_across_a_structural_gap(positions, expected_se):
    mean, se, p, bounds = inference([1.0, 3.0, 1.0, 3.0], positions, length=2, replicates=10)
    assert mean == 2
    assert se == pytest.approx(expected_se)
    assert p == pytest.approx(math.erfc(abs(mean / se) / math.sqrt(2)))
    assert bounds[0] <= mean <= bounds[1]
