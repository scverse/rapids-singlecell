from __future__ import annotations

import numpy as np
from numpy.testing import assert_array_equal

from rapids_singlecell._compat import _rng_kwargs
from rapids_singlecell._utils._random import _LegacyRng


def _legacy_paga_init(*, random_state) -> np.ndarray:
    """Model Scanpy <1.13 reseeding NumPy's global RNG from a scalar."""
    return np.random.RandomState(random_state).random(20)


def test_rng_kwargs_preserves_legacy_seed() -> None:
    expected = np.random.RandomState(0).random(20)

    rng = _LegacyRng(0)
    actual = _legacy_paga_init(**_rng_kwargs(_legacy_paga_init, rng))

    assert_array_equal(actual, expected)
    assert_array_equal(rng.random(20), expected)
