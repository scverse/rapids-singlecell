from __future__ import annotations

import numpy as np
from numpy.testing import assert_array_equal
from scanpy.preprocessing._utils import sample_comb
from sklearn.random_projection import sample_without_replacement

from rapids_singlecell._compat import _rng_kwargs
from rapids_singlecell._utils._random import _LegacyRng


def _legacy_paga_init(*, random_state) -> np.ndarray:
    """Model Scanpy <1.13 reseeding NumPy's global RNG from a scalar."""
    return np.random.RandomState(random_state).random(20)


def test_sample_comb_rng_kwargs_advances_legacy_state() -> None:
    expected_rng = np.random.RandomState(0)
    flat_idx = sample_without_replacement(100, 20, random_state=expected_rng)
    expected_pairs = np.vstack(np.unravel_index(flat_idx, (10, 10))).T
    expected_draw = expected_rng.binomial(10, 0.5, size=20)

    rng = _LegacyRng(0)
    actual_pairs = sample_comb(
        (10, 10), 20, **_rng_kwargs(sample_comb, rng, always_state=True)
    )
    actual_draw = rng.binomial(10, 0.5, size=20)

    assert_array_equal(actual_pairs, expected_pairs)
    assert_array_equal(actual_draw, expected_draw)


def test_rng_kwargs_preserves_legacy_seed() -> None:
    expected = np.random.RandomState(0).random(20)

    rng = _LegacyRng(0)
    actual = _legacy_paga_init(**_rng_kwargs(_legacy_paga_init, rng))

    assert_array_equal(actual, expected)
    assert_array_equal(rng.random(20), expected)
