"""Block and row-tile boundaries for the native Harmony column reductions."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

from rapids_singlecell._cuda import _harmony_colsum_cuda


@pytest.mark.parametrize("shape", [(0, 1), (1, 33), (33, 1), (1025, 33), (8193, 7)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
@pytest.mark.parametrize("accumulate", [False, True])
def test_column_sum_block_and_row_tile_boundaries(shape, dtype, accumulate):
    rng = np.random.default_rng(15)
    if dtype == np.int32:
        values = rng.integers(-(2**30), 2**30, size=shape, dtype=dtype)
        expected = values.sum(axis=0, dtype=np.int64).astype(dtype)
    else:
        values = rng.normal(size=shape).astype(dtype)
        expected = values.sum(axis=0, dtype=np.float64)
    initial = np.full(shape[1], 7, dtype=dtype)
    if accumulate:
        expected = (expected + initial).astype(dtype)
    stream = cp.cuda.Stream(non_blocking=True)
    function = (
        _harmony_colsum_cuda.colsum_atomic
        if accumulate
        else _harmony_colsum_cuda.colsum
    )
    with stream:
        data = cp.asarray(values)
        out = cp.asarray(initial)
        function(data, out=out, rows=shape[0], cols=shape[1], stream=stream.ptr)
    stream.synchronize()
    if dtype == np.int32:
        np.testing.assert_array_equal(out.get(), expected)
    else:
        tolerance = 2e-5 if dtype == np.float32 else 1e-12
        np.testing.assert_allclose(out.get(), expected, rtol=tolerance, atol=tolerance)
