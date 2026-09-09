"""Native replacements for custom elementwise kernels retain numerical behavior."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

from rapids_singlecell.decoupler_gpu._helper._data import getnnz_0
from rapids_singlecell.get._kernels._aggr_elementwise import (
    _scatter_count_nonzero,
    _scatter_mean_var,
    _scatter_sum,
    _sum_duplicates_assign,
    _sum_duplicates_diff,
)
from rapids_singlecell.preprocessing._hvg._seurat_v3 import _clip_square_sum_sparse
from rapids_singlecell.preprocessing._kernels._mean_var_kernel import mean_sum, sq_sum
from rapids_singlecell.preprocessing._sparse_pca._svd_lanczos import _kernel_axpy


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("shape", [(51, 37), (513, 769)])
def test_dense_moments(dtype, order, axis, shape):
    rng = np.random.default_rng(42)
    data = cp.asarray(rng.standard_normal(shape), dtype=dtype, order=order)
    cp.testing.assert_allclose(
        mean_sum(data, axis=axis),
        data.sum(axis=axis, dtype=cp.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    cp.testing.assert_allclose(
        sq_sum(data, axis=axis),
        cp.square(data).sum(axis=axis, dtype=cp.float64),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lanczos_axpy_borrows_device_scalar(dtype):
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        alpha = cp.asarray(2.0, dtype=dtype)
        y = cp.arange(1000, dtype=dtype)
        x = cp.full(1000, 4, dtype=dtype)
        assert _kernel_axpy(alpha, y, x) is x
    stream.synchronize()
    cp.testing.assert_array_equal(x, 4 - 2 * y)


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_duplicate_and_scatter_primitives(index_dtype):
    row = cp.asarray([0, 0, 0, 1, 1, 2], dtype=index_dtype)
    col = cp.asarray([0, 0, 2, 1, 1, 0], dtype=index_dtype)
    diff = _sum_duplicates_diff(row, col, size=6)
    cp.testing.assert_array_equal(diff, [0, 0, 1, 1, 0, 1])
    index = cp.cumsum(diff, dtype=index_dtype)
    rows = cp.empty(4, dtype=index_dtype)
    cols = cp.empty(4, dtype=index_dtype)
    _sum_duplicates_assign(row, col, index, rows, cols)
    cp.testing.assert_array_equal(rows, [0, 0, 1, 2])
    cp.testing.assert_array_equal(cols, [0, 2, 1, 0])
    values = cp.asarray([2, 0, 1, -3, 2, 4], dtype=cp.float64)
    sums = cp.zeros(4, dtype=cp.float64)
    squares = cp.zeros(4, dtype=cp.float64)
    _scatter_mean_var(values, index, sums, squares)
    cp.testing.assert_array_equal(sums, [2, 1, -1, 4])
    cp.testing.assert_array_equal(squares, [4, 1, 13, 16])
    counts = cp.zeros(4, dtype=cp.float32)
    _scatter_count_nonzero(values, index, counts)
    cp.testing.assert_array_equal(counts, [1, 1, 2, 1])
    _scatter_sum(values, index, sums)
    cp.testing.assert_array_equal(sums, [4, 2, -2, 8])
    occurrences = cp.zeros(4, dtype=cp.int32)
    assert getnnz_0(index, occurrences) is occurrences
    cp.testing.assert_array_equal(occurrences, [2, 1, 2, 1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_clipped_sparse_moments(dtype):
    from cupyx.scipy.sparse import csr_matrix

    values = cp.asarray([[1, 5, 0], [3, 4, 7]], dtype=dtype)
    clipped = cp.minimum(
        values.astype(cp.float64), cp.asarray([2, 3, 4], dtype=cp.float64)
    )
    squares, sums = _clip_square_sum_sparse(
        csr_matrix(values), cp.asarray([2, 3, 4], dtype=cp.float64)
    )
    cp.testing.assert_array_equal(sums, clipped.sum(axis=0))
    cp.testing.assert_array_equal(squares, cp.square(clipped).sum(axis=0))
