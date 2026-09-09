"""Sparse ranks preserve implicit-zero ties without allocating dense windows."""

from __future__ import annotations

import cupy as cp
import cupyx.scipy.sparse as gpu_sparse
import numpy as np
import pytest
import scipy.sparse as sparse
from scipy.stats import rankdata, tiecorrect

from rapids_singlecell._cuda import _wilcoxon_sparse_cuda as kernel


@pytest.mark.parametrize(
    "values,expected_ranks,expected_tie",
    [
        ([1.0, 0.0, -0.0, -2.0], [6.5, 3.5], 0.9),
        ([np.nan, np.nan, 0.0, 1.0], [4.0, 6.0], 1.0),
        ([-np.nan, -np.nan, 0.0, 1.0], [4.0, 6.0], 0.6),
        ([-np.inf, np.inf, 0.0, 1.0], [3.0, 7.0], 1.0),
        ([np.nan, np.nan, np.nan, np.nan], [4.0, 6.0], 1.0),
    ],
)
def test_sparse_ovr_preserves_special_value_ranking(
    values, expected_ranks, expected_tie
):
    # These expectations preserve the original sparse backend's ordering and
    # analytic-zero rules, including singleton positive-NaN ranks.
    data = cp.asarray(values, dtype=cp.float32)
    indices = cp.arange(4, dtype=cp.int32)
    indptr = cp.asarray([0, 4], dtype=cp.int32)
    codes = cp.asarray([0, 1, 0, 1], dtype=cp.int32)
    sizes = cp.asarray([2, 2], dtype=cp.float64)
    ranks = cp.empty((2, 1), dtype=cp.float64)
    ties = cp.empty(1, dtype=cp.float64)
    kernel.ovr_sparse_csc_device(
        data, indices, indptr, codes, sizes, ranks, ties, compute_tie_corr=True
    )
    np.testing.assert_array_equal(ranks.get().ravel(), expected_ranks)
    np.testing.assert_allclose(ties.get(), [expected_tie])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_sparse_ovr_keeps_sort_storage_sparse(monkeypatch, dtype, index_dtype):
    rng = np.random.default_rng(42)
    rows, cols = 20001, 67
    source = sparse.random(rows, cols, density=0.003, format="csc", random_state=rng)
    source.data = rng.integers(-3, 7, source.nnz).astype(dtype)
    # One dense column beside sparse columns exercises skewed padding budgets.
    source = sparse.hstack(
        [sparse.csc_matrix(np.ones((rows, 1))), source], format="csc"
    )
    source.data = source.data.astype(dtype)
    source.indices = source.indices.astype(index_dtype)
    source.indptr = source.indptr.astype(index_dtype)
    matrix = gpu_sparse.csc_matrix(source)
    matrix.indices = matrix.indices.astype(index_dtype)
    matrix.indptr = matrix.indptr.astype(index_dtype)
    codes_host = np.arange(rows, dtype=np.int32) % 4 - 1
    codes = cp.asarray(codes_host)
    sizes = cp.asarray(
        [(codes_host == group).sum() for group in range(3)], dtype=cp.float64
    )
    ranks = cp.empty((3, cols + 1), dtype=cp.float64)
    ties = cp.empty(cols + 1, dtype=cp.float64)
    allocations = []
    original_empty = cp.empty
    original_zeros = cp.zeros

    def record_shape(shape):
        if isinstance(shape, (tuple, list)) and len(shape) == 2:
            allocations.append(tuple(shape))
            assert np.prod(shape) < rows * 8, (
                "sparse ranking exceeded its padding budget"
            )

    def bounded_empty(shape, *args, **kwargs):
        record_shape(shape)
        return original_empty(shape, *args, **kwargs)

    def bounded_zeros(shape, *args, **kwargs):
        record_shape(shape)
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(cp, "empty", bounded_empty)
    monkeypatch.setattr(cp, "zeros", bounded_zeros)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        kernel.ovr_sparse_csc_device(
            matrix.data,
            matrix.indices,
            matrix.indptr,
            codes,
            sizes,
            ranks,
            ties,
            compute_tie_corr=True,
            sub_batch_cols=64,
        )
    stream.synchronize()
    dense = source.toarray().astype(np.float32)
    ranked = rankdata(dense, axis=0)
    expected = np.stack(
        [
            ranked[codes_host == group].sum(axis=0, dtype=np.float64)
            for group in range(3)
        ]
    )
    np.testing.assert_array_equal(ranks.get(), expected)
    np.testing.assert_allclose(
        ties.get(), [tiecorrect(ranked[:, col]) for col in range(cols + 1)]
    )
    # The heavily populated column must not force every sparse column to use
    # rows entries of sorting scratch.
    assert allocations
    assert max(np.prod(shape) for shape in allocations) < rows * 8
