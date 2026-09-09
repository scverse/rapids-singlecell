"""Direct numerical and allocation contracts for preprocessing device kernels."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import scipy.sparse as sp

from rapids_singlecell._cuda import (
    _hvg_cuda,
    _mean_var_cuda,
    _nanmean_cuda,
    _norm_cuda,
    _pr_cuda,
    _qc_cuda,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_masked_sparse_moments_on_external_stream(dtype, index_dtype):
    """Masking excludes NaNs and implicit zeros without changing the stream."""
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        indptr = cp.asarray([0, 3, 4, 6], dtype=index_dtype)
        indices = cp.asarray([0, 1, 3, 2, 1, 3], dtype=index_dtype)
        data = cp.asarray([2, np.nan, 5, 7, np.nan, -1], dtype=dtype)
        mask = cp.asarray([True, True, False, False])
        means = cp.full(3, -100, dtype=cp.float64)
        nans = cp.full(3, -100, dtype=cp.int32)
        minor_means = cp.zeros(4, dtype=cp.float64)
        minor_nans = cp.zeros(4, dtype=cp.int32)
    _nanmean_cuda.nan_mean_major(
        indptr,
        indices,
        data,
        means=means,
        nans=nans,
        mask=mask,
        major=3,
        minor=4,
        stream=stream.ptr,
    )
    _nanmean_cuda.nan_mean_minor(
        indices,
        data,
        means=minor_means,
        nans=minor_nans,
        mask=mask,
        nnz=6,
        stream=stream.ptr,
    )
    stream.synchronize()
    cp.testing.assert_array_equal(means, [2, 0, 0])
    cp.testing.assert_array_equal(nans, [1, 0, 1])
    cp.testing.assert_array_equal(minor_means, [2, 0, 0, 0])
    cp.testing.assert_array_equal(minor_nans, [0, 2, 0, 0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_sparse_qc_counts_explicit_zeros(dtype, index_dtype):
    """Sparse QC counts stored entries, including explicit zero/negative values."""
    indptr = cp.asarray([0, 3, 4], dtype=index_dtype)
    indices = cp.asarray([0, 1, 2, 0], dtype=index_dtype)
    data = cp.asarray([0, 3, -2, 4], dtype=dtype)
    cells = cp.zeros(2, dtype=dtype)
    genes = cp.zeros(3, dtype=dtype)
    cell_ex = cp.zeros(2, dtype=cp.int32)
    gene_ex = cp.zeros(3, dtype=cp.int32)
    _qc_cuda.sparse_qc_csr(
        indptr,
        indices,
        data,
        sums_cells=cells,
        sums_genes=genes,
        cell_ex=cell_ex,
        gene_ex=gene_ex,
        n_cells=2,
    )
    cp.testing.assert_array_equal(cells, [1, 4])
    cp.testing.assert_array_equal(genes, [4, 3, -2])
    cp.testing.assert_array_equal(cell_ex, [3, 1])
    cp.testing.assert_array_equal(gene_ex, [2, 1, 1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_masked_normalization_matches_numpy(dtype, index_dtype):
    original = np.asarray([[0, 2, 0, 6], [0, 0, 0, 0], [3, 0, 4, 1]], dtype=dtype)
    sparse = sp.csr_matrix(original)
    indptr = cp.asarray(sparse.indptr, dtype=index_dtype)
    indices = cp.asarray(sparse.indices, dtype=index_dtype)
    data = cp.asarray(sparse.data)
    mask = cp.asarray([False, True, False, True])
    sums = cp.full(3, -1, dtype=dtype)
    _norm_cuda.masked_sum_major(
        indptr, indices, data, gene_mask=mask, sums=sums, major=3
    )
    cp.testing.assert_array_equal(sums, [0, 0, 7])
    _norm_cuda.masked_mul_csr(indptr, indices, data, gene_mask=mask, nrows=3, tsum=14)
    expected = original.copy()
    expected[2] *= 2
    cp.testing.assert_array_equal(data, sp.csr_matrix(expected).data)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_expected_zeros_matches_poisson(dtype):
    means = np.asarray([0, 0.01, 0.25, 1], dtype=dtype)
    counts = np.asarray([0, 1, 10, 100], dtype=dtype)
    expected = cp.empty(4, dtype=dtype)
    _hvg_cuda.expected_zeros(cp.asarray(means), cp.asarray(counts), expected, 4, 4)
    reference = np.exp(-means[:, None] * counts).mean(axis=1)
    cp.testing.assert_allclose(
        expected, reference, rtol=5e-7 if dtype == np.float32 else 1e-14
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_pearson_variance_dense_sparse_agree(dtype, index_dtype):
    values = np.asarray([[1, 0, 3], [4, 2, 0], [0, 1, 6], [2, 0, 1]], dtype=dtype)
    csc = sp.csc_matrix(values)
    data = cp.asarray(csc.data)
    indptr = cp.asarray(csc.indptr, dtype=index_dtype)
    index = cp.asarray(csc.indices, dtype=index_dtype)
    cells = cp.asarray(values.sum(axis=1))
    genes = cp.asarray(values.sum(axis=0))
    sparse_variance = cp.empty(3, dtype=dtype)
    dense_variance = cp.empty(3, dtype=dtype)
    kwargs = {
        "sums_genes": genes,
        "sums_cells": cells,
        "inv_sum_total": 1 / float(values.sum()),
        "clip": 2.0,
        "inv_theta": 0.01,
        "n_genes": 3,
        "n_cells": 4,
    }
    _pr_cuda.csc_hvg_res(indptr, index, data, residuals=sparse_variance, **kwargs)
    _pr_cuda.dense_hvg_res(
        cp.asarray(values, order="F"), residuals=dense_variance, **kwargs
    )
    mu = values.sum(axis=1)[:, None] * values.sum(axis=0) / values.sum()
    residuals = np.clip((values - mu) / np.sqrt(mu + mu * mu * dtype(0.01)), -2, 2)
    cp.testing.assert_allclose(sparse_variance, residuals.var(axis=0), rtol=1e-6)
    cp.testing.assert_array_equal(dense_variance, sparse_variance)


def test_rust_moments_rejects_short_output():
    if getattr(_mean_var_cuda, "__backend__", None) != "rust":
        pytest.skip("Rust allocation validation contract")
    indptr = cp.asarray([0, 1, 2], dtype=cp.int32)
    index = cp.asarray([0, 1], dtype=cp.int32)
    data = cp.asarray([1, 2], dtype=cp.float32)
    with pytest.raises(ValueError, match="length"):
        _mean_var_cuda.mean_var_major(
            indptr,
            index,
            data,
            cp.zeros(1, dtype=cp.float64),
            cp.zeros(2, dtype=cp.float64),
            major=2,
            minor=2,
        )
