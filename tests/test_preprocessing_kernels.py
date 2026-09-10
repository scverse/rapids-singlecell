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


def test_sparse_qc_matches_cuda_atomic_add_subnormal_behavior():
    tiny = np.nextafter(np.float32(0), np.float32(1))
    indptr = cp.asarray([0, 1, 2], dtype=np.int32)
    indices = cp.zeros(2, dtype=np.int32)
    data = cp.full(2, tiny, dtype=np.float32)
    cells = cp.zeros(2, dtype=np.float32)
    genes = cp.zeros(1, dtype=np.float32)
    cell_ex = cp.zeros(2, dtype=np.int32)
    gene_ex = cp.zeros(1, dtype=np.int32)
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
    # Row-local arithmetic preserves subnormals; CUDA's global float32
    # atomicAdd flushes subnormal inputs for the shared gene accumulator.
    cp.testing.assert_array_equal(cells, tiny)
    cp.testing.assert_array_equal(genes, 0)
    cp.testing.assert_array_equal(cell_ex, 1)
    cp.testing.assert_array_equal(gene_ex, 2)


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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("width", [257, 4097])
@pytest.mark.parametrize(
    "operation", ["dense", "csr", "csr_masked", "dense_prescaled", "csr_prescaled"]
)
def test_normalization_preserves_long_row_reduction(dtype, width, operation):
    rng = np.random.default_rng(826)
    original = rng.integers(0, 10, size=(19, width)).astype(dtype)
    original[3] = 0
    source = sp.csr_matrix(original)
    scales = np.linspace(0.5, 2, len(original)).astype(dtype)
    mask = np.arange(width) % 3 == 0
    with cp.cuda.Stream(non_blocking=True) as work:
        values = cp.asarray(source.data if operation.startswith("csr") else original)
        pointers = cp.asarray(source.indptr, dtype=np.int64)
        indices = cp.asarray(source.indices, dtype=np.int64)
        d_scales = cp.asarray(scales)
        d_mask = cp.asarray(mask)
        if operation == "dense":
            _norm_cuda.mul_dense(
                values, nrows=19, ncols=width, target_sum=10_000, stream=work.ptr
            )
        elif operation == "csr":
            _norm_cuda.mul_csr(
                pointers, values, nrows=19, target_sum=10_000, stream=work.ptr
            )
        elif operation == "csr_masked":
            _norm_cuda.masked_mul_csr(
                pointers,
                indices,
                values,
                gene_mask=d_mask,
                nrows=19,
                tsum=10_000,
                stream=work.ptr,
            )
        elif operation == "dense_prescaled":
            _norm_cuda.prescaled_mul_dense(
                values, scales=d_scales, nrows=19, ncols=width, stream=work.ptr
            )
        else:
            _norm_cuda.prescaled_mul_csr(
                pointers, values, scales=d_scales, nrows=19, stream=work.ptr
            )
        actual = values.get(stream=work)
    if operation.endswith("prescaled"):
        factors = scales
    else:
        totals = (
            original[:, ~mask].sum(1) if operation == "csr_masked" else original.sum(1)
        )
        factors = np.divide(
            dtype(10_000), totals, out=np.zeros_like(totals), where=totals > 0
        )
    expected = original * factors[:, None]
    if operation.startswith("csr"):
        expected = sp.csr_matrix(expected).data
    np.testing.assert_allclose(
        actual, expected, rtol=2e-6 if dtype == np.float32 else 1e-13, atol=1e-7
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("rows", [3, 4, 5, 259])
def test_pearson_layouts_preserve_clipping_and_loop_tails(dtype, index_dtype, rows):
    """Cover unrolled tails, zero libraries, and sparse output accumulation."""
    rng = np.random.default_rng(347)
    values = rng.integers(0, 8, size=(rows, 5)).astype(dtype)
    values[0] = 0
    values[:, 0] = 0
    cells, genes = values.sum(axis=1), values.sum(axis=0)
    scale, theta, clip = dtype(1 / float(cells.sum())), dtype(0.01), dtype(2)
    kwargs = {
        "sums_cells": cp.asarray(cells),
        "sums_genes": cp.asarray(genes),
        "inv_sum_total": float(scale),
        "clip": float(clip),
        "inv_theta": float(theta),
        "n_cells": rows,
        "n_genes": values.shape[1],
    }
    mu = (cells[:, None] * genes) * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1 / np.sqrt(mu + mu * mu * theta)
        expected = np.clip((values - mu) * inverse, -clip, clip)
        accumulated = np.clip((values + dtype(0.25) - mu) * inverse, -clip, clip)
        # CUDA's HVG fmin/fmax selects -clip for a NaN residual.
        hvg = np.fmin(np.fmax((values - mu) * inverse, -clip), clip).var(axis=0)
    tolerance = 3e-6 if dtype == np.float32 else 1e-12
    for layout in ("csr", "csc", "dense", "csc_hvg", "dense_hvg"):
        variance = layout.endswith("hvg")
        output = cp.full(
            values.shape[1] if variance else values.shape, 0.25, dtype=dtype
        )
        if layout in {"csr", "csc", "csc_hvg"}:
            compressed = (
                sp.csr_matrix(values) if layout == "csr" else sp.csc_matrix(values)
            )
            function = getattr(
                _pr_cuda, "csc_hvg_res" if variance else f"sparse_norm_res_{layout}"
            )
            function(
                cp.asarray(compressed.indptr, dtype=index_dtype),
                cp.asarray(compressed.indices, dtype=index_dtype),
                cp.asarray(compressed.data),
                residuals=output,
                **kwargs,
            )
        else:
            function = _pr_cuda.dense_hvg_res if variance else _pr_cuda.dense_norm_res
            function(
                cp.asarray(values, order="F" if variance else "C"),
                residuals=output,
                **kwargs,
            )
        reference = hvg if variance else expected if layout == "dense" else accumulated
        cp.testing.assert_allclose(output, reference, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_major_statistics_preserve_cross_warp_cancellation(dtype, index_dtype):
    """The original 64-lane tree retains two units lost by warp-first sums."""
    values = np.zeros(64, dtype=dtype)
    values[[0, 16, 32, 48]] = [1e16, 1, -1e16, 1]
    pointers = cp.asarray([0, 64], dtype=index_dtype)
    indices = cp.arange(64, dtype=index_dtype)
    mean = cp.empty(1, dtype=cp.float64)
    second = cp.empty_like(mean)
    _mean_var_cuda.mean_var_major(
        pointers, indices, cp.asarray(values), mean, second, major=1, minor=64
    )
    cp.testing.assert_array_equal(mean, [2.0])
    cp.testing.assert_allclose(second, [2 * float(values[0]) ** 2], rtol=2e-16)
    values[62:] = np.nan
    mask = cp.ones(64, dtype=cp.bool_)
    mask[63] = False
    nans = cp.empty(1, dtype=cp.int32)
    _nanmean_cuda.nan_mean_major(
        pointers,
        indices,
        cp.asarray(values),
        means=mean,
        nans=nans,
        mask=mask,
        major=1,
        minor=64,
    )
    cp.testing.assert_array_equal(mean, [2.0])
    cp.testing.assert_array_equal(nans, [1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("width", [31, 32, 33, 257])
@pytest.mark.parametrize("operation", ["dense", "csr", "csr_masked"])
def test_normalization_broadcasts_original_cancellation_result(dtype, width, operation):
    # The halving reduction combines lanes 0+16, then 8+24: exactly 2.
    # A sequential or adjacent-pair sum would lose one or both small values.
    original = np.zeros((3, width), dtype=dtype)
    large = dtype(1e8 if dtype == np.float32 else 1e16)
    original[0, [0, 8, 16, 24]] = [large, 1, -large, 1]
    original[1] = -original[0]
    values = cp.asarray(original if operation == "dense" else original.ravel())
    ready = cp.cuda.get_current_stream().record()
    with cp.cuda.Stream(non_blocking=True) as work:
        work.wait_event(ready)
        if operation == "dense":
            _norm_cuda.mul_dense(
                values, nrows=3, ncols=width, target_sum=4, stream=work.ptr
            )
        else:
            # Retain explicit zeros so sparse and dense reductions use identical
            # lane assignments, including the partial final warp.
            pointers = cp.arange(4, dtype=cp.int64) * width
            if operation == "csr":
                _norm_cuda.mul_csr(
                    pointers, values, nrows=3, target_sum=4, stream=work.ptr
                )
            else:
                indices = cp.tile(cp.arange(width, dtype=cp.int64), 3)
                mask = cp.zeros(width, dtype=cp.bool_)
                _norm_cuda.masked_mul_csr(
                    pointers,
                    indices,
                    values,
                    gene_mask=mask,
                    nrows=3,
                    tsum=4,
                    stream=work.ptr,
                )
        actual = values.get(stream=work).reshape(original.shape)
    expected = original.copy()
    expected[0] *= dtype(2)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_major_statistics_reuse_shared_storage_after_grid_limit(dtype):
    rows = 65_537
    with cp.cuda.Stream(non_blocking=True) as work:
        values = cp.tile(cp.asarray([2, -1, 3, 4], dtype=dtype), rows)
        pointers = cp.arange(rows + 1, dtype=cp.int64) * 4
        indices = cp.tile(cp.arange(4, dtype=cp.int64), rows)
        sums = cp.empty(rows, dtype=cp.float64)
        squares = cp.empty_like(sums)
        _mean_var_cuda.mean_var_major(
            pointers,
            indices,
            values,
            sums,
            squares,
            major=rows,
            minor=4,
            stream=work.ptr,
        )
        np.testing.assert_array_equal(sums.get(stream=work), 8)
        np.testing.assert_array_equal(squares.get(stream=work), 30)
        values[3::4] = cp.nan
        counts = cp.empty(rows, dtype=cp.int32)
        mask = cp.ones(4, dtype=cp.bool_)
        _nanmean_cuda.nan_mean_major(
            pointers,
            indices,
            values,
            means=sums,
            nans=counts,
            mask=mask,
            major=rows,
            minor=4,
            stream=work.ptr,
        )
        np.testing.assert_array_equal(sums.get(stream=work), 4)
        np.testing.assert_array_equal(counts.get(stream=work), 1)
