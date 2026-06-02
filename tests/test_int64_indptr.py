"""Kernel-level int32 vs int64 parity tests.

Each test calls a single ``_cuda`` kernel with bit-identical inputs except for
the indptr/indices dtype (int32 vs int64) and asserts the outputs match
exactly. The purpose is to catch silent dispatch bugs in the int64 nanobind
overloads.

These tests build raw int64 indptr/indices arrays directly with
``cp.asarray(..., dtype=cp.int64)`` — they do *not* go through
``cupyx.scipy.sparse``, so they work on any CuPy version (the upstream PR
cupy/cupy#9825 is only needed for end users who want to pass int64 sparse
matrices through the high-level ``rsc.pp`` API).
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
import pytest
import scipy.sparse as sp

from rapids_singlecell._cuda import _aggr_cuda as _aggr
from rapids_singlecell._cuda import _autocorr_cuda as _ac
from rapids_singlecell._cuda import _ligrec_cuda as _lc
from rapids_singlecell._cuda import _mean_var_cuda as _mv
from rapids_singlecell._cuda import _nanmean_cuda as _nm
from rapids_singlecell._cuda import _norm_cuda as _norm
from rapids_singlecell._cuda import _pr_cuda as _pr
from rapids_singlecell._cuda import _qc_cuda as _qc
from rapids_singlecell._cuda import _qc_dask_cuda as _qcd
from rapids_singlecell._cuda import _scale_cuda as _scale
from rapids_singlecell._cuda import _sparse2dense_cuda as _s2d
from rapids_singlecell._cuda import _spca_cuda as _spca
from rapids_singlecell._cuda import _wilcoxon_binned_cuda as _wb

DTYPES = [np.float32, np.float64]

N_ROWS, N_COLS = 64, 48
SEED = 0


# ----------------------------- shared fixtures ------------------------------


@dataclass
class SparseGPU:
    """Minimal GPU sparse handle: indptr/indices/data + shape + nnz.

    Built directly from numpy primitives, NOT through ``cupyx.scipy.sparse``,
    so int64 indptr/indices survive on any CuPy build.
    """

    indptr: cp.ndarray
    indices: cp.ndarray
    data: cp.ndarray
    shape: tuple

    @property
    def nnz(self) -> int:
        return int(self.data.size)

    def copy(self) -> SparseGPU:
        return SparseGPU(
            indptr=self.indptr.copy(),
            indices=self.indices.copy(),
            data=self.data.copy(),
            shape=self.shape,
        )


def _make_dense_cpu(dtype, *, rows=N_ROWS, cols=N_COLS, density=0.3, seed=SEED):
    rng = np.random.default_rng(seed)
    X = rng.poisson(0.6, size=(rows, cols)).astype(dtype)
    keep = rng.random((rows, cols)) <= density
    return X * keep


def _make_csr(dtype, idx_dtype, **kwargs) -> SparseGPU:
    dense = _make_dense_cpu(dtype, **kwargs)
    sp_csr = sp.csr_matrix(dense)
    return SparseGPU(
        indptr=cp.asarray(sp_csr.indptr.astype(idx_dtype)),
        indices=cp.asarray(sp_csr.indices.astype(idx_dtype)),
        data=cp.asarray(sp_csr.data.astype(dtype)),
        shape=sp_csr.shape,
    )


def _make_csc(dtype, idx_dtype, **kwargs) -> SparseGPU:
    dense = _make_dense_cpu(dtype, **kwargs)
    sp_csc = sp.csc_matrix(dense)
    return SparseGPU(
        indptr=cp.asarray(sp_csc.indptr.astype(idx_dtype)),
        indices=cp.asarray(sp_csc.indices.astype(idx_dtype)),
        data=cp.asarray(sp_csc.data.astype(dtype)),
        shape=sp_csc.shape,
    )


def _make_adjacency(dtype, idx_dtype) -> SparseGPU:
    row = np.repeat(np.arange(N_ROWS), 2)
    col = np.column_stack(
        ((np.arange(N_ROWS) - 1) % N_ROWS, (np.arange(N_ROWS) + 1) % N_ROWS)
    ).ravel()
    data = np.ones(row.size, dtype=dtype)
    adj = sp.csr_matrix((data, (row, col)), shape=(N_ROWS, N_ROWS))
    return SparseGPU(
        indptr=cp.asarray(adj.indptr.astype(idx_dtype)),
        indices=cp.asarray(adj.indices.astype(idx_dtype)),
        data=cp.asarray(adj.data.astype(dtype)),
        shape=adj.shape,
    )


def _zeros(shape, dtype=cp.float32):
    return cp.zeros(shape, dtype=dtype)


def _stream():
    return cp.cuda.get_current_stream().ptr


def _eq(a, b):
    """Bit-exact equality — the int32 and int64 paths run the same code with
    the same input bytes, so outputs should match exactly."""
    cp.testing.assert_array_equal(a, b)


def _close(a, b):
    """For kernels with atomicAdd on float accumulators, the int32 vs int64
    paths can shift thread scheduling enough to reorder float adds (ULP-level
    differences). Use tight relative tolerance to catch real dispatch bugs."""
    if a.dtype == cp.float32:
        cp.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    else:
        cp.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-12)


def _close_morans_i(a, b):
    """Sparse Moran's I has many float32 atomic adds into small numerators.
    Turing/T4 can reorder the int32 and int64 instantiations enough to exceed
    the generic atomic tolerance without indicating a dispatch regression."""
    if a.dtype == cp.float32:
        cp.testing.assert_allclose(a, b, rtol=1e-4, atol=5e-6)
    else:
        cp.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-12)


# ----------------------------- _qc_cuda -------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_qc_csr(dtype):
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        sums_cells = _zeros(N_ROWS, dtype)
        sums_genes = _zeros(N_COLS, dtype)
        cell_ex = _zeros(N_ROWS, cp.int32)
        gene_ex = _zeros(N_COLS, cp.int32)
        _qc.sparse_qc_csr(
            A.indptr,
            A.indices,
            A.data,
            sums_cells=sums_cells,
            sums_genes=sums_genes,
            cell_ex=cell_ex,
            gene_ex=gene_ex,
            n_cells=N_ROWS,
            stream=_stream(),
        )
        outs[idx_dtype] = (sums_cells, sums_genes, cell_ex, gene_ex)
    for a, b in zip(outs[np.int32], outs[np.int64], strict=True):
        _eq(a, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_qc_csc(dtype):
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csc(dtype, idx_dtype)
        sums_cells = _zeros(N_ROWS, dtype)
        sums_genes = _zeros(N_COLS, dtype)
        cell_ex = _zeros(N_ROWS, cp.int32)
        gene_ex = _zeros(N_COLS, cp.int32)
        _qc.sparse_qc_csc(
            A.indptr,
            A.indices,
            A.data,
            sums_cells=sums_cells,
            sums_genes=sums_genes,
            cell_ex=cell_ex,
            gene_ex=gene_ex,
            n_genes=N_COLS,
            stream=_stream(),
        )
        outs[idx_dtype] = (sums_cells, sums_genes, cell_ex, gene_ex)
    for a, b in zip(outs[np.int32], outs[np.int64], strict=True):
        _eq(a, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_qc_csr_sub(dtype):
    mask = cp.zeros(N_COLS, dtype=cp.bool_)
    mask[::3] = True
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        sums_cells = _zeros(N_ROWS, dtype)
        _qc.sparse_qc_csr_sub(
            A.indptr,
            A.indices,
            A.data,
            sums_cells=sums_cells,
            mask=mask,
            n_cells=N_ROWS,
            stream=_stream(),
        )
        outs[idx_dtype] = sums_cells
    _eq(outs[np.int32], outs[np.int64])


@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_qc_csc_sub(dtype):
    mask = cp.zeros(N_COLS, dtype=cp.bool_)
    mask[::3] = True
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csc(dtype, idx_dtype)
        sums_cells = _zeros(N_ROWS, dtype)
        _qc.sparse_qc_csc_sub(
            A.indptr,
            A.indices,
            A.data,
            sums_cells=sums_cells,
            mask=mask,
            n_genes=N_COLS,
            stream=_stream(),
        )
        outs[idx_dtype] = sums_cells
    _eq(outs[np.int32], outs[np.int64])


# --------------------------- _qc_dask_cuda ----------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_qc_csr_cells(dtype):
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        sums_cells = _zeros(N_ROWS, dtype)
        cell_ex = _zeros(N_ROWS, cp.int32)
        _qcd.sparse_qc_csr_cells(
            A.indptr,
            A.indices,
            A.data,
            sums_cells=sums_cells,
            cell_ex=cell_ex,
            n_cells=N_ROWS,
            stream=_stream(),
        )
        outs[idx_dtype] = (sums_cells, cell_ex)
    for a, b in zip(outs[np.int32], outs[np.int64], strict=True):
        _eq(a, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_qc_csr_genes(dtype):
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        sums_genes = _zeros(N_COLS, dtype)
        gene_ex = _zeros(N_COLS, cp.int32)
        _qcd.sparse_qc_csr_genes(
            A.indices,
            A.data,
            sums_genes=sums_genes,
            gene_ex=gene_ex,
            nnz=A.nnz,
            stream=_stream(),
        )
        outs[idx_dtype] = (sums_genes, gene_ex)
    for a, b in zip(outs[np.int32], outs[np.int64], strict=True):
        _eq(a, b)


# ----------------------------- _mean_var ------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_mean_var_major(dtype):
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        means = _zeros(N_ROWS, cp.float64)
        vars_ = _zeros(N_ROWS, cp.float64)
        _mv.mean_var_major(
            A.indptr,
            A.indices,
            A.data,
            means,
            vars_,
            major=N_ROWS,
            minor=N_COLS,
            stream=_stream(),
        )
        outs[idx_dtype] = (means, vars_)
    for a, b in zip(outs[np.int32], outs[np.int64], strict=True):
        _eq(a, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_mean_var_minor(dtype):
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        means = _zeros(N_COLS, cp.float64)
        vars_ = _zeros(N_COLS, cp.float64)
        _mv.mean_var_minor(
            A.indices,
            A.data,
            means,
            vars_,
            nnz=A.nnz,
            stream=_stream(),
        )
        outs[idx_dtype] = (means, vars_)
    for a, b in zip(outs[np.int32], outs[np.int64], strict=True):
        _eq(a, b)


# ----------------------------- _nanmean -------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_nan_mean_major(dtype):
    mask = cp.ones(N_COLS, dtype=cp.bool_)
    mask[::5] = False
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        means = _zeros(N_ROWS, cp.float64)
        nans = _zeros(N_ROWS, cp.int32)
        _nm.nan_mean_major(
            A.indptr,
            A.indices,
            A.data,
            means=means,
            nans=nans,
            mask=mask,
            major=N_ROWS,
            minor=N_COLS,
            stream=_stream(),
        )
        outs[idx_dtype] = (means, nans)
    for a, b in zip(outs[np.int32], outs[np.int64], strict=True):
        _eq(a, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_nan_mean_minor(dtype):
    mask = cp.ones(N_COLS, dtype=cp.bool_)
    mask[::5] = False
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        means = _zeros(N_COLS, cp.float64)
        nans = _zeros(N_COLS, cp.int32)
        _nm.nan_mean_minor(
            A.indices,
            A.data,
            means=means,
            nans=nans,
            mask=mask,
            nnz=A.nnz,
            stream=_stream(),
        )
        outs[idx_dtype] = (means, nans)
    for a, b in zip(outs[np.int32], outs[np.int64], strict=True):
        _eq(a, b)


# ----------------------------- _norm ----------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_mul_csr(dtype):
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        _norm.mul_csr(
            A.indptr,
            A.data,
            nrows=N_ROWS,
            target_sum=dtype(1.0),
            stream=_stream(),
        )
        outs[idx_dtype] = A.data
    _eq(outs[np.int32], outs[np.int64])


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_major(dtype):
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        sums = _zeros(N_ROWS, dtype)
        _norm.sum_major(
            A.indptr,
            A.data,
            sums=sums,
            major=N_ROWS,
            stream=_stream(),
        )
        outs[idx_dtype] = sums
    _eq(outs[np.int32], outs[np.int64])


@pytest.mark.parametrize("dtype", DTYPES)
def test_find_hi_genes_csr(dtype):
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        out = cp.zeros(N_COLS, dtype=cp.bool_)
        _norm.find_hi_genes_csr(
            A.indptr,
            A.indices,
            A.data,
            gene_is_hi=out,
            max_fraction=dtype(0.05),
            nrows=N_ROWS,
            stream=_stream(),
        )
        outs[idx_dtype] = out
    _eq(outs[np.int32], outs[np.int64])


@pytest.mark.parametrize("dtype", DTYPES)
def test_masked_mul_csr(dtype):
    mask = cp.ones(N_COLS, dtype=cp.bool_)
    mask[::4] = False
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        _norm.masked_mul_csr(
            A.indptr,
            A.indices,
            A.data,
            gene_mask=mask,
            nrows=N_ROWS,
            tsum=dtype(1.0),
            stream=_stream(),
        )
        outs[idx_dtype] = A.data
    _eq(outs[np.int32], outs[np.int64])


@pytest.mark.parametrize("dtype", DTYPES)
def test_masked_sum_major(dtype):
    mask = cp.ones(N_COLS, dtype=cp.bool_)
    mask[::4] = False
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        sums = _zeros(N_ROWS, dtype)
        _norm.masked_sum_major(
            A.indptr,
            A.indices,
            A.data,
            gene_mask=mask,
            sums=sums,
            major=N_ROWS,
            stream=_stream(),
        )
        outs[idx_dtype] = sums
    _eq(outs[np.int32], outs[np.int64])


@pytest.mark.parametrize("dtype", DTYPES)
def test_prescaled_mul_csr(dtype):
    scales = cp.full(N_ROWS, 0.5, dtype=dtype)
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        _norm.prescaled_mul_csr(
            A.indptr,
            A.data,
            scales=scales,
            nrows=N_ROWS,
            stream=_stream(),
        )
        outs[idx_dtype] = A.data
    _eq(outs[np.int32], outs[np.int64])


# ----------------------------- _pr ------------------------------------------


def _row_sums_cpu(dtype):
    """Pre-computed sums_cells/sums_genes from the shared fixture, on GPU."""
    dense = cp.asarray(_make_dense_cpu(dtype))
    return dense.sum(axis=1).astype(dtype), dense.sum(axis=0).astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_norm_res_csr(dtype):
    sums_cells, sums_genes = _row_sums_cpu(dtype)
    inv_total = dtype(1.0 / float(sums_cells.sum()))
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        res = cp.zeros((N_ROWS, N_COLS), dtype=dtype)
        _pr.sparse_norm_res_csr(
            A.indptr,
            A.indices,
            A.data,
            sums_cells=sums_cells,
            sums_genes=sums_genes,
            residuals=res,
            inv_sum_total=inv_total,
            clip=dtype(10.0),
            inv_theta=dtype(0.01),
            n_cells=N_ROWS,
            n_genes=N_COLS,
            stream=_stream(),
        )
        outs[idx_dtype] = res
    _eq(outs[np.int32], outs[np.int64])


@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_norm_res_csc(dtype):
    sums_cells, sums_genes = _row_sums_cpu(dtype)
    inv_total = dtype(1.0 / float(sums_cells.sum()))
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csc(dtype, idx_dtype)
        res = cp.zeros((N_ROWS, N_COLS), dtype=dtype)
        _pr.sparse_norm_res_csc(
            A.indptr,
            A.indices,
            A.data,
            sums_cells=sums_cells,
            sums_genes=sums_genes,
            residuals=res,
            inv_sum_total=inv_total,
            clip=dtype(10.0),
            inv_theta=dtype(0.01),
            n_cells=N_ROWS,
            n_genes=N_COLS,
            stream=_stream(),
        )
        outs[idx_dtype] = res
    _eq(outs[np.int32], outs[np.int64])


@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_sum_csc(dtype):
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csc(dtype, idx_dtype)
        sums_genes = _zeros(N_COLS, dtype)
        sums_cells = _zeros(N_ROWS, dtype)
        _pr.sparse_sum_csc(
            A.indptr,
            A.indices,
            A.data,
            sums_genes=sums_genes,
            sums_cells=sums_cells,
            n_genes=N_COLS,
            stream=_stream(),
        )
        outs[idx_dtype] = (sums_genes, sums_cells)
    for a, b in zip(outs[np.int32], outs[np.int64], strict=True):
        _eq(a, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_csc_hvg_res(dtype):
    sums_cells, sums_genes = _row_sums_cpu(dtype)
    inv_total = dtype(1.0 / float(sums_cells.sum()))
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csc(dtype, idx_dtype)
        res = cp.zeros((N_ROWS, N_COLS), dtype=dtype)
        _pr.csc_hvg_res(
            A.indptr,
            A.indices,
            A.data,
            sums_genes=sums_genes,
            sums_cells=sums_cells,
            residuals=res,
            inv_sum_total=inv_total,
            clip=dtype(10.0),
            inv_theta=dtype(0.01),
            n_genes=N_COLS,
            n_cells=N_ROWS,
            stream=_stream(),
        )
        outs[idx_dtype] = res
    _eq(outs[np.int32], outs[np.int64])


# ----------------------------- _scale ---------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_csc_scale_diff(dtype):
    std = cp.full(N_COLS, 2.0, dtype=dtype)
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csc(dtype, idx_dtype)
        _scale.csc_scale_diff(
            A.indptr,
            A.data,
            std,
            ncols=N_COLS,
            stream=_stream(),
        )
        outs[idx_dtype] = A.data
    _eq(outs[np.int32], outs[np.int64])


@pytest.mark.parametrize("dtype", DTYPES)
def test_csr_scale_diff(dtype):
    std = cp.full(N_COLS, 2.0, dtype=dtype)
    mask = cp.ones(N_ROWS, dtype=cp.int32)
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        _scale.csr_scale_diff(
            A.indptr,
            A.indices,
            A.data,
            std,
            mask,
            clipper=dtype(10.0),
            nrows=N_ROWS,
            stream=_stream(),
        )
        outs[idx_dtype] = A.data
    _eq(outs[np.int32], outs[np.int64])


# ----------------------------- _sparse2dense --------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse2dense_c(dtype):
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        max_nnz = int((A.indptr[1:] - A.indptr[:-1]).max())
        out = cp.zeros((N_ROWS, N_COLS), dtype=dtype, order="C")
        _s2d.sparse2dense(
            A.indptr,
            A.indices,
            A.data,
            out=out,
            major=N_ROWS,
            minor=N_COLS,
            c_switch=True,
            max_nnz=max_nnz,
            stream=_stream(),
        )
        outs[idx_dtype] = out
    _eq(outs[np.int32], outs[np.int64])


@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse2dense_f(dtype):
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        max_nnz = int((A.indptr[1:] - A.indptr[:-1]).max())
        out = cp.zeros((N_ROWS, N_COLS), dtype=dtype, order="F")
        _s2d.sparse2dense(
            A.indptr,
            A.indices,
            A.data,
            out=out,
            major=N_ROWS,
            minor=N_COLS,
            c_switch=False,
            max_nnz=max_nnz,
            stream=_stream(),
        )
        outs[idx_dtype] = out
    _eq(outs[np.int32], outs[np.int64])


# ----------------------------- _spca ----------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_gram_csr_upper(dtype):
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        out = cp.zeros((N_COLS, N_COLS), dtype=dtype)
        _spca.gram_csr_upper(
            A.indptr,
            A.indices,
            A.data,
            nrows=N_ROWS,
            ncols=N_COLS,
            out=out,
            stream=_stream(),
        )
        outs[idx_dtype] = out
    _eq(outs[np.int32], outs[np.int64])


def test_check_zero_genes():
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(np.float32, idx_dtype)
        out = cp.zeros(N_COLS, dtype=cp.int32)
        _spca.check_zero_genes(
            A.indices,
            out=out,
            nnz=A.nnz,
            num_genes=N_COLS,
            stream=_stream(),
        )
        outs[idx_dtype] = out
    _eq(outs[np.int32], outs[np.int64])


# ----------------------------- _aggr ----------------------------------------


@pytest.mark.parametrize("is_csc", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_aggr(dtype, is_csc):
    n_groups = 3
    rng = np.random.default_rng(SEED)
    cats = cp.asarray(rng.integers(0, n_groups, size=N_ROWS).astype(np.int32))
    mask = cp.ones(N_ROWS, dtype=cp.bool_)
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csc(dtype, idx_dtype) if is_csc else _make_csr(dtype, idx_dtype)
        out_sum = cp.zeros((n_groups, N_COLS), dtype=cp.float64)
        out_count = cp.zeros((n_groups, N_COLS), dtype=cp.float64)
        out_sqsum = cp.zeros((n_groups, N_COLS), dtype=cp.float64)
        _aggr.sparse_aggr(
            A.indptr,
            A.indices,
            A.data,
            out_sum=out_sum,
            out_count=out_count,
            out_sqsum=out_sqsum,
            cats=cats,
            mask=mask,
            n_cells=N_ROWS,
            n_genes=N_COLS,
            is_csc=is_csc,
            stream=_stream(),
        )
        outs[idx_dtype] = cp.stack([out_sum, out_count, out_sqsum])
    _eq(outs[np.int32], outs[np.int64])


@pytest.mark.parametrize("dtype", DTYPES)
def test_csr_to_coo(dtype):
    rng = np.random.default_rng(SEED)
    cats = cp.asarray(rng.integers(0, 3, size=N_ROWS).astype(np.int32))
    mask = cp.ones(N_ROWS, dtype=cp.bool_)
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        out_idx_dtype = cp.dtype(idx_dtype)
        out_row = cp.zeros(A.nnz, dtype=out_idx_dtype)
        out_col = cp.zeros(A.nnz, dtype=out_idx_dtype)
        out_data = cp.zeros(A.nnz, dtype=cp.float64)
        _aggr.csr_to_coo(
            A.indptr,
            A.indices,
            A.data,
            out_row=out_row,
            out_col=out_col,
            out_data=out_data,
            cats=cats,
            mask=mask,
            n_cells=N_ROWS,
            stream=_stream(),
        )
        outs[idx_dtype] = (out_row, out_col, out_data)
    assert outs[np.int64][0].dtype == cp.int64
    assert outs[np.int64][1].dtype == cp.int64
    _eq(outs[np.int32][0].astype(cp.int64), outs[np.int64][0])
    _eq(outs[np.int32][1].astype(cp.int64), outs[np.int64][1])
    _eq(outs[np.int32][2], outs[np.int64][2])


def test_sparse_var():
    n_groups = 3
    # Per-group CSR: group major, gene minor, each row dense over all genes
    # so indptr is uniform; data is arbitrary float64.
    rng = np.random.default_rng(SEED)
    n_cells_per_group = cp.asarray(
        rng.integers(1, 20, size=n_groups).astype(np.float64)
    )
    means = cp.asarray(rng.random((n_groups, N_COLS)))
    base_data = cp.asarray(rng.random((n_groups, N_COLS)))
    base_indptr = (np.arange(n_groups + 1) * N_COLS).astype(np.int64)
    base_indices = np.tile(np.arange(N_COLS), n_groups).astype(np.int64)

    outs = {}
    for idx_dtype in (np.int32, np.int64):
        indptr = cp.asarray(base_indptr.astype(idx_dtype))
        indices = cp.asarray(base_indices.astype(idx_dtype))
        data = base_data.copy().ravel()
        _aggr.sparse_var(
            indptr,
            indices,
            data,
            means=means,
            n_cells=n_cells_per_group.copy(),
            dof=1,
            n_groups=n_groups,
            stream=_stream(),
        )
        outs[idx_dtype] = data
    _eq(outs[np.int32], outs[np.int64])


def test_sparse_aggregate_elementwise_int64_helpers():
    from rapids_singlecell.get._kernels._aggr_elementwise import (
        _scatter_count_nonzero,
        _scatter_mean_var,
        _scatter_sum,
        _sum_duplicates_assign,
        _sum_duplicates_diff,
    )

    src_row = cp.asarray([0, 0, 1, 1, 1], dtype=cp.int64)
    src_col = cp.asarray([0, 0, 0, 1, 1], dtype=cp.int64)
    src_data = cp.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cp.float64)

    diff = _sum_duplicates_diff(src_row, src_col, size=src_row.size)
    index = cp.cumsum(diff, dtype=cp.int64)
    nnz = int(index[-1].get())

    rows = cp.zeros(nnz + 1, dtype=cp.int64)
    indices = cp.zeros(nnz + 1, dtype=cp.int64)
    _sum_duplicates_assign(src_row, src_col, index, rows, indices)

    sums = cp.zeros(nnz + 1, dtype=cp.float64)
    means = cp.zeros(nnz + 1, dtype=cp.float64)
    vars_ = cp.zeros(nnz + 1, dtype=cp.float64)
    counts = cp.zeros(nnz + 1, dtype=cp.float32)
    _scatter_sum(src_data, index, sums)
    _scatter_mean_var(src_data, index, means, vars_)
    _scatter_count_nonzero(src_data, index, counts)

    assert rows.dtype == cp.int64
    assert indices.dtype == cp.int64
    _eq(rows, cp.asarray([0, 1, 1], dtype=cp.int64))
    _eq(indices, cp.asarray([0, 0, 1], dtype=cp.int64))
    _eq(sums, cp.asarray([3.0, 3.0, 9.0], dtype=cp.float64))
    _eq(means, cp.asarray([3.0, 3.0, 9.0], dtype=cp.float64))
    _eq(vars_, cp.asarray([5.0, 9.0, 41.0], dtype=cp.float64))
    _eq(counts, cp.asarray([2.0, 1.0, 2.0], dtype=cp.float32))


# ----------------------------- _ligrec --------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_count_sparse(dtype):
    ncls = 4
    rng = np.random.default_rng(SEED)
    clusters = cp.asarray(rng.integers(0, ncls, size=N_ROWS).astype(np.int32))
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        sum_out = cp.zeros((ncls, N_COLS), dtype=dtype)
        count_out = cp.zeros((ncls, N_COLS), dtype=cp.int32)
        _lc.sum_count_sparse(
            A.indptr,
            A.indices,
            A.data,
            clusters=clusters,
            sum=sum_out,
            count=count_out,
            rows=N_ROWS,
            ncls=ncls,
            stream=_stream(),
        )
        outs[idx_dtype] = (sum_out, count_out)
    for a, b in zip(outs[np.int32], outs[np.int64], strict=True):
        _eq(a, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_mean_sparse(dtype):
    ncls = 4
    rng = np.random.default_rng(SEED)
    clusters = cp.asarray(rng.integers(0, ncls, size=N_ROWS).astype(np.int32))
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        g = cp.zeros((ncls, N_COLS), dtype=dtype)
        _lc.mean_sparse(
            A.indptr,
            A.indices,
            A.data,
            clusters=clusters,
            g=g,
            rows=N_ROWS,
            ncls=ncls,
            stream=_stream(),
        )
        outs[idx_dtype] = g
    _eq(outs[np.int32], outs[np.int64])


# ----------------------------- _autocorr ------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_morans_sparse(dtype):
    mean_array = cp.full(N_COLS, 0.3, dtype=dtype)
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        adj = _make_adjacency(dtype, idx_dtype)
        num = cp.zeros(N_COLS, dtype=dtype)
        _ac.morans_sparse(
            adj.indptr,
            adj.indices,
            adj.data,
            data_row_ptr=A.indptr,
            data_col_ind=A.indices,
            data_values=A.data,
            n_samples=N_ROWS,
            n_features=N_COLS,
            mean_array=mean_array,
            num=num,
            stream=_stream(),
        )
        outs[idx_dtype] = num
    _close_morans_i(outs[np.int32], outs[np.int64])


@pytest.mark.parametrize("dtype", DTYPES)
def test_gearys_sparse(dtype):
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        adj = _make_adjacency(dtype, idx_dtype)
        num = cp.zeros(N_COLS, dtype=dtype)
        _ac.gearys_sparse(
            adj.indptr,
            adj.indices,
            adj.data,
            data_row_ptr=A.indptr,
            data_col_ind=A.indices,
            data_values=A.data,
            n_samples=N_ROWS,
            n_features=N_COLS,
            num=num,
            stream=_stream(),
        )
        outs[idx_dtype] = num
    _close(outs[np.int32], outs[np.int64])


@pytest.mark.parametrize("dtype", DTYPES)
def test_pre_den_sparse(dtype):
    # Uses atomicAdd on a float accumulator: int32/int64 paths can reorder
    # adds at the ULP level. counter is integer → bit-exact.
    mean_array = cp.full(N_COLS, 0.3, dtype=dtype)
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        den = cp.zeros(N_COLS, dtype=dtype)
        counter = cp.zeros(N_COLS, dtype=cp.int32)
        _ac.pre_den_sparse(
            A.indices,
            A.data,
            nnz=A.nnz,
            mean_array=mean_array,
            den=den,
            counter=counter,
            stream=_stream(),
        )
        outs[idx_dtype] = (den, counter)
    _close(outs[np.int32][0], outs[np.int64][0])
    _eq(outs[np.int32][1], outs[np.int64][1])


# ----------------------------- _wilcoxon_binned -----------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_csr_hist(dtype):
    n_groups, n_bins = 3, 16
    rng = np.random.default_rng(SEED)
    gcodes = cp.asarray(rng.integers(0, n_groups, size=N_ROWS).astype(np.int32))
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csr(dtype, idx_dtype)
        hist = cp.zeros((N_COLS, n_groups, n_bins + 1), dtype=cp.uint32)
        _wb.csr_hist(
            A.data,
            A.indices,
            A.indptr,
            gcodes,
            hist,
            n_cells=N_ROWS,
            n_genes=N_COLS,
            n_groups=n_groups,
            n_bins=n_bins,
            bin_low=0.0,
            inv_bin_width=1.0,
            gene_start=0,
            stream=_stream(),
        )
        outs[idx_dtype] = hist
    _eq(outs[np.int32], outs[np.int64])


@pytest.mark.parametrize("dtype", DTYPES)
def test_csc_hist(dtype):
    n_groups, n_bins = 3, 16
    rng = np.random.default_rng(SEED)
    gcodes = cp.asarray(rng.integers(0, n_groups, size=N_ROWS).astype(np.int32))
    outs = {}
    for idx_dtype in (np.int32, np.int64):
        A = _make_csc(dtype, idx_dtype)
        hist = cp.zeros((N_COLS, n_groups, n_bins + 1), dtype=cp.uint32)
        _wb.csc_hist(
            A.data,
            A.indices,
            A.indptr,
            gcodes,
            hist,
            n_cells=N_ROWS,
            n_genes=N_COLS,
            n_groups=n_groups,
            n_bins=n_bins,
            bin_low=0.0,
            inv_bin_width=1.0,
            gene_start=0,
            stream=_stream(),
        )
        outs[idx_dtype] = hist
    _eq(outs[np.int32], outs[np.int64])


# ---------- mixed-precision morans/gearys (AdjIdxT != DataIdxT) -------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_morans_sparse_mixed_indices(dtype):
    """Exercise the cross-typed (AdjIdxT, DataIdxT) instantiations registered
    in autocorr.cu — the same-dtype tests above don't reach them."""
    mean_array = cp.full(N_COLS, 0.3, dtype=dtype)

    A32 = _make_csr(dtype, np.int32)
    adj32 = _make_adjacency(dtype, np.int32)
    num_ref = cp.zeros(N_COLS, dtype=dtype)
    _ac.morans_sparse(
        adj32.indptr,
        adj32.indices,
        adj32.data,
        data_row_ptr=A32.indptr,
        data_col_ind=A32.indices,
        data_values=A32.data,
        n_samples=N_ROWS,
        n_features=N_COLS,
        mean_array=mean_array,
        num=num_ref,
        stream=_stream(),
    )

    for adj_dtype, data_dtype in [(np.int32, np.int64), (np.int64, np.int32)]:
        A = _make_csr(dtype, data_dtype)
        adj = _make_adjacency(dtype, adj_dtype)
        num = cp.zeros(N_COLS, dtype=dtype)
        _ac.morans_sparse(
            adj.indptr,
            adj.indices,
            adj.data,
            data_row_ptr=A.indptr,
            data_col_ind=A.indices,
            data_values=A.data,
            n_samples=N_ROWS,
            n_features=N_COLS,
            mean_array=mean_array,
            num=num,
            stream=_stream(),
        )
        _close_morans_i(num_ref, num)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gearys_sparse_mixed_indices(dtype):
    """Exercise the cross-typed (AdjIdxT, DataIdxT) instantiations registered
    in autocorr.cu — the same-dtype tests above don't reach them."""
    A32 = _make_csr(dtype, np.int32)
    adj32 = _make_adjacency(dtype, np.int32)
    num_ref = cp.zeros(N_COLS, dtype=dtype)
    _ac.gearys_sparse(
        adj32.indptr,
        adj32.indices,
        adj32.data,
        data_row_ptr=A32.indptr,
        data_col_ind=A32.indices,
        data_values=A32.data,
        n_samples=N_ROWS,
        n_features=N_COLS,
        num=num_ref,
        stream=_stream(),
    )

    for adj_dtype, data_dtype in [(np.int32, np.int64), (np.int64, np.int32)]:
        A = _make_csr(dtype, data_dtype)
        adj = _make_adjacency(dtype, adj_dtype)
        num = cp.zeros(N_COLS, dtype=dtype)
        _ac.gearys_sparse(
            adj.indptr,
            adj.indices,
            adj.data,
            data_row_ptr=A.indptr,
            data_col_ind=A.indices,
            data_values=A.data,
            n_samples=N_ROWS,
            n_features=N_COLS,
            num=num,
            stream=_stream(),
        )
        _close(num_ref, num)


# ---------- long-long parameter ABI smoke test ------------------------------


def test_sparse2dense_long_long_max_nnz_not_truncated():
    """``sparse2dense`` binds ``max_nnz`` as ``long long``. Verify a value
    above 2^32 is not silently truncated to int32 by the nanobind layer.

    Mechanism: the kernel uses ``max_nnz`` only to size the y-grid via
    ``strided_grid_y`` (which caps at the device limit); actual work is
    bounded by indptr/indices. If the binding ever regresses to ``int``,
    passing 2^33 from Python would raise OverflowError. With correct
    ``long long`` passing the kernel runs (also exercising the
    strided-grid cap path) and produces the same result as the canonical
    small-``max_nnz`` call.
    """
    dtype = np.float32
    A = _make_csr(dtype, np.int32)
    canonical = int((A.indptr[1:] - A.indptr[:-1]).max())

    out_ref = cp.zeros((N_ROWS, N_COLS), dtype=dtype, order="C")
    _s2d.sparse2dense(
        A.indptr,
        A.indices,
        A.data,
        out=out_ref,
        major=N_ROWS,
        minor=N_COLS,
        c_switch=True,
        max_nnz=canonical,
        stream=_stream(),
    )
    assert cp.any(out_ref != 0), "canonical run produced no output — bad fixture"

    out_huge = cp.zeros((N_ROWS, N_COLS), dtype=dtype, order="C")
    _s2d.sparse2dense(
        A.indptr,
        A.indices,
        A.data,
        out=out_huge,
        major=N_ROWS,
        minor=N_COLS,
        c_switch=True,
        max_nnz=1 << 33,
        stream=_stream(),
    )
    _eq(out_ref, out_huge)
