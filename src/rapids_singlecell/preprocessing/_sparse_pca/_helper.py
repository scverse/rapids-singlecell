from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
from cupyx.scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from cupyx.scipy.sparse import spmatrix
from rapids_singlecell._cuda import _spca_cuda as _spca


def _copy_gram(gram_matrix: cp.ndarray, n_cols: int) -> cp.ndarray:
    _spca.copy_upper_to_lower(
        out=gram_matrix,
        ncols=n_cols,
        stream=cp.cuda.get_current_stream().ptr,
    )
    return gram_matrix


def _compute_cov(cov_result: cp.ndarray, gram_matrix: cp.ndarray, mean_x: cp.ndarray) -> cp.ndarray:
    _spca.cov_from_gram(
        gram_matrix,
        mean_x,
        mean_x,
        cov=cov_result,
        ncols=gram_matrix.shape[0],
        stream=cp.cuda.get_current_stream().ptr,
    )
    return cov_result


def _check_matrix_for_zero_genes(X: spmatrix) -> None:
    gene_ex = cp.zeros(X.shape[1], dtype=cp.int32)
    if X.nnz > 0:
        _spca.check_zero_genes(
            X.indices,
            out=gene_ex,
            nnz=X.nnz,
            num_genes=X.shape[1],
            stream=cp.cuda.get_current_stream().ptr,
        )
    if cp.any(gene_ex == 0):
        raise ValueError("There are genes with zero expression. Please remove them before running PCA.")


_sum_csr_duplicates = cp.ElementwiseKernel(  # type: ignore[attr-defined]
    "int64 start, int64 stop, raw int64 order, raw T data",
    "T total",
    """
    double sum = 0;
    for (long long j = start; j < stop; ++j) sum += data[order[j]];
    total = sum;
    """,
    "pca_sum_csr_duplicates",
)


def _canonical_csr(X):
    if X.has_canonical_format or not X.nnz:
        return X
    # Stable sorting keeps repeated entries in input order. CuPy's
    # sum_duplicates uses floating-point atomics for this reduction.
    keys = cp.searchsorted(X.indptr[1:], cp.arange(X.nnz, dtype=X.indptr.dtype), side="right")
    keys *= X.shape[1]
    keys += X.indices
    order = cp.argsort(keys, kind="stable")
    bounds = cp.flatnonzero(cp.diff(keys[order], prepend=-1, append=-1))
    starts = bounds[:-1]
    data = _sum_csr_duplicates(starts, bounds[1:], order, X.data)
    indptr = cp.searchsorted(starts, X.indptr).astype(X.indptr.dtype, copy=False)
    indices = X.indices[order[starts]]
    result = csr_matrix((data, indices, indptr), shape=X.shape)
    result.indices, result.indptr = indices, indptr
    result.has_canonical_format = True
    return result


def _csr_mean(x):
    rows, cols = x.shape
    partial = cp.empty((min(1024, rows), cols), dtype=cp.float64)
    _spca.pca_column_sums(x.indptr, x.indices, x.data, partial, cp.cuda.get_current_stream().ptr)
    return partial.sum(axis=0) / rows
