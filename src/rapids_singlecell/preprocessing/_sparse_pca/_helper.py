from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp

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


def _compute_cov(
    cov_result: cp.ndarray, gram_matrix: cp.ndarray, mean_x: cp.ndarray
) -> cp.ndarray:
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
        raise ValueError(
            "There are genes with zero expression. Please remove them before running PCA."
        )
