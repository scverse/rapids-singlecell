from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cupy as cp
import cuvs
import numpy as np
from cupyx.scipy import sparse as cp_sparse
from packaging.version import parse as parse_version
from scipy import sparse as sc_sparse

if TYPE_CHECKING:
    from rapids_singlecell.preprocessing._neighbors import _Algorithms, _Metrics


def _compute_nlist(N):
    base = math.sqrt(N)
    next_pow2 = 2 ** math.ceil(math.log2(base))
    return int(next_pow2 * 2)


def _cuvs_switch():
    return parse_version(cuvs.__version__) > parse_version("24.10")


def _check_neighbors_X(
    X: cp_sparse.spmatrix | sc_sparse.spmatrix | np.ndarray | cp.ndarray,
    algorithm: _Algorithms,
) -> cp_sparse.spmatrix | cp.ndarray | np.ndarray:
    """Check and convert input X to the expected format based on algorithm.

    Parameters
    ----------
    X (array-like or sparse matrix): Input data.
    algorithm (str): The algorithm to be used.
    use_multi_gpu (bool): Whether to use multi-gpu mode.

    Returns
    -------
    X_contiguous (cupy.ndarray or sparse.csr_matrix): Contiguous array or CSR matrix.

    """
    if cp_sparse.issparse(X) or sc_sparse.issparse(X):
        if algorithm != "brute":
            raise ValueError(
                f"Sparse input is not supported for {algorithm} algorithm. Use 'brute' instead."
            )
        X_contiguous = X.tocsr()
    elif algorithm in ["all_neighbors", "mg_ivfflat", "mg_ivfpq"]:
        if isinstance(X, np.ndarray):
            X_contiguous = np.asarray(X, order="C", dtype=np.float32)
        elif isinstance(X, cp.ndarray):
            X_contiguous = cp.ascontiguousarray(X, dtype=cp.float32).get()
        else:
            raise TypeError(
                "Unsupported type for X. Expected ndarray or sparse matrix."
            )
    else:
        if isinstance(X, np.ndarray):
            X_contiguous = cp.asarray(X, order="C", dtype=np.float32)
        elif isinstance(X, cp.ndarray):
            X_contiguous = cp.ascontiguousarray(X, dtype=np.float32)
        else:
            raise TypeError(
                "Unsupported type for X. Expected ndarray or sparse matrix."
            )

    return X_contiguous


def _check_metrics(algorithm: _Algorithms, metric: _Metrics) -> bool:
    """Check if the provided metric is compatible with the chosen algorithm.

    Parameters
    ----------
    algorithm (str): The algorithm to be used.
    metric (str): The metric for distance computation.

    Returns
    -------
    bool: True if the metric is compatible, otherwise ValueError is raised.

    """
    if algorithm == "brute":
        # 'brute' support all metrics, no need to check further.
        return True
    elif algorithm == "cagra":
        if metric not in ["euclidean", "sqeuclidean", "inner_product"]:
            raise ValueError(
                "cagra only supports 'euclidean', 'inner_product' and 'sqeuclidean' metrics."
            )
    elif algorithm in ["ivfpq", "ivfflat", "mg_ivfflat", "mg_ivfpq"]:
        if metric not in ["euclidean", "sqeuclidean", "inner_product", "cosine"]:
            raise ValueError(
                f"{algorithm} only supports 'euclidean', 'sqeuclidean', 'cosine', and 'inner_product' metrics."
            )
    elif algorithm == "nn_descent":
        if metric not in ["euclidean", "sqeuclidean", "cosine", "inner_product"]:
            raise ValueError(
                "nn_descent only supports 'euclidean', 'sqeuclidean', 'inner_product' and 'cosine' metrics."
            )
    elif algorithm == "all_neighbors":
        if metric not in ["euclidean", "sqeuclidean"]:
            raise ValueError(
                "all_neighbors only supports 'euclidean' and 'sqeuclidean' metrics."
            )
    else:
        raise NotImplementedError(f"The {algorithm} algorithm is not implemented yet.")

    return True


def _fix_self_distances(knn_dist: cp.ndarray, metric: _Metrics) -> cp.ndarray:
    """Ensure zero self-distances for all definitionally applicable metrics.

    Parameters
    ----------
    knn_dist (cupy.ndarray): Array of distances to nearest neighbors.
    metric (str): The metric for distance computation.

    Returns
    -------
    knn_dist (cupy.ndarray): Array with self-distances set to zero if definitionally required by metric.

    """
    if metric not in ["inner_product"]:
        knn_dist[:, 0] = 0.0

    return knn_dist


# Empirically, the block-cooperative CUB sort kernel is faster for trim >= 100;
# below this threshold the per-thread top-k kernel has less launch overhead.
_TRIM_SORT_THRESHOLD = 100


def _trimming(
    cnts: cp_sparse.csr_matrix,
    trim: int,
    *,
    kernel: str = "auto",
) -> cp_sparse.csr_matrix:
    from rapids_singlecell._cuda._bbknn_cuda import (
        cut_smaller,
        find_top_k_per_row,
        find_top_k_per_row_sorted,
        sort_tile_size,
    )

    n_rows = cnts.shape[0]
    vals_gpu = cp.zeros(n_rows, dtype=cp.float32)
    stream = cp.cuda.get_current_stream().ptr

    if kernel == "auto":
        if trim >= _TRIM_SORT_THRESHOLD:
            max_row_nnz = int(cp.diff(cnts.indptr).max().get())
            kernel = "sorted" if max_row_nnz <= sort_tile_size() else "thread"
        else:
            kernel = "thread"

    if kernel == "sorted":
        find_top_k_per_row_sorted(
            cnts.data,
            cnts.indptr,
            n_rows=n_rows,
            trim=trim,
            vals=vals_gpu,
            stream=stream,
        )
    elif kernel == "thread":
        find_top_k_per_row(
            cnts.data,
            cnts.indptr,
            n_rows=n_rows,
            trim=trim,
            vals=vals_gpu,
            stream=stream,
        )
    else:
        raise ValueError(f"Unknown trim kernel: {kernel!r}")

    cut_smaller(
        cnts.indptr,
        cnts.indices,
        cnts.data,
        vals=vals_gpu,
        n_rows=n_rows,
        stream=stream,
    )
    cnts.eliminate_zeros()
    return cnts
