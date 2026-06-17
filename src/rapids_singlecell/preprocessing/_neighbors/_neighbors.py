from __future__ import annotations

from typing import Literal

import cuml.internals.logger as logger
import cupy as cp
import numpy as np
from cuml.manifold.umap import fuzzy_simplicial_set
from cupyx.scipy import sparse as cp_sparse
from scipy import sparse as sc_sparse

from rapids_singlecell._utils import _get_logger_level
from rapids_singlecell.preprocessing._neighbors._algorithms._all_neighbors import (
    _all_neighbors_knn,
)
from rapids_singlecell.preprocessing._neighbors._algorithms._brute import _brute_knn
from rapids_singlecell.preprocessing._neighbors._algorithms._cagra import _cagra_knn
from rapids_singlecell.preprocessing._neighbors._algorithms._ivfflat import (
    _ivf_flat_knn,
)
from rapids_singlecell.preprocessing._neighbors._algorithms._ivfpq import _ivf_pq_knn
from rapids_singlecell.preprocessing._neighbors._algorithms._mg_ivfflat import (
    _mg_ivf_flat_knn,
)
from rapids_singlecell.preprocessing._neighbors._algorithms._mg_ivfpq import (
    _mg_ivf_pq_knn,
)
from rapids_singlecell.preprocessing._neighbors._algorithms._nn_descent import (
    _nn_descent_knn,
)

AnyRandom = None | int | np.random.RandomState

_Algorithms = Literal[
    "brute",
    "cagra",
    "ivfflat",
    "ivfpq",
    "nn_descent",
    "all_neighbors",
    "mg_ivfflat",
    "mg_ivfpq",
]
_MetricsDense = Literal[
    "l2",
    "chebyshev",
    "manhattan",
    "taxicab",
    "correlation",
    "inner_product",
    "euclidean",
    "canberra",
    "lp",
    "minkowski",
    "cosine",
    "jensenshannon",
    "linf",
    "cityblock",
    "l1",
    "haversine",
    "sqeuclidean",
]
_MetricsSparse = Literal[
    "canberra",
    "chebyshev",
    "cityblock",
    "cosine",
    "euclidean",
    "hellinger",
    "inner_product",
    "jaccard",
    "l1",
    "l2",
    "linf",
    "lp",
    "manhattan",
    "minkowski",
    "taxicab",
]
_Metrics = _MetricsDense | _MetricsSparse


KNN_ALGORITHMS = {
    "brute": _brute_knn,
    "cagra": _cagra_knn,
    "ivfflat": _ivf_flat_knn,
    "ivfpq": _ivf_pq_knn,
    "nn_descent": _nn_descent_knn,
    "all_neighbors": _all_neighbors_knn,
    "mg_ivfflat": _mg_ivf_flat_knn,
    "mg_ivfpq": _mg_ivf_pq_knn,
}


def _build_sparse_distances(
    knn_indices: cp.ndarray,
    knn_dist: cp.ndarray,
    *,
    n_obs: int,
) -> sc_sparse.csr_matrix:
    """Build a scipy CSR distance matrix from KNN arrays.

    Parameters
    ----------
    knn_indices
        KNN index array, shape ``(n_obs, k)``.
    knn_dist
        KNN distance array, shape ``(n_obs, k)``.
    n_obs
        Number of observations.

    Returns
    -------
    Scipy CSR distance matrix on host.
    """
    k = knn_dist.shape[1]
    n_nonzero = n_obs * k
    rowptr = cp.arange(0, n_nonzero + 1, k)
    if n_nonzero >= np.iinfo(np.int32).max:
        return sc_sparse.csr_matrix(
            (
                cp.ravel(knn_dist).get(),
                cp.ravel(knn_indices).get(),
                rowptr.get(),
            ),
            shape=(n_obs, n_obs),
        )
    distances = cp_sparse.csr_matrix(
        (cp.ravel(knn_dist), cp.ravel(knn_indices), rowptr),
        shape=(n_obs, n_obs),
    )
    return distances.get()


def _get_connectivities_umap(
    knn_indices: cp.ndarray,
    knn_dist: cp.ndarray,
    *,
    n_obs: int,
    n_neighbors: int,
    random_state: AnyRandom,
    metric: str,
) -> cp_sparse.coo_matrix:
    """UMAP fuzzy simplicial set connectivities."""
    set_op_mix_ratio = 1.0
    local_connectivity = 1.0

    X_conn = cp.zeros((n_obs, 1), dtype=np.float32)
    logger_level = _get_logger_level(logger)
    connectivities = fuzzy_simplicial_set(
        X_conn,
        n_neighbors,
        random_state,
        metric=metric,
        knn_indices=knn_indices,
        knn_dists=knn_dist,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )
    logger.set_level(logger_level)
    return connectivities


def _get_connectivities_gauss(
    knn_indices: cp.ndarray,
    knn_dist: cp.ndarray,
    *,
    n_obs: int,
) -> cp_sparse.csr_matrix:
    """Adaptive Gaussian kernel connectivities (GPU).

    Parameters
    ----------
    knn_indices
        KNN index array, shape ``(n_obs, k)``.
    knn_dist
        KNN distance array, shape ``(n_obs, k)``.
    n_obs
        Number of observations.

    Returns
    -------
    Symmetric CSR connectivity matrix.
    """
    k = knn_indices.shape[1]

    # Per-cell bandwidth from median of squared distances (exclude self at col 0)
    d_sq = knn_dist**2
    sigmas_sq = cp.median(d_sq[:, 1:], axis=1)
    sigmas = cp.sqrt(sigmas_sq)

    # Build sparse CSR from KNN edges (exclude self)
    rows = cp.repeat(cp.arange(n_obs, dtype=cp.int32), k - 1)
    cols = knn_indices[:, 1:].ravel()
    d_sq_vals = d_sq[:, 1:].ravel()

    sig_i = sigmas[rows]
    sig_j = sigmas[cols]
    sigsq_i = sigmas_sq[rows]
    sigsq_j = sigmas_sq[cols]

    den = sigsq_i + sigsq_j
    num = 2.0 * sig_i * sig_j
    vals = cp.sqrt(num / den) * cp.exp(-d_sq_vals / den)

    W = cp_sparse.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs)).tocsr()

    # Symmetrize: W = max(W, W^T)
    W = W.maximum(W.T.tocsr()).tocsr()
    return W


def _get_connectivities_jaccard(
    knn_indices: cp.ndarray,
    *,
    n_obs: int,
    n_neighbors: int,
) -> cp_sparse.csr_matrix:
    """Jaccard connectivities (PhenoGraph method, GPU).

    Parameters
    ----------
    knn_indices
        KNN index array, shape ``(n_obs, n_neighbors)``.
    n_obs
        Number of observations.
    n_neighbors
        Number of nearest neighbors (including self).

    Returns
    -------
    Symmetric CSR connectivity matrix.
    """
    k_no_self = n_neighbors - 1

    # Binary adjacency (self excluded)
    rows_adj = cp.repeat(cp.arange(n_obs, dtype=cp.int32), k_no_self)
    cols_adj = knn_indices[:, 1:].ravel()
    data_adj = cp.ones(n_obs * k_no_self, dtype=cp.float32)
    adjacency = cp_sparse.csr_matrix(
        (data_adj, (rows_adj, cols_adj)), shape=(n_obs, n_obs)
    )

    # For each directed KNN edge (i->j), compute shared neighbor count
    i_idx = rows_adj
    j_idx = cols_adj
    rows_i = adjacency[i_idx]
    rows_j = adjacency[j_idx]
    shared = cp.asarray(rows_i.multiply(rows_j).sum(axis=1)).ravel()

    # Jaccard: |N(i) & N(j)| / (2*(k-1) - |N(i) & N(j)|)
    jaccard_vals = shared / (2 * k_no_self - shared)

    # Filter zeros and build sparse matrix
    mask = jaccard_vals != 0
    W = cp_sparse.coo_matrix(
        (jaccard_vals[mask], (i_idx[mask], j_idx[mask])),
        shape=(n_obs, n_obs),
    ).tocsr()

    # Symmetrize by averaging
    W = (W + W.T) / 2
    return W


def _calc_connectivities(
    knn_indices: cp.ndarray,
    knn_dist: cp.ndarray,
    *,
    n_obs: int,
    n_neighbors: int,
    random_state: AnyRandom,
    metric: str,
    method: Literal["umap", "gauss", "jaccard"] = "umap",
) -> cp_sparse.spmatrix:
    """Compute connectivities from KNN arrays.

    Parameters
    ----------
    knn_indices
        KNN index array, shape ``(n_obs, k)``.
    knn_dist
        KNN distance array, shape ``(n_obs, k)``.
    n_obs
        Number of observations.
    n_neighbors
        Number of nearest neighbors.
    random_state
        Random seed (passed through to UMAP fuzzy simplicial set).
    metric
        Distance metric name.
    method
        Method for computing connectivities.

    Returns
    -------
    CuPy sparse matrix on GPU.
    """
    if method == "gauss":
        return _get_connectivities_gauss(
            knn_indices,
            knn_dist,
            n_obs=n_obs,
        )
    if method == "jaccard":
        return _get_connectivities_jaccard(
            knn_indices,
            n_obs=n_obs,
            n_neighbors=n_neighbors,
        )
    return _get_connectivities_umap(
        knn_indices,
        knn_dist,
        n_obs=n_obs,
        n_neighbors=n_neighbors,
        random_state=random_state,
        metric=metric,
    )
