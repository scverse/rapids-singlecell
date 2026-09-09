from __future__ import annotations

from typing import Literal

import cuml.internals.logger as logger
import cupy as cp
import numpy as np
from cuml.manifold.umap import fuzzy_simplicial_set
from cupyx.scipy import sparse as cp_sparse
from scipy import sparse as sc_sparse

from rapids_singlecell._utils import _get_logger_level
from rapids_singlecell._utils._random import _seed_from_rng
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
    rng: np.random.Generator,
) -> cp_sparse.coo_matrix:
    """UMAP fuzzy simplicial set connectivities.

    The graph is built from the precomputed ``knn_indices``/``knn_dist``, so the
    metric is never recomputed here. Forwarding it would only make cuML reject
    the metrics it does not know itself, such as ``inner_product``.
    """
    set_op_mix_ratio = 1.0
    local_connectivity = 1.0

    X_conn = cp.zeros((n_obs, 1), dtype=np.float32)
    logger_level = _get_logger_level(logger)
    connectivities = fuzzy_simplicial_set(
        X_conn,
        n_neighbors,
        # cuML seeds its fuzzy simplicial set, so draw the seed right here
        _seed_from_rng(rng),
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
    from rapids_singlecell._cuda._jaccard_cuda import jaccard_shared_counts

    knn = cp.ascontiguousarray(knn_indices, dtype=cp.int32)

    # Jaccard weight per KNN entry (i -> j), straight from the KNN lists
    # (no replicated adjacency matrix, so no int32 nnz overflow).
    jaccard_vals = cp.empty(n_obs * n_neighbors, dtype=cp.float32)
    jaccard_shared_counts(
        knn,
        n_obs=n_obs,
        k=n_neighbors,
        jaccard_vals=jaccard_vals,
        stream=cp.cuda.get_current_stream().ptr,
    )

    i_idx = cp.repeat(cp.arange(n_obs, dtype=cp.int32), n_neighbors)
    j_idx = knn.ravel()

    # Filter zeros and build sparse matrix
    mask = jaccard_vals != 0
    W = cp_sparse.coo_matrix(
        (jaccard_vals[mask], (i_idx[mask], j_idx[mask])),
        shape=(n_obs, n_obs),
    ).tocsr()

    # Symmetrize by averaging
    W = (W + W.T) / 2
    return W


def _inner_product_distances(
    knn_indices: cp.ndarray,
    similarities: cp.ndarray,
    *,
    batch_codes: cp.ndarray | None = None,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Prepare row-local score gaps with one self-neighbor for graph weighting."""
    n_obs, k = knn_indices.shape
    self_indices = cp.arange(n_obs, dtype=knn_indices.dtype)[:, None]
    scores = cp.where(knn_indices == self_indices, -cp.inf, similarities)
    if batch_codes is not None:
        # Replace self within its own batch, preserving BBKNN's batch balance.
        same_batch = batch_codes[knn_indices] == batch_codes[:, None]
        replace = cp.argmin(cp.where(same_batch, scores, cp.inf), axis=1)
        scores[cp.arange(n_obs), replace] = -cp.inf
    order = cp.argsort(-scores, axis=1)[:, : k - 1]
    indices = cp.take_along_axis(knn_indices, order, axis=1)
    scores = cp.take_along_axis(scores, order, axis=1)
    gaps = scores[:, :1] - scores
    if k > 1:
        # A positive offset keeps UMAP's rho on the best nonself neighbor;
        # using the local span preserves score differences without a global shift.
        span = gaps[:, -1:]
        gaps += cp.where(span > 0, span, 1)
    return (
        cp.concatenate((self_indices, indices), axis=1),
        cp.concatenate((cp.zeros((n_obs, 1), dtype=similarities.dtype), gaps), axis=1),
    )


def _calc_connectivities(
    knn_indices: cp.ndarray,
    knn_dist: cp.ndarray,
    *,
    n_obs: int,
    n_neighbors: int,
    rng: np.random.Generator,
    method: Literal["umap", "gauss", "jaccard"] = "umap",
    metric: _Metrics = "euclidean",
    batch_codes: cp.ndarray | None = None,
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
    rng
        Random generator (a seed is drawn for the UMAP fuzzy simplicial set).
    method
        Method for computing connectivities.
    metric
        Search metric; inner-product similarities are converted for weighting.
    batch_codes
        Per-cell batch codes for preserving BBKNN's self-neighbor allocation.

    Returns
    -------
    CuPy sparse matrix on GPU.
    """
    if metric == "inner_product" and method != "jaccard":
        knn_indices, knn_dist = _inner_product_distances(
            knn_indices, knn_dist, batch_codes=batch_codes
        )
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
        rng=rng,
    )
