from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp

from rapids_singlecell.preprocessing._neighbors._helper import _check_knn_complete

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rapids_singlecell.preprocessing._neighbors import _Metrics


try:
    from cuvs.neighbors import nn_descent
except ImportError:
    nn_descent = None


def _nn_descent_knn(
    X: cp.ndarray,
    Y: cp.ndarray,
    k: int,
    *,
    metric: _Metrics,
    metric_kwds: Mapping,
    algorithm_kwds: Mapping,
) -> tuple[cp.ndarray, cp.ndarray]:
    if nn_descent is None:
        raise ImportError(
            "The 'nn_descent' algorithm is only available in cuvs >= 25.02. Please update your cuvs installation."
        )

    # cuVS's default (96) is below the recommended 1.5 * k for k > 64, and fails for k > 96
    default = max(nn_descent.IndexParams().intermediate_graph_degree, int(1.5 * k))
    intermediate_graph_degree = max(algorithm_kwds.get("intermediate_graph_degree", default), k)

    idxparams = nn_descent.IndexParams(
        graph_degree=k,
        intermediate_graph_degree=intermediate_graph_degree,
        metric="sqeuclidean" if metric == "euclidean" else metric,
    )
    idx = nn_descent.build(
        idxparams,
        dataset=X,
    )
    neighbors = cp.array(idx.graph).astype(cp.uint32)
    # the distance kernels index X with these IDs
    _check_knn_complete(neighbors, n_index=X.shape[0], algorithm="nn_descent")
    if metric == "euclidean" or metric == "sqeuclidean":
        from rapids_singlecell._cuda._nn_descent_cuda import (
            sqeuclidean as dist_func,
        )
    elif metric == "cosine":
        from rapids_singlecell._cuda._nn_descent_cuda import (
            cosine as dist_func,
        )
    elif metric == "inner_product":
        from rapids_singlecell._cuda._nn_descent_cuda import (
            inner as dist_func,
        )
    # grid_size = (X.shape[0] + 32 - 1) // 32
    distances = cp.zeros((X.shape[0], neighbors.shape[1]), dtype=cp.float32)

    dist_func(
        X,
        out=distances,
        pairs=neighbors,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        n_neighbors=neighbors.shape[1],
        stream=cp.cuda.get_current_stream().ptr,
    )
    if metric == "euclidean":
        distances = cp.sqrt(distances)
    if metric in ("cosine", "euclidean", "sqeuclidean"):
        # Add self-neighbors and self-distances for distance metrics.
        # This is not needed for inner_product, as it is a similarity metric.
        add_self_neighbors = cp.arange(X.shape[0], dtype=cp.uint32)
        neighbors = cp.concatenate((add_self_neighbors[:, None], neighbors[:, :-1]), axis=1)
        add_self_distances = cp.zeros((X.shape[0], 1), dtype=cp.float32)
        distances = cp.concatenate((add_self_distances, distances[:, :-1]), axis=1)
    return neighbors, distances
