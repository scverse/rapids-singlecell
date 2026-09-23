from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cupy as cp
import cuvs
import numpy as np
from packaging.version import parse as parse_version

from rapids_singlecell.preprocessing._neighbors._helper import _compute_nlist

try:
    from cuvs.neighbors import all_neighbors
except ImportError:
    all_neighbors = None
if TYPE_CHECKING:
    from collections.abc import Mapping

    from rapids_singlecell.preprocessing._neighbors import _Metrics


_CUVS_HOST_OUTPUT_MIN_VERSION = parse_version("26.08")


def _default_overlap_factor(n_clusters: int) -> int:
    """Overlap needed to hold recall as the dataset is split into more clusters."""
    if n_clusters <= 1:
        return 1
    return max(2, math.ceil(math.log2(n_clusters)))


def _all_neighbors_batching(algorithm_kwds: Mapping) -> tuple[int, int]:
    """Resolve ``(n_clusters, overlap_factor)`` for the cuVS all-neighbors build."""
    n_devices = cp.cuda.runtime.getDeviceCount()
    n_clusters = algorithm_kwds.get("n_clusters")
    overlap_factor = algorithm_kwds.get("overlap_factor")
    if n_clusters is None:
        n_clusters = 1 if n_devices == 1 else n_devices
        while n_clusters > 1 and n_clusters <= (
            _default_overlap_factor(n_clusters)
            if overlap_factor is None
            else overlap_factor
        ):
            n_clusters += n_devices
    if overlap_factor is None:
        overlap_factor = _default_overlap_factor(n_clusters)
        if n_clusters > 1:
            overlap_factor = min(overlap_factor, n_clusters - 1)
    if n_clusters > 1 and overlap_factor >= n_clusters:
        raise ValueError(
            f"'n_clusters' ({n_clusters}) must be greater than 'overlap_factor' "
            f"({overlap_factor}) when batching the all_neighbors build."
        )
    return n_clusters, overlap_factor


def _all_neighbors_knn(
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
    *,
    metric: _Metrics,
    metric_kwds: Mapping,
    algorithm_kwds: Mapping,
) -> tuple[cp.ndarray, cp.ndarray]:
    if all_neighbors is None:
        raise ImportError(
            "The 'all_neighbors' algorithm is only available in cuvs >= 25.10. "
            "Please update your cuvs installation."
        )
    algo = algorithm_kwds.get("algo", "nn_descent")
    n_clusters, overlap_factor = _all_neighbors_batching(algorithm_kwds)
    use_host_output = (
        n_clusters > 1
        and parse_version(cuvs.__version__) >= _CUVS_HOST_OUTPUT_MIN_VERSION
    )
    if use_host_output:
        from cuvs.common import MultiGpuResources

        res = MultiGpuResources()
    else:
        from cuvs.common import Resources

        res = Resources()
    cuvs_metric = "sqeuclidean" if metric == "euclidean" else metric
    if algo == "ivf_pq" or algo == "ivfpq":
        from cuvs.neighbors import ivf_pq

        algo = "ivf_pq"
        if cuvs_metric != "sqeuclidean":
            raise ValueError(
                f"all_neighbors with algo='ivf_pq' only supports 'euclidean' and "
                f"'sqeuclidean' metrics, got {metric!r}. Use algo='nn_descent' instead."
            )
        n_lists = algorithm_kwds.get("n_lists", _compute_nlist(X.shape[0]))
        ivf_pq_params = ivf_pq.IndexParams(n_lists=n_lists, metric=cuvs_metric)
        nn_descent_params = None
    elif algo == "nn_descent":
        from cuvs.neighbors import nn_descent

        graph_degree = max(algorithm_kwds.get("graph_degree", 64), k)
        intermediate_graph_degree = algorithm_kwds.get(
            "intermediate_graph_degree", max(128, int(1.5 * graph_degree))
        )
        intermediate_graph_degree = max(intermediate_graph_degree, graph_degree)
        nn_descent_params = nn_descent.IndexParams(
            graph_degree=graph_degree,
            intermediate_graph_degree=intermediate_graph_degree,
            metric=cuvs_metric,
        )
        ivf_pq_params = None
    else:
        raise ValueError(f"Invalid algorithm: {algo}")
    build_params = all_neighbors.AllNeighborsParams(
        algo=algo,
        overlap_factor=overlap_factor,
        n_clusters=n_clusters,
        metric=cuvs_metric,
        ivf_pq_params=ivf_pq_params,
        nn_descent_params=nn_descent_params,
    )
    # Host outputs use cuVS's synchronized merge path. Older releases only accept
    # device outputs, so keep batching on one GPU to avoid concurrent writes.
    output_module = np if use_host_output else cp
    neighbors = output_module.zeros([X.shape[0], k], dtype=np.int64)
    distances = output_module.zeros([X.shape[0], k], dtype=np.float32)

    all_neighbors.build(
        dataset=X,
        k=k,
        params=build_params,
        indices=neighbors,
        distances=distances,
        resources=res,
    )
    neighbors = cp.asarray(neighbors, dtype=np.int32)
    distances = cp.asarray(distances)
    if metric == "euclidean":
        distances = cp.sqrt(distances)
    return neighbors, distances
