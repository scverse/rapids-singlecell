from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp

from rapids_singlecell.preprocessing._neighbors._helper import _cuvs_switch

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rapids_singlecell.preprocessing._neighbors import _Metrics


def _cagra_knn(
    X: cp.ndarray,
    Y: cp.ndarray,
    k: int,
    *,
    metric: _Metrics,
    metric_kwds: Mapping,
    algorithm_kwds: Mapping,
) -> tuple[cp.ndarray, cp.ndarray]:
    if not _cuvs_switch():
        try:
            from pylibraft.neighbors import cagra
        except ImportError as e:
            raise ImportError(
                "The 'cagra' module is not available in your current RAFT installation. "
                "Please update RAFT to a version that supports 'cagra'."
            ) from e
        from pylibraft.common import DeviceResources

        resources = DeviceResources()
        build_kwargs = {"handle": resources}
        search_kwargs = {"handle": resources}
        warnings.warn(
            "Using `pylibraft` for ANN was removed in RAFT 24.12 and is deprecated in rapids-singlecell. Please update RAPIDS to use `cuvs` for ANN.",
            stacklevel=2,
        )
    else:
        from cuvs.neighbors import cagra

        resources = None
        build_kwargs = {}
        search_kwargs = {}

    if metric == "euclidean":
        metric_to_use = "sqeuclidean"
    else:
        metric_to_use = metric
    build_params = cagra.IndexParams(metric=metric_to_use, graph_degree=k, build_algo="nn_descent")
    index = cagra.build(build_params, X, **build_kwargs)

    search_params = cagra.SearchParams()
    distances, neighbors = cagra.search(search_params, index, Y, k, **search_kwargs)

    if resources is not None:
        resources.sync()

    neighbors = cp.asarray(neighbors)
    distances = cp.asarray(distances)

    if metric == "euclidean":
        distances = cp.sqrt(distances)

    return neighbors, distances
