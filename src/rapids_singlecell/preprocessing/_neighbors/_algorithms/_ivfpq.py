from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp

from rapids_singlecell.preprocessing._neighbors._algorithms._brute import _brute_knn
from rapids_singlecell.preprocessing._neighbors._helper import (
    _IVF_PQ_MIN_ROWS,
    _batched_search,
    _compute_nlist,
    _cuvs_switch,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rapids_singlecell.preprocessing._neighbors import _Metrics


def _ivf_pq_knn(
    X: cp.ndarray,
    Y: cp.ndarray,
    k: int,
    *,
    metric: _Metrics,
    metric_kwds: Mapping,
    algorithm_kwds: Mapping,
) -> tuple[cp.ndarray, cp.ndarray]:
    if X.shape[0] < _IVF_PQ_MIN_ROWS:  # too few rows to train the PQ codebooks
        return _brute_knn(X, Y, k, metric=metric, metric_kwds=metric_kwds, algorithm_kwds=algorithm_kwds)
    if not _cuvs_switch():
        from pylibraft.common import DeviceResources
        from pylibraft.neighbors import ivf_pq

        resources = DeviceResources()
        build_kwargs = {"handle": resources}
        search_kwargs = {"handle": resources}
        warnings.warn(
            "Using `pylibraft` for ANN was removed in RAFT 24.12 and is deprecated in rapids-singlecell. Please update RAPIDS to use `cuvs` for ANN.",
            stacklevel=2,
        )
    else:
        from cuvs.neighbors import ivf_pq

        resources = None
        build_kwargs = {}
        search_kwargs = {}

    # Extract n_lists and nprobes from algorithm_kwds, with defaults
    n_lists = algorithm_kwds.get("n_lists", _compute_nlist(X.shape[0]))
    n_probes = algorithm_kwds.get("n_probes", 20)

    index_params = ivf_pq.IndexParams(n_lists=n_lists, metric=metric)
    index = ivf_pq.build(index_params, X, **build_kwargs)
    # Create SearchParams with nprobes if provided
    search_params = ivf_pq.SearchParams(n_probes=n_probes)
    distances, neighbors = _batched_search(lambda q: ivf_pq.search(search_params, index, q, k, **search_kwargs), Y)
    if resources is not None:
        resources.sync()

    distances = cp.asarray(distances)
    neighbors = cp.asarray(neighbors)

    return neighbors, distances
