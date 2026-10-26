from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp

from rapids_singlecell.preprocessing._neighbors._helper import _compute_nlist

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

    from rapids_singlecell.preprocessing._neighbors import _Metrics


def _mg_ivf_flat_knn(
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
    *,
    metric: _Metrics,
    metric_kwds: Mapping,
    algorithm_kwds: Mapping,
) -> tuple[cp.ndarray, cp.ndarray]:
    try:
        from cuvs.neighbors.mg import ivf_flat as mg_ivf_flat
    except ImportError:
        raise ImportError(
            "The 'mg_ivf_flat' algorithm is only available in cuvs >= 25.10. Please update your cuvs installation."
        )
    distribution_mode = algorithm_kwds.get("distribution_mode", "replicated")
    assert distribution_mode in ["replicated", "shared"], "Invalid distribution mode"
    n_lists = algorithm_kwds.get("n_lists", _compute_nlist(X.shape[0]))
    n_probes = algorithm_kwds.get("n_probes", 20)
    # Build multi-GPU index
    build_params = mg_ivf_flat.IndexParams(
        distribution_mode=distribution_mode,
        n_lists=n_lists,
        metric=metric,
    )
    index = mg_ivf_flat.build(build_params, X)

    # Search with multi-GPU
    search_params = mg_ivf_flat.SearchParams(
        search_mode="load_balancer",
        merge_mode="merge_on_root_rank",
        n_probes=n_probes,
    )
    distances, neighbors = mg_ivf_flat.search(search_params, index, Y, k=k)
    distances = cp.asarray(distances)
    neighbors = cp.asarray(neighbors)
    return neighbors, distances
