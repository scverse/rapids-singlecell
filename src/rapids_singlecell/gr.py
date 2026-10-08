from __future__ import annotations

import sys

from .squidpy_gpu import (
    SpatialNeighborsResult,
    calculate_niche_cellcharter,
    calculate_niche_neighborhood,
    calculate_niche_utag,
    co_occurrence,
    ligrec,
    neighbors,
    ripley,
    spatial_autocorr,
    spatial_neighbors_delaunay,
    spatial_neighbors_from_builder,
    spatial_neighbors_grid,
    spatial_neighbors_knn,
    spatial_neighbors_radius,
)
from .squidpy_gpu import (
    calculate_niche as calculate_niche,
)

# Importable like ``squidpy.gr.neighbors``.
sys.modules[f"{__name__}.neighbors"] = neighbors

__all__ = [
    "SpatialNeighborsResult",
    "calculate_niche_cellcharter",
    "calculate_niche_neighborhood",
    "calculate_niche_utag",
    "co_occurrence",
    "ligrec",
    "ripley",
    "spatial_autocorr",
    "spatial_neighbors_delaunay",
    "spatial_neighbors_from_builder",
    "spatial_neighbors_grid",
    "spatial_neighbors_knn",
    "spatial_neighbors_radius",
]

__deprecated_exports__ = {
    "calculate_niche": calculate_niche.__deprecated__,
}

# Public class members are part of the installed API contract used by agent tooling.
# ``SpatialNeighborsResult`` is a plain ``NamedTuple`` with no public methods.
__api_members__: dict[str, list[str]] = {
    "SpatialNeighborsResult": [],
}
