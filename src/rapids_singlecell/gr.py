from __future__ import annotations

from .squidpy_gpu import (
    calculate_niche as calculate_niche,
)
from .squidpy_gpu import (
    calculate_niche_cellcharter,
    calculate_niche_neighborhood,
    calculate_niche_utag,
    co_occurrence,
    ligrec,
    spatial_autocorr,
)

__all__ = [
    "calculate_niche_cellcharter",
    "calculate_niche_neighborhood",
    "calculate_niche_utag",
    "co_occurrence",
    "ligrec",
    "spatial_autocorr",
]

__deprecated_exports__ = {
    "calculate_niche": calculate_niche.__deprecated__,
}
