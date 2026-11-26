from __future__ import annotations

from .pertpy_gpu import (
    Distance,
    DistanceTest,
    GuideAssignment,
    Mixscale,
    Mixscape,
)
from .pertpy_gpu import (
    MeanVar as MeanVar,
)

__all__ = ["Distance", "DistanceTest", "GuideAssignment", "Mixscale", "Mixscape"]

__deprecated_exports__ = {
    "MeanVar": "Deprecated; do not use in new analyses.",
}

# Public class members are part of the installed API contract used by agent tooling.
# Keep these in sync with docs/api/pertpy_gpu.md.
__api_members__ = {
    "Distance": [
        "__call__",
        "pairwise",
        "onesided_distances",
        "contrast_distances",
        "create_contrasts",
        "bootstrap",
    ],
    "DistanceTest": ["__call__", "test_xy", "test_precomputed"],
    "GuideAssignment": [
        "assign_by_threshold",
        "assign_to_max_guide",
        "assign_mixture_model",
    ],
    "Mixscale": ["perturbation_signature", "mixscale"],
    "Mixscape": ["perturbation_signature", "mixscape", "lda"],
}
