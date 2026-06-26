from __future__ import annotations

import cupy as cp


@cp.fuse
def _calc_R(term: float, dotproduct: cp.ndarray) -> cp.ndarray:
    return cp.exp(term * (1 - dotproduct))
