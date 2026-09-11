from __future__ import annotations

import math

import cupy as cp
from cupyx.scipy.spatial._kdtree_utils import KD_MODULE

_BLOCK_SIZE = 128


def _build_kdtree(points: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """Fix CuPy's KDTree tag-precision bug by keeping sorting tags as integers."""
    x = points.copy()
    track_idx = cp.arange(x.shape[0], dtype=cp.int64)
    tags = cp.zeros(x.shape[0], dtype=cp.int64)
    length, dims = x.shape
    n_iter = int(math.log2(length))
    n_blocks = (length + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    update_tags = KD_MODULE.get_function("update_tags")

    level = 0
    for level in range(n_iter):
        dim = level % dims
        # Stable sorts keep tags integral, including beyond float32's 2**24.
        idx = cp.argsort(x[:, dim])
        idx = idx[cp.argsort(tags[idx])]
        x = x[idx]
        tags = tags[idx]
        track_idx = track_idx[idx]
        update_tags((n_blocks,), (_BLOCK_SIZE,), (length, level, tags))

    if n_iter > 1:
        level += 1

    dim = level % dims
    idx = cp.argsort(x[:, dim])
    idx = idx[cp.argsort(tags[idx])]
    return x[idx], track_idx[idx]
