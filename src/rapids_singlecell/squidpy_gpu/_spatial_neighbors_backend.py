from __future__ import annotations

import cupy as cp
from cupyx.scipy.spatial import KDTree

from rapids_singlecell._cuda import _spatial_cuda


def _knn_edges(
    coords: cp.ndarray, n_neighs: int
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Exact GPU search using CuPy's left-balanced KDTree."""
    tree = KDTree(coords)
    rows = cp.repeat(cp.arange(len(coords), dtype=cp.int64), n_neighs)
    columns = cp.empty(len(rows), dtype=cp.int64)
    distances = cp.empty(len(rows), dtype=coords.dtype)
    _spatial_cuda.knn(
        coords,
        tree.tree,
        tree.index,
        k=n_neighs,
        columns=columns,
        distances=distances,
        stream=cp.cuda.get_current_stream().ptr,
    )
    return rows, columns, distances


def _radius_edges(
    coords: cp.ndarray, radius: float
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Count then fill radius edges with O(observations + edges) GPU memory."""
    tree = KDTree(coords)
    counts = cp.empty(len(coords), dtype=cp.int64)
    stream = cp.cuda.get_current_stream().ptr
    # Keep the caller's double radius, including for float32 coordinates.
    _spatial_cuda.radius_count(
        coords,
        tree.tree,
        tree.index,
        radius=radius,
        counts=counts,
        stream=stream,
    )
    offsets = cp.empty(len(coords) + 1, dtype=cp.int64)
    offsets[0] = 0
    cp.cumsum(counts, out=offsets[1:])
    n_edges = int(offsets[-1])
    rows = cp.empty(n_edges, dtype=cp.int64)
    columns = cp.empty(n_edges, dtype=cp.int64)
    distances = cp.empty(n_edges, dtype=coords.dtype)
    if n_edges:
        _spatial_cuda.radius_fill(
            coords,
            tree.tree,
            tree.index,
            radius=radius,
            offsets=offsets,
            rows=rows,
            columns=columns,
            distances=distances,
            stream=stream,
        )
    return rows, columns, distances


def _edge_distances(
    coords: cp.ndarray, rows: cp.ndarray, columns: cp.ndarray
) -> cp.ndarray:
    distances = cp.empty(len(rows), dtype=coords.dtype)
    _spatial_cuda.edge_distances(
        coords,
        rows,
        columns,
        distances=distances,
        stream=cp.cuda.get_current_stream().ptr,
    )
    return distances
