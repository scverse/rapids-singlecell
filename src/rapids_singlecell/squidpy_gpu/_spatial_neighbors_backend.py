from __future__ import annotations

from functools import partial

import cupy as cp

from rapids_singlecell._cuda import _spatial_cuda


def _build_kdtree(
    points: cp.ndarray, codes: cp.ndarray | None = None
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """CuPy's KDTree layout per library: tree, index, node boxes, and offsets."""
    n, dims = points.shape
    # Rank float bits as ordered integers, as CuPy kernels flush subnormals.
    # Equal coordinates share a rank, so ties keep CuPy's stable sort order.
    bits = points.view(cp.int32 if points.itemsize == 4 else cp.int64)
    bits = cp.where(bits < 0, -(bits & cp.iinfo(bits.dtype).max), bits)
    ranks = cp.empty((n, dims), dtype=cp.int32)
    for dim in range(dims):
        ranks[:, dim] = cp.searchsorted(cp.sort(bits[:, dim]), bits[:, dim])
    sizes = cp.asarray([n]) if codes is None else cp.bincount(codes)
    segments = cp.zeros(len(sizes) + 1, dtype=cp.int64)
    cp.cumsum(sizes, out=segments[1:])
    index = cp.empty(n, dtype=cp.int32)
    boxes = cp.empty((n, 2, dims), dtype=points.dtype)
    max_length = n if codes is None else int(sizes.max())  # Avoid a sync.
    stream = cp.cuda.get_current_stream().ptr
    _spatial_cuda.build_tree(
        points, ranks, codes, segments, max_length, index, boxes, stream
    )
    return points[index], index, boxes, segments


def _knn_edges(
    coords: cp.ndarray, n_neighs: int, codes: cp.ndarray | None = None
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Exact kNN within each library; distance ties prefer lower indices."""
    return _tree_edges(coords, codes, n_neighs, 0.0)


def _radius_edges(
    coords: cp.ndarray, radius: float, codes: cp.ndarray | None = None
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Edges within ``radius`` (kept as double) in each library."""
    return _tree_edges(coords, codes, 0, radius)


def _tree_edges(coords, codes, k, radius):
    stream = cp.cuda.get_current_stream().ptr
    tree = _build_kdtree(coords, codes)
    search = partial(_spatial_cuda.tree_search, coords, *tree, codes, k, radius)
    offsets = None if k else cp.zeros(len(coords) + 1, dtype=cp.int64)
    if not k:  # Count, then fill with O(observations + edges) memory.
        search(offsets, stream=stream)
    n_edges = len(coords) * k if k else int(offsets.cumsum(out=offsets)[-1])
    rows = cp.empty(n_edges, dtype=cp.int64)
    columns = cp.empty(n_edges, dtype=cp.int64)
    distances = cp.empty(n_edges, dtype=coords.dtype)
    if n_edges:
        search(offsets, rows, columns, distances, stream=stream)
    return rows, columns, distances


def _edge_distances(
    coords: cp.ndarray, rows: cp.ndarray, columns: cp.ndarray
) -> cp.ndarray:
    distances = cp.empty(len(rows), dtype=coords.dtype)
    stream = cp.cuda.get_current_stream().ptr
    _spatial_cuda.edge_distances(coords, rows, columns, distances, stream)
    return distances
