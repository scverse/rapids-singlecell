from __future__ import annotations

from unittest.mock import Mock

import cupy as cp
import numpy as np
import pytest
from cupyx.scipy import sparse as cp_sparse
from cupyx.scipy.spatial import KDTree
from scipy.spatial import ConvexHull, Delaunay

from rapids_singlecell.squidpy_gpu import _spatial_delaunay as delaunay
from rapids_singlecell.squidpy_gpu import _spatial_neighbors_backend as backend
from rapids_singlecell.squidpy_gpu._spatial_graph import build_adjacency, build_graphs

_UNEXPECTED = Mock(side_effect=AssertionError("Unexpected call"))


def _subtree_boxes(tree):
    boxes = np.stack([tree, tree], axis=1)
    for child in range(len(tree) - 1, 0, -1):
        low, high = boxes[(child - 1) // 2]
        np.minimum(low, boxes[child, 0], out=low)
        np.maximum(high, boxes[child, 1], out=high)
    return boxes


@pytest.mark.parametrize("sizes", [[4096], [1, 2, 3, 0, 7, 1025, 4097]])
def test_kdtrees_match_cupy_per_library(sizes):
    # Rounded coordinates add ties that must keep CuPy's stable sort order.
    rng = np.random.default_rng(5)
    codes = rng.permutation(np.repeat(np.arange(len(sizes)), sizes))
    points = cp.asarray(rng.uniform(size=(len(codes), 2)).round(2), dtype=cp.float32)
    library = cp.asarray(codes, dtype=cp.int32) if len(sizes) > 1 else None
    tree, index, boxes, segments = backend._build_kdtree(points, library)
    assert tree.dtype == boxes.dtype == points.dtype
    np.testing.assert_array_equal(segments.get(), np.r_[0, np.cumsum(sizes)])
    for code in np.flatnonzero(sizes):
        group = np.flatnonzero(codes == code)
        part = slice(*segments[code : code + 2].tolist())
        expected = KDTree(points[cp.asarray(group)].astype(cp.float64))
        reference = expected.tree.get()
        np.testing.assert_array_equal(tree[part].get(), reference)
        np.testing.assert_array_equal(index[part].get(), group[expected.index.get()])
        np.testing.assert_array_equal(boxes[part].get(), _subtree_boxes(reference))


def _searches(dtype, dimensions, offset, scale):
    points = np.random.default_rng(21).normal(size=(40, dimensions)) * scale + offset
    points[-1] = points[0]
    searches = [("knn", 8), ("knn", 9), ("knn", 17), ("knn", 33)]  # Top-k buckets.
    return [(dtype, points, *s) for s in [*searches, ("radius", 1.25 * scale)]]


_OUTLIER = np.random.default_rng(8).uniform(size=(2000, 2))
_OUTLIER[0] = 1e4  # Only bounding boxes, not split lines, prune its search.


@pytest.mark.parametrize(
    "dtype,points,method,parameter",
    [
        *_searches(np.float32, 2, 1e6, 1),
        *_searches(np.float64, 3, 1e12, 1),
        *_searches(np.float32, 2, 0, 1e-30),
        *_searches(np.float64, 3, 0, 1e-200),
        *_searches(np.float64, 4, 0, 1e200),
        (np.float32, _OUTLIER, "knn", 6),
        (np.float32, _OUTLIER, "radius", 0.05),
        # Radii stay double: the largest double below 1 excludes length 1.
        *[
            (dtype, [[0, 0], [0, 0], [1, 0]], "radius", radius)
            for dtype in (np.float32, np.float64)
            for radius in (0, np.nextafter(1.0, 0.0), 1)
        ],
    ],
)
def test_tree_matches_dense_search(dtype, points, method, parameter):
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        coords = cp.asarray(points, dtype=dtype)
        rows, cols, distances = getattr(backend, f"_{method}_edges")(coords, parameter)
        # Fused edge distances use the search's formula, exact at any scale.
        fused = backend._edge_distances(coords, rows, cols.astype(cp.int32))
    stream.synchronize()
    direct = cp.zeros((len(coords), len(coords)), dtype=dtype)
    for dim in range(coords.shape[1]):
        cp.hypot(direct, coords[:, dim, None] - coords[None, :, dim], out=direct)
    cp.fill_diagonal(direct, cp.inf)
    keys = rows * len(coords) + cols
    assert distances.dtype == fused.dtype == coords.dtype
    assert bool((rows != cols).all()) and bool((cp.diff(rows) >= 0).all())
    assert len(cp.unique(keys)) == len(rows)
    if method == "knn":
        # Ties may choose different indices, but ascending distances must match.
        np.testing.assert_array_equal(cp.bincount(rows).get(), parameter)
        np.testing.assert_array_equal(
            distances.reshape(-1, parameter).get(),
            cp.sort(direct, axis=1)[:, :parameter].get(),
        )
    else:
        expected = cp.flatnonzero(direct.astype(cp.float64) <= parameter)
        np.testing.assert_array_equal(cp.sort(keys).get(), expected.get())
    np.testing.assert_array_equal(distances.get(), direct[rows, cols].get())
    np.testing.assert_array_equal(fused.get(), distances.get())


def test_knn_subnormal_coordinates():
    # CuPy kernels flush subnormals; the tree must still order them.
    x = [[4e-41, 0], [3e-41, 0], [1e-41, 0], [2e-41, 0]]
    _, cols, _ = backend._knn_edges(cp.asarray(x, dtype=cp.float32), 1)
    np.testing.assert_array_equal(cols.get(), [1, 0, 3, 2])


_EDGES = [2, 2, 2, 5, 5, 6], [6, 0, 1, 7, 2, 3], [0, 1, 3, 4, 2, 1]


@pytest.mark.parametrize(
    "edges,n,dtype,set_diag",
    [
        (_EDGES, 9, np.float32, False),
        (_EDGES, 9, np.float64, True),
        (([], [], []), 513, np.float32, True),
        (([0, 100000], [1, 99999], [0, 2]), 200000, np.float64, False),
    ],
    ids=["unsorted", "unsorted-diag", "empty", "gap"],
)
def test_csr_assembly_storage_gaps_and_stream(edges, n, dtype, set_diag):
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        # Strided input conversion and assembly must respect the caller's stream.
        rows, cols, values = (
            cp.repeat(cp.asarray(x, dtype=t), 2)[::2]
            for x, t in zip(edges, (cp.int64, cp.int32, dtype), strict=True)
        )
        adj, dst = build_graphs(rows, cols, values, n, set_diag=set_diag)
        adjacency = build_adjacency(rows, cols, n, set_diag=set_diag)
        ones = cp.ones(len(rows), dtype=cp.float32)
        expected_adj = cp_sparse.csr_matrix((ones, (rows, cols)), shape=(n, n))
        expected_dst = cp_sparse.csr_matrix((values, (rows, cols)), shape=(n, n))
        expected_adj.setdiag(1 if set_diag else 0)
        expected_dst.setdiag(0)
    stream.synchronize()
    pairs = (adj, expected_adj), (dst, expected_dst), (adjacency, expected_adj)
    for actual, expected in pairs:
        actual, expected = actual.get(), expected.get()
        actual.sort_indices()
        expected.sort_indices()
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual.indptr, expected.indptr)
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_allclose(actual.data, expected.data, rtol=2e-6)
    assert adj.nnz == dst.nnz == len(rows) + n  # Stored diagonal zeros survive.
    for name in ("indices", "indptr", "data"):
        assert getattr(adj, name).data.ptr != getattr(dst, name).data.ptr


def _assert_delaunay_edges(edges, points):
    rows, cols = (cp.asarray(edge, dtype=cp.int64) for edge in edges)
    indptr, neighbors = Delaunay(points).vertex_neighbor_vertices
    n = len(points)
    expected = np.repeat(np.arange(n), np.diff(indptr)) * n + neighbors
    np.testing.assert_array_equal(cp.sort(rows * n + cols).get(), np.sort(expected))
    assert bool((cp.diff(rows) >= 0).all()) and bool((rows != cols).all())


@pytest.fixture
def force_gpu(monkeypatch):
    monkeypatch.setattr(delaunay, "_GPU_MIN_POINTS", 3)


@pytest.mark.parametrize("kind", ["float32", "translated", "morton_collision"])
def test_gpu_delaunay_exact_edges(force_gpu, kind):
    points = np.random.default_rng(171).uniform(size=(32768, 2))
    if kind == "float32":
        points = points.astype(np.float32)
    elif kind == "translated":
        points = (points + 4) * 2.0**-100
    else:
        lower, upper = points.min(), points.max()
        offsets = np.array([[5000.2, 11000.2], [5000.3, 11000.3]])
        points[:2] = lower + offsets * ((upper - lower) / 32768)
        gpu = cp.asarray(points)
        keys = cp.empty(len(points), dtype=cp.int32)
        low, span = gpu.min(), gpu.max() - gpu.min()
        delaunay.get_morton_number(gpu, len(points), low, span, keys)
        assert int(keys[0]) == int(keys[1])
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        edges = delaunay._gpu_delaunay_edges(cp.asarray(points))
    stream.synchronize()
    _assert_delaunay_edges(edges, points)
    points[3, 1] = np.nan  # Rejected before triangulation, which can hang.
    with pytest.raises(ValueError, match="finite"):
        delaunay._gpu_delaunay_edges(cp.asarray(points[:64]))


@pytest.mark.parametrize("kind", ["duplicates", "circle", "close_pair"])
def test_ambiguous_delaunay_uses_gpu(force_gpu, monkeypatch, kind):
    points = np.random.default_rng(147).uniform(size=(2048, 2))
    points[1] = points[0] + (1e-10 if kind == "close_pair" else 0)
    if kind == "circle":
        # Integer points are exactly cocircular, so either diagonal is valid.
        z = np.array([5, 4 + 3j, 3 + 4j]) * np.array([1, 1j, -1, -1j])[:, None]
        points = np.c_[z.real.ravel(), z.imag.ravel()]
    monkeypatch.setattr(delaunay, "Delaunay", _UNEXPECTED)
    rows, cols = (x.get() for x in delaunay._delaunay_edges(cp.asarray(points)))
    n = len(points)
    keys = rows * n + cols
    assert np.all(rows != cols) and np.all(np.diff(rows) >= 0)
    np.testing.assert_array_equal(np.sort(keys), np.unique(keys))
    np.testing.assert_array_equal(np.sort(keys), np.sort(cols * n + rows))
    unique, representatives = np.unique(points, axis=0, return_index=True)
    np.testing.assert_array_equal(np.unique(rows), np.sort(representatives))
    hull = representatives[ConvexHull(unique).vertices]
    assert len(rows) == 2 * (3 * len(unique) - len(hull) - 3)
    assert np.isin(hull * n + np.roll(hull, -1), keys).all()
    if kind == "circle":
        # The input follows the convex hull; interleaved chords would cross.
        left, right = rows[rows < cols], cols[rows < cols]
        assert not np.any(
            (left[:, None] < left) & (left < right[:, None]) & (right[:, None] < right)
        )


@pytest.mark.parametrize("case", ["small", "3d", AttributeError, TypeError])
def test_delaunay_falls_back_to_qhull(monkeypatch, recwarn, case):
    # Small and 3D inputs use Qhull, as do changed private CuPy APIs.
    points = np.random.default_rng(177).normal(size=(34, 3 if case == "3d" else 2))
    monkeypatch.setattr(delaunay, "_GPU_MIN_POINTS", 35 if case == "small" else 3)
    error = AssertionError if isinstance(case, str) else case
    monkeypatch.setattr(delaunay, "_FullDelaunay", Mock(side_effect=error))
    _assert_delaunay_edges(delaunay._delaunay_edges(cp.asarray(points)), points)
    if case == "3d":
        message = str(recwarn.pop(UserWarning).message)
        issues = "https://github.com/scverse/rapids_singlecell/issues"
        for part in ("3D", "using CPU triangulation", "feature request", issues):
            assert part in message
    assert not recwarn


_QUAD = [[0, 0], [1, 0], [1, 1.1], [0, 1]]
# CuPy's adaptive incircle predicate gives the wrong sign on this nearly
# cocircular quad.
_COCIRCULAR = [
    [0.3497366723201419, 0.3573293439313845],
    [0.3265864214768884, 0.37860442325324223],
    [0.33804635178765785, -0.368408284438685],
    [0.42119129982159276, -0.26943995426550443],
]


@pytest.mark.parametrize(
    "points,corruption",
    [(_QUAD, None), (_COCIRCULAR, None), (_QUAD, (1, 2, 0)), (_QUAD, (0, 0, -2))],
    ids=["ordinary", "nearly_cocircular", "reciprocal", "boundary"],
)
def test_validation_repairs_or_rejects_triangulations(monkeypatch, points, corruption):
    points = cp.asarray(points, dtype=cp.float64)
    triangles = cp.array([[0, 1, 2], [0, 2, 3]], dtype=cp.int32)
    opposite = cp.array([[-1, 18, -1], [-1, -1, 1]], dtype=cp.int32)
    if corruption:
        opposite[corruption[:2]] = corruption[2]
        monkeypatch.setattr(delaunay, "_flip_delaunay_edges", _UNEXPECTED)
        assert delaunay._validated_edges(points, triangles, opposite) is None
        return
    rows, cols = delaunay._validated_edges(points, triangles, opposite)
    # Preserve the hull and replace the illegal 0--2 diagonal with 1--3.
    left, right = np.array([[0, 1, 2, 3, 1], [1, 2, 3, 0, 3]])
    expected = np.sort(np.r_[left * 4 + right, right * 4 + left])
    np.testing.assert_array_equal(cp.sort(rows * 4 + cols).get(), expected)
