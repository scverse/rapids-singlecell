from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from cupyx.scipy import sparse as cp_sparse
from scipy.spatial import Delaunay

from rapids_singlecell.squidpy_gpu import _spatial_delaunay as delaunay
from rapids_singlecell.squidpy_gpu import _spatial_neighbors_backend as backend
from rapids_singlecell.squidpy_gpu._spatial_graph import build_adjacency, build_graphs


def _dense_distances(coords):
    distances = cp.zeros((len(coords), len(coords)), dtype=coords.dtype)
    for dim in range(coords.shape[1]):
        cp.hypot(distances, coords[:, dim, None] - coords[None, :, dim], out=distances)
    return distances


@pytest.mark.parametrize("method,n_neighs", [("knn", 8), ("knn", 9), ("radius", None)])
@pytest.mark.parametrize(
    "dtype,dimensions,offset,scale",
    [
        (np.float32, 2, 1e6, 1),
        (np.float64, 3, 1e12, 1),
        (np.float32, 2, 0, 1e-30),
        (np.float64, 3, 0, 1e-200),
        (np.float64, 4, 0, 1e200),
    ],
)
def test_tree_matches_dense_search(
    *, method, n_neighs, dtype, dimensions, offset, scale
):
    points = np.random.default_rng(21).normal(size=(31, dimensions)) * scale + offset
    search = getattr(backend, f"_{method}_edges")
    parameter = n_neighs if method == "knn" else 1.25 * scale
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        coords = cp.asarray(points, dtype=dtype)
        coords[-1] = coords[0]
        rows, cols, distances = search(coords, parameter)
    stream.synchronize()
    direct = _dense_distances(coords)
    cp.fill_diagonal(direct, cp.inf)
    assert distances.dtype == coords.dtype
    assert bool((rows != cols).all())
    assert bool((cp.diff(rows) >= 0).all())
    assert len(cp.unique(rows * len(coords) + cols)) == len(rows)
    if method == "knn":
        # Ties may choose different indices, but neighbor distances must match.
        np.testing.assert_array_equal(cp.bincount(rows).get(), n_neighs)
        np.testing.assert_allclose(
            cp.sort(distances.reshape(-1, n_neighs), axis=1).get(),
            cp.sort(direct, axis=1)[:, :n_neighs].get(),
            rtol=1e-6,
            atol=0,
        )
    else:
        expected_rows, expected_cols = cp.nonzero(
            direct.astype(cp.float64) <= parameter
        )
        np.testing.assert_array_equal(
            cp.sort(rows * len(coords) + cols).get(),
            cp.sort(expected_rows * len(coords) + expected_cols).get(),
        )
    np.testing.assert_array_equal(distances.get(), direct[rows, cols].get())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_radius_boundary_precision_and_duplicates(dtype):
    coords = cp.asarray([[0, 0], [0, 0], [1, 0]], dtype=dtype)
    direct = _dense_distances(coords).astype(cp.float64)
    cp.fill_diagonal(direct, cp.inf)
    for radius, count in [(0, 2), (np.nextafter(1.0, 0.0), 2), (1.0, 6)]:
        rows, cols, distances = backend._radius_edges(coords, radius)
        expected_rows, expected_cols = cp.nonzero(direct <= radius)
        assert len(rows) == len(cp.unique(rows * 3 + cols)) == count
        np.testing.assert_array_equal(
            cp.sort(rows * 3 + cols).get(), (expected_rows * 3 + expected_cols).get()
        )
        np.testing.assert_array_equal(distances.get(), direct[rows, cols].get())


def _assert_csr(actual, expected):
    actual, expected = actual.get(), expected.get()
    actual.sort_indices()
    expected.sort_indices()
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_allclose(actual.data, expected.data, rtol=2e-6)


@pytest.mark.parametrize(
    "case,dtype,set_diag",
    [
        ("unsorted", np.float32, False),
        ("unsorted", np.float64, True),
        ("empty", np.float32, True),
        ("gap", np.float64, False),
    ],
)
def test_csr_assembly_storage_gaps_and_stream(case, dtype, set_diag):
    rows, cols, values, n = (
        [2, 2, 2, 5, 5, 6],
        [6, 0, 1, 7, 2, 3],
        [0, 1, 3, 4, 2, 1],
        9,
    )
    if case == "empty":
        rows, cols, values, n = [], [], [], 513
    elif case == "gap":
        rows, cols, values, n = [0, 100000], [1, 99999], [0, 2], 200000
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        # Strided input conversion and assembly must respect the caller's stream.
        rows = cp.repeat(cp.asarray(rows, dtype=cp.int64), 2)[::2]
        cols = cp.repeat(cp.asarray(cols, dtype=cp.int32), 2)[::2]
        values = cp.repeat(cp.asarray(values, dtype=dtype), 2)[::2]
        adj, dst = build_graphs(rows, cols, values, n, set_diag=set_diag)
        adjacency = build_adjacency(rows, cols, n, set_diag=set_diag)
        expected_adj = cp_sparse.csr_matrix(
            (cp.ones(len(rows), dtype=cp.float32), (rows, cols)), shape=(n, n)
        )
        expected_dst = cp_sparse.csr_matrix((values, (rows, cols)), shape=(n, n))
        expected_adj.setdiag(1 if set_diag else 0)
        expected_dst.setdiag(0)
    stream.synchronize()
    _assert_csr(adj, expected_adj)
    _assert_csr(dst, expected_dst)
    _assert_csr(adjacency, expected_adj)
    assert adj.nnz == dst.nnz == len(rows) + n  # Stored diagonal zeros survive.
    for name in ("indices", "indptr", "data"):
        assert getattr(adj, name).data.ptr != getattr(dst, name).data.ptr


def _assert_delaunay_edges(edges, points):
    rows, cols = edges
    indptr, reference_cols = Delaunay(points).vertex_neighbor_vertices
    reference_rows = np.repeat(np.arange(len(points)), np.diff(indptr))
    np.testing.assert_array_equal(
        cp.sort(rows.astype(cp.int64) * len(points) + cols).get(),
        np.sort(reference_rows * len(points) + reference_cols),
    )
    assert bool((cp.diff(rows) >= 0).all())
    assert bool((rows != cols).all())


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
        points[:2] = lower + np.array([[5000.2, 11000.2], [5000.3, 11000.3]]) * (
            (upper - lower) / 32768
        )
        gpu = cp.asarray(points)
        keys = cp.empty(len(points), dtype=cp.int32)
        delaunay.get_morton_number(
            gpu, len(points), gpu.min(), gpu.max() - gpu.min(), keys
        )
        assert int(keys[0]) == int(keys[1])
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        edges = delaunay._gpu_delaunay_edges(cp.asarray(points))
    stream.synchronize()
    assert edges is not None
    _assert_delaunay_edges(edges, points)


@pytest.mark.parametrize("kind", ["duplicates", "circle", "close_pair"])
def test_ambiguous_delaunay_falls_back(force_gpu, monkeypatch, kind):
    points = np.random.default_rng(147).uniform(size=(2048, 2))
    if kind == "circle":
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        points = np.column_stack((np.cos(angles), np.sin(angles)))
    else:
        points[1] = points[0] + (1e-10 if kind == "close_pair" else 0)
    calls = []

    def qhull(coords):
        calls.append(coords)
        return Delaunay(coords)

    monkeypatch.setattr(delaunay, "Delaunay", qhull)
    _assert_delaunay_edges(delaunay._delaunay_edges(cp.asarray(points)), points)
    assert len(calls) == 1


@pytest.mark.parametrize("dimensions", [2, 3])
def test_delaunay_small_2d_and_3d_use_qhull(monkeypatch, dimensions):
    points = np.random.default_rng(177).normal(size=(34, dimensions))
    if dimensions == 3:
        monkeypatch.setattr(delaunay, "_GPU_MIN_POINTS", 0)
    monkeypatch.setattr(
        delaunay, "_FullDelaunay", lambda *args: pytest.fail("Unexpected GPU")
    )
    _assert_delaunay_edges(delaunay._delaunay_edges(cp.asarray(points)), points)


def test_validation_rejects_coherent_non_delaunay_triangulation():
    points = cp.array([[0, 0], [1, 0], [1, 1.1], [0, 1]], dtype=cp.float64)
    triangles = cp.array([[0, 1, 2], [0, 2, 3]], dtype=cp.int32)
    opposite = cp.array([[-1, 18, -1], [-1, -1, 1]], dtype=cp.int32)
    # Reciprocal topology is valid, but this diagonal violates the incircle test.
    assert delaunay._validated_edges(points, triangles, opposite, tolerance=0) is None


@pytest.mark.parametrize("dtype,scale", [(np.float32, 1e30), (np.float64, 1e-200)])
def test_fused_distances_preserve_extreme_scales(dtype, scale):
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        coords = cp.asarray(
            np.random.default_rng(33).normal(size=(20, 3)) * scale, dtype=dtype
        )
        rows = cp.arange(len(coords), dtype=cp.int64)
        cols = ((rows + 3) % len(coords)).astype(cp.int32)
        actual = backend._edge_distances(coords, rows, cols)
        expected = _dense_distances(coords)[rows, cols]
    stream.synchronize()
    assert actual.dtype == dtype
    np.testing.assert_allclose(actual.get(), expected.get(), rtol=2e-7, atol=0)
