from __future__ import annotations

from unittest.mock import Mock

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from cupyx.scipy import sparse as cp_sparse
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import Delaunay
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

import rapids_singlecell as rsc
from rapids_singlecell.squidpy_gpu import _spatial_delaunay

_MODES = [
    ("knn", {"n_neighs": 4}),
    ("radius", {"radius": 1.3}),
    ("delaunay", {}),
    ("grid", {"n_neighs": 4, "n_rings": 2}),
]
_TIED = 1e12 + np.arange(16).reshape(8, 2)  # Distinct in float64, tied in float32.


def _adata(coords, spatial_key="spatial"):
    adata = AnnData(np.zeros((len(coords), 1), dtype=np.float32))
    adata.obsm[spatial_key] = coords
    return adata


def _call(method, adata, **kwargs):
    return getattr(rsc.gr, f"spatial_neighbors_{method}")(adata, **kwargs)


def _reference(coords, method, *, radius=None, percentile=None, set_diag=False, **kw):
    """Dense SciPy/sklearn topology, processed in Squidpy's order."""
    coords = coords.astype(np.float32)
    lengths = np.linalg.norm(coords[:, None] - coords, axis=-1)
    tree = NearestNeighbors().fit(coords)
    if method == "delaunay" or kw.get("delaunay"):
        indptr, cols = Delaunay(coords).vertex_neighbor_vertices
        edges = sparse.csr_matrix((np.ones(len(cols)), cols, indptr), lengths.shape)
    elif method == "radius":
        edges = tree.radius_neighbors_graph(radius=np.max(radius))
    else:
        edges = tree.kneighbors_graph(n_neighbors=kw.get("n_neighs", 6))
    edges = edges.toarray() > 0
    if method == "grid":
        if not kw.get("delaunay"):
            edges &= lengths < 1.3 * np.median(lengths[edges])
        lengths = shortest_path(edges, unweighted=True)
        edges = (lengths > 0) & (lengths <= kw.get("n_rings", 1))
    eye = np.eye(len(coords), dtype=bool)
    stored = edges | eye  # Distances store the diagonal and pruned edges as zeros.
    adj, dst = np.where(eye, set_diag, edges).astype(float), np.where(edges, lengths, 0)
    if radius is not None and (method == "delaunay" or isinstance(radius, tuple)):
        lower, upper = np.sort(np.r_[0, radius])[-2:]  # A scalar means (0, radius).
        outside = edges & ((dst < lower) | (dst > upper))
        adj[outside] = dst[outside] = 0
    if percentile is not None:
        outside = dst > np.percentile(dst[stored], percentile)
        adj[outside] = dst[outside] = 0
    if kw.get("transform") == "spectral":
        degrees = adj.sum(axis=0)
        scale = np.where(degrees > 0, np.maximum(degrees, 1) ** -0.5, 0)
        adj = scale[:, None] * adj * scale
    elif kw.get("transform") == "cosine":
        adj = cosine_similarity(adj)
    return adj, dst


def _library_reference(coords, labels, method, **kwargs):
    expected = np.zeros((2, len(coords), len(coords)))
    for label in np.unique(labels):
        i = np.flatnonzero(labels == label)
        expected[np.ix_([0, 1], i, i)] = _reference(coords[i], method, **kwargs)
    return expected


def _assert_graphs(result, expected, rtol=2e-5, atol=1e-6):
    assert isinstance(result, rsc.gr.SpatialNeighborsResult)
    for observed, reference in zip(result, expected, strict=True):
        assert sparse.isspmatrix_csr(observed) and observed.dtype == np.float32
        assert observed.has_canonical_format and observed.data.all()  # No zeros.
        reference = reference.toarray() if sparse.issparse(reference) else reference
        np.testing.assert_allclose(observed.toarray(), reference, rtol, atol)


@pytest.mark.parametrize("method,kwargs", _MODES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_dims", [2, 3])
@pytest.mark.parametrize("transform", [None, "spectral", "cosine"])
def test_cpu_parity(method, kwargs, dtype, n_dims, transform):
    coords = np.random.default_rng(12).normal(size=(24, n_dims)).astype(dtype)
    if transform:
        kwargs = dict(kwargs, transform=transform, set_diag=transform == "cosine")
        kwargs |= {} if method == "grid" else {"percentile": 70}
        kwargs |= {"radius": (0.4, 1.5)} if method in {"radius", "delaunay"} else {}
    adata = _adata(coords.copy())
    result = _call(method, adata, **kwargs, copy=True)
    _assert_graphs(result, _reference(coords, method, **kwargs))
    np.testing.assert_array_equal(adata.obsm["spatial"], coords, strict=True)


@pytest.mark.parametrize("n_neighs,n_rings", [(4, 1), (6, 3)])
def test_grid_lattice_hops(n_neighs, n_rings):
    y, x = np.indices((7, 7)).reshape(2, -1).astype(float)
    if n_neighs == 6:  # Hexagonal
        x, y = x + 0.5 * (y % 2), y * (np.sqrt(3) / 2)
    coords = np.c_[x, y]
    kwargs = {"n_neighs": n_neighs, "n_rings": n_rings, "set_diag": True}
    result = _call("grid", _adata(coords), **kwargs, copy=True)
    _assert_graphs(result, _reference(coords, "grid", **kwargs))
    assert result.distances.max() == n_rings


_LINE = [[0, 0], [0, 0], [1, 0], [2, 0], [4, 0]]
_OVERFLOW = [[-3e38, 0], [-3e38, 0], [-3e38, 1], [3e38, 0], [3e38, 0], [3e38, 1]]


@pytest.mark.parametrize(
    "coords,radius",
    [
        *[(_LINE, radius) for radius in (0, 1, (2, 1), 1e40)],
        *[(_OVERFLOW, radius) for radius in (0, 1, 1e38, 5e38)],
        ([[0, 0], [1, 0]], 1 - 1e-8),
        ([[0, 0], [1, 0]], (1 + 1e-8, 2)),
        (_TIED, 0),
    ],
)
def test_radius_matches_float64_bounds(coords, radius):
    # Inclusive double bounds; duplicates connect, overflowing lengths do not.
    coords = np.asarray(coords, dtype=float)
    result = _call("radius", _adata(coords), radius=radius, copy=True)
    points = coords.astype(np.float32).astype(np.float64)
    distances = np.linalg.norm(points[:, None] - points, axis=-1)
    lower, upper = sorted(radius) if isinstance(radius, tuple) else (0, radius)
    expected = (distances >= lower) & (distances <= upper)
    np.fill_diagonal(expected, False)
    _assert_graphs(result, (expected, distances * expected), rtol=0, atol=0)


def test_knn_float32_ties_prefer_lower_indices():
    result = _call("knn", _adata(_TIED), n_neighs=3, copy=True)
    others = 1 - np.eye(8)
    expected = others * (np.cumsum(others, axis=1) <= 3)  # The lowest three.
    _assert_graphs(result, (expected, np.zeros((8, 8))))


@pytest.mark.parametrize("interval", [False, True])
def test_percentile_counts_stored_zeros(interval):
    # Diagonal zeros and interval-pruned entries both count toward the median.
    coords = np.c_[[0, 1, 3, 7] if interval else [0, 1, 4, 10], np.zeros(4)]
    method = "radius" if interval else "knn"
    kwargs = {"radius": (2, 6)} if interval else {"n_neighs": 1}
    result = _call(method, _adata(coords), **kwargs, percentile=50, copy=True)
    expected = np.zeros((4, 4))
    expected[[1, 2], [2, 1]] = interval
    _assert_graphs(result, (expected, 2 * expected), rtol=0, atol=0)


@pytest.mark.parametrize("method,kwargs", _MODES)
@pytest.mark.parametrize("transform,set_diag", [("cosine", False), ("spectral", True)])
def test_library_batches_match_separate_builds(method, kwargs, transform, set_diag):
    rng = np.random.default_rng(41)
    labels = rng.permutation(np.repeat(np.arange(4), [5, 40, 300, 1200]))
    coords = rng.normal(size=(len(labels), 2))
    coords[labels == 1] *= 10
    adata = _adata(coords)
    # Unused and reordered categories; interleaved libraries restore their order.
    adata.obs["library"] = pd.Categorical(labels, categories=[3, 1, 9, 0, 2])
    kwargs = dict(kwargs, transform=transform, set_diag=set_diag)
    kwargs |= {"radius": (0.2, 1.3)} if method == "radius" else {}
    kwargs |= {} if method == "grid" else {"percentile": 70}
    library = {"library_key": "library", "copy": True}
    batched = _call(method, adata, **kwargs, **library)
    # Subclasses keep the per-library loop.
    name = method.title().replace("Knn", "KNN")
    loop = type("Loop", (getattr(rsc.gr.neighbors, f"{name}Builder"),), {})
    separate = rsc.gr.spatial_neighbors_from_builder(adata, loop(**kwargs), **library)
    _assert_graphs(batched, _library_reference(coords, labels, method, **kwargs))
    # cuSPARSE sums cosine products in a size-dependent order.
    tolerances = (1e-6 if transform == "cosine" else 0, 0)
    for actual, expected, rtol in zip(batched, separate, tolerances, strict=True):
        expected.sort_indices()
        np.testing.assert_array_equal(actual.indptr, expected.indptr)
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_allclose(actual.data, expected.data, rtol=rtol)


def test_cosine_products_beyond_cusparse_limits():
    # cuSPARSE's int32 SpGEMM rejects nonempty rows 2**15 apart.
    n = 2**15 + 1
    ends, ones = cp.asarray([0, n - 1]), cp.ones(2, dtype=cp.float32)
    adj = cp_sparse.csr_matrix((ones, (ends, ends)), shape=(n, n))
    adj, _ = rsc.gr.neighbors.TransformPostprocessor("cosine")(adj, adj.copy())
    np.testing.assert_array_equal(adj.indices.get(), [0, n - 1])
    np.testing.assert_array_equal(adj.data.get(), [1, 1])
    # Dense libraries together exceed cuSPARSE's SpGEMM work limit.
    adata = _adata(np.zeros((3000, 2)))
    adata.obs["library"] = pd.Categorical(np.repeat([0, 1, 2], 1000))
    kwargs = {"transform": "cosine", "library_key": "library", "copy": True}
    assert _call("radius", adata, radius=1, **kwargs).connectivities.nnz == 3 * 1000**2


@pytest.mark.parametrize("method", ["delaunay", "grid"])
def test_gpu_delaunay_integration(monkeypatch, method):
    # Nondegenerate input must use GPU triangulation.
    monkeypatch.setattr(_spatial_delaunay, "_GPU_MIN_POINTS", 3)
    monkeypatch.setattr(_spatial_delaunay, "Delaunay", Mock(side_effect=AssertionError))
    coords = np.random.default_rng(1989).normal(size=(256, 2))
    labels = np.array(["a", "b"] * 128)
    adata = _adata(coords)
    adata.obs["library"] = pd.Categorical(labels, categories=["b", "a"])
    grid = {"delaunay": True, "n_rings": 2}
    kwargs = grid if method == "grid" else {"radius": (0.05, 1.0), "percentile": 75}
    kwargs = dict(kwargs, transform="cosine", set_diag=True)
    result = _call(method, adata, library_key="library", copy=True, **kwargs)
    _assert_graphs(result, _library_reference(coords, labels, method, **kwargs))


def test_copy_view_and_metadata():
    parent = _adata(np.random.default_rng(19).normal(size=(24, 2)), "coordinates")
    parent.uns["preserved"] = 3
    view = parent[::2]
    kwargs = {"n_neighs": 3, "spatial_key": "coordinates", "key_added": "custom"}
    copied = _call("knn", view, **kwargs, copy=True)
    assert view.is_view and not view.obsp
    assert _call("knn", view, **kwargs) is None
    assert not view.is_view and not parent.obsp
    assert parent.uns == {"preserved": 3}
    assert view.uns["custom_neighbors"] == {
        "connectivities_key": "custom_connectivities",
        "distances_key": "custom_distances",
        "params": {"coord_type": "generic", "transform": None, "n_neighbors": 3},
    }
    _assert_graphs(copied, [view.obsp[f"custom_{key}"] for key in copied._fields])


@pytest.mark.parametrize("kind", ["cupy", "dataframe", "integer", "noncontiguous"])
def test_coordinate_containers(kind):
    coords = np.array([[0, 0], [1, 0], [4, 0], [10, 0]], dtype=float)
    container = {
        "cupy": cp.asarray(coords),
        "dataframe": pd.DataFrame(coords, index=list("0123")),
        "integer": coords.astype(np.int32),
        "noncontiguous": np.asfortranarray(coords),
    }[kind]
    result = _call("knn", _adata(container), n_neighs=1, copy=True)
    _assert_graphs(result, _reference(coords, "knn", n_neighs=1))


def test_public_namespace():
    from rapids_singlecell.gr.neighbors import KNNBuilder

    assert KNNBuilder is rsc.gr.neighbors.KNNBuilder
    assert not hasattr(rsc.gr, "spatial_neighbors")  # No legacy dispatch.


def test_custom_builder_protocol():
    class Cycle:
        def build(self, coords):
            assert isinstance(coords, cp.ndarray)
            adj = cp_sparse.csr_matrix(cp.roll(cp.eye(len(coords)), 1, axis=1))
            return adj, adj * 5

        def uns_params(self):
            return {"builder": "cycle"}

    adata = _adata(np.zeros((4, 2)))
    rsc.gr.spatial_neighbors_from_builder(adata, Cycle())
    expected = np.roll(np.eye(4), 1, axis=1)
    obsp = adata.obsp
    np.testing.assert_array_equal(obsp["spatial_connectivities"].toarray(), expected)
    np.testing.assert_array_equal(obsp["spatial_distances"].toarray(), 5 * expected)
    assert adata.uns["spatial_neighbors"]["params"] == {"builder": "cycle"}


def test_builtin_subclass_overrides():
    calls = []

    class CustomKNN(rsc.gr.neighbors.KNNBuilder):
        def build(self, coords):
            calls.append(len(coords))
            adj, dst = super().build(coords)
            return adj * 2, dst

        def build_graph(self, coords):
            # Overrides may return matrices that share CSR arrays.
            _, dst = super().build_graph(coords)
            weights = cp.exp(-dst.data)
            return cp_sparse.csr_matrix((weights, dst.indices, dst.indptr)), dst

    adata = _adata(np.random.default_rng(31).normal(size=(24, 2)))
    adata.obs["library"] = pd.Categorical(["a", "b"] * 12)
    library = {"library_key": "library", "copy": True}
    distances = _call("knn", adata, **library).distances
    result = rsc.gr.spatial_neighbors_from_builder(adata, CustomKNN(), **library)
    assert calls == [12, 12]
    weights = distances.copy()
    weights.data = np.exp(-weights.data)  # Stored diagonal zeros weigh one.
    _assert_graphs(result, (2 * (weights + sparse.eye(24)), distances))


_FILLER = np.random.default_rng(4).uniform(size=(64, 2))


@pytest.mark.parametrize(
    "method,kwargs,coords",
    [
        ("knn", {"n_neighs": 0}, np.zeros((4, 2))),
        ("radius", {"radius": np.nan}, np.zeros((4, 2))),
        *[(method, kwargs, np.full((4, 2), 1e200)) for method, kwargs in _MODES],
        # Unrepresentable lengths from subtraction and from the norm; filler
        # points put tree levels between the root and the overflowing pair.
        ("radius", {"radius": 1e39}, np.r_[[[-3e38, 0], [3e38, 0]], _FILLER]),
        ("radius", {"radius": 1e39}, np.r_[[[0, 0], [3e38, 3e38]], _FILLER]),
    ],
)
def test_invalid_input_does_not_write(method, kwargs, coords):
    adata = _adata(coords)
    with pytest.raises(ValueError):
        _call(method, adata, **kwargs)
    assert not adata.obsp and not adata.uns
