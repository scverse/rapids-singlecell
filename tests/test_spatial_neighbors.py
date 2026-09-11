from __future__ import annotations

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

_MODES = [
    ("knn", {"n_neighs": 4}),
    ("radius", {"radius": 1.3}),
    ("delaunay", {}),
    ("grid", {"n_neighs": 4, "n_rings": 2}),
]


def _adata(coords, spatial_key="spatial"):
    adata = AnnData(np.zeros((len(coords), 1), dtype=np.float32))
    adata.obsm[spatial_key] = coords
    return adata


def _call(method, adata, **kwargs):
    return getattr(rsc.gr, f"spatial_neighbors_{method}")(adata, **kwargs)


def _reference(
    coords,
    method,
    *,
    n_neighs=6,
    radius=None,
    n_rings=1,
    delaunay=False,
    percentile=None,
    set_diag=False,
    transform=None,
):
    """Independent SciPy/sklearn topology and Squidpy's processing order."""
    coords = coords.astype(np.float32)
    shape = (len(coords), len(coords))
    if method == "delaunay" or delaunay:
        dense = np.zeros(shape)
        for simplex in Delaunay(coords).simplices:
            dense[np.ix_(simplex, simplex)] = 1
        np.fill_diagonal(dense, 0)
        rows, cols = np.nonzero(dense)
        values = np.linalg.norm(coords[rows] - coords[cols], axis=1)
    else:
        tree = NearestNeighbors().fit(coords)
        if method in {"knn", "grid"}:
            distances, indices = tree.kneighbors(n_neighbors=n_neighs)
        else:
            upper = max(radius) if isinstance(radius, tuple) else radius
            distances, indices = tree.radius_neighbors(radius=upper)
        rows = np.repeat(np.arange(len(coords)), [len(row) for row in indices])
        cols, values = np.concatenate(indices), np.concatenate(distances)
        if method == "grid":
            keep = values < 1.3 * np.median(values)
            rows, cols, values = rows[keep], cols[keep], values[keep]
    adj = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=shape)
    dst = sparse.csr_matrix((values, (rows, cols)), shape=shape)
    if method == "grid":
        hops = shortest_path(adj, directed=True, unweighted=True)
        keep = (hops > 0) & (hops <= n_rings)
        adj = sparse.csr_matrix(keep, dtype=float)
        dst = sparse.csr_matrix(np.where(keep, hops, 0))
    adj.setdiag(float(set_diag))
    dst.setdiag(0)
    if method == "delaunay" and radius is not None and not isinstance(radius, tuple):
        radius = (0, radius)
    if isinstance(radius, tuple):
        lower, upper = sorted(radius)
        outside = (dst.data < lower) | (dst.data > upper)
        adj.data[outside] = dst.data[outside] = 0
        adj.setdiag(float(set_diag))
    if percentile is not None:
        outside = dst.data > np.percentile(dst.data, percentile)
        adj.data[outside] = dst.data[outside] = 0
    adj.eliminate_zeros()
    dst.eliminate_zeros()
    if transform == "spectral":
        degrees = np.asarray(adj.sum(axis=0)).ravel()
        scale = np.zeros_like(degrees)
        np.divide(1, np.sqrt(degrees), out=scale, where=degrees > 0)
        adj = (sparse.diags(scale) @ adj @ sparse.diags(scale)).tocsr()
    elif transform == "cosine":
        adj = cosine_similarity(adj, dense_output=False).tocsr()
    return adj, dst


def _assert_graphs(actual, expected):
    assert isinstance(actual, rsc.gr.SpatialNeighborsResult)
    for observed, reference in zip(actual, expected, strict=True):
        assert sparse.isspmatrix_csr(observed)
        assert np.isfinite(observed.data).all()
        np.testing.assert_allclose(
            observed.toarray(), reference.toarray(), rtol=2e-5, atol=1e-6
        )
    assert not actual.distances.diagonal().any()


def _library_reference(coords, labels, method, **kwargs):
    expected = [np.zeros((len(coords), len(coords))) for _ in range(2)]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        for matrix, block in zip(
            expected, _reference(coords[indices], method, **kwargs), strict=True
        ):
            matrix[np.ix_(indices, indices)] = block.toarray()
    return tuple(sparse.csr_matrix(matrix) for matrix in expected)


@pytest.mark.parametrize("method,kwargs", _MODES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_dims", [2, 3])
def test_cpu_parity(method, kwargs, dtype, n_dims):
    coords = np.random.default_rng(12).normal(size=(24, n_dims)).astype(dtype)
    adata = _adata(coords.copy())
    result = _call(method, adata, **kwargs, copy=True)
    _assert_graphs(result, _reference(coords, method, **kwargs))
    np.testing.assert_array_equal(adata.obsm["spatial"], coords)
    assert adata.obsm["spatial"].dtype == dtype
    assert result.connectivities.dtype == result.distances.dtype == np.float32
    assert not result.connectivities.diagonal().any()
    if method == "knn":
        np.testing.assert_array_equal(result.connectivities.getnnz(axis=1), 4)


@pytest.mark.parametrize("method,kwargs", _MODES)
@pytest.mark.parametrize("transform,set_diag", [("spectral", False), ("cosine", True)])
def test_transforms_and_pruning(method, kwargs, transform, set_diag):
    coords = np.random.default_rng(42).normal(size=(24, 2))
    kwargs = dict(kwargs, transform=transform, set_diag=set_diag)
    if method != "grid":
        kwargs["percentile"] = 70
    if method in {"radius", "delaunay"}:
        kwargs["radius"] = (0.4, 1.5)
    result = _call(method, _adata(coords), **kwargs, copy=True)
    _assert_graphs(result, _reference(coords, method, **kwargs))


@pytest.mark.parametrize("n_neighs,n_rings", [(4, 1), (6, 3)])
def test_grid_lattice_hops(n_neighs, n_rings):
    rows, cols = np.indices((7, 7))
    coords = np.column_stack((cols.ravel(), rows.ravel())).astype(float)
    if n_neighs == 6:
        coords[:, 0] += 0.5 * (rows.ravel() % 2)
        coords[:, 1] *= np.sqrt(3) / 2
    kwargs = {"n_neighs": n_neighs, "n_rings": n_rings, "set_diag": True}
    result = _call("grid", _adata(coords), **kwargs, copy=True)
    _assert_graphs(result, _reference(coords, "grid", **kwargs))
    assert result.distances.max() == n_rings


@pytest.mark.parametrize("radius", [0.0, 1.0, (2.0, 1.0), 1e40])
def test_radius_duplicates_and_inclusive_bounds(radius):
    coords = np.array([[0, 0], [0, 0], [1, 0], [2, 0], [4, 0]], dtype=float)
    result = _call("radius", _adata(coords), radius=radius, copy=True)
    distances = np.abs(coords[:, 0, None] - coords[None, :, 0])
    lower, upper = sorted(radius) if isinstance(radius, tuple) else (0, radius)
    expected = (distances >= lower) & (distances <= upper)
    np.fill_diagonal(expected, False)
    np.testing.assert_array_equal(result.connectivities.toarray(), expected)
    np.testing.assert_allclose(result.distances.toarray(), distances * expected)
    assert result.distances.dtype == np.float32


@pytest.mark.parametrize(
    "coords",
    [
        [[-3e38, 0], [3e38, 0]],
        [[0, 0], [3e38, 3e38]],
    ],
    ids=["subtraction", "norm"],
)
def test_radius_rejects_unrepresentable_distances(coords):
    with pytest.raises(ValueError, match="Rescale"):
        _call("radius", _adata(np.array(coords, dtype=np.float32)), radius=1e39)


@pytest.mark.parametrize("radius", [0, 1, 1e38, 5e38])
def test_radius_excludes_overflowing_distances(radius):
    coords = np.array(
        [[-3e38, 0], [-3e38, 0], [-3e38, 1], [3e38, 0], [3e38, 0], [3e38, 1]],
        dtype=np.float32,
    )
    result = _call("radius", _adata(coords), radius=radius, copy=True)
    delta = coords[:, None].astype(np.float64) - coords[None, :]
    distances = np.linalg.norm(delta, axis=-1)
    expected = distances <= radius
    np.fill_diagonal(expected, False)
    np.testing.assert_array_equal(result.connectivities.toarray(), expected)
    np.testing.assert_array_equal(result.distances.toarray(), distances * expected)


@pytest.mark.parametrize(
    "method,kwargs,count", [("knn", {"n_neighs": 3}, 3), ("radius", {"radius": 0}, 7)]
)
def test_float32_rounding_and_ties_exclude_self(method, kwargs, count):
    coords = 1e12 + np.arange(16).reshape(8, 2)  # Distinct in float64, tied in float32.
    result = _call(method, _adata(coords), **kwargs, copy=True)
    np.testing.assert_array_equal(result.connectivities.getnnz(axis=1), count)
    assert not result.connectivities.diagonal().any()
    assert result.distances.nnz == 0


@pytest.mark.parametrize("radius", [1 - 1e-8, (1 + 1e-8, 2)])
def test_radius_preserves_bound_precision(radius):
    result = _call(
        "radius",
        _adata(np.array([[0, 0], [1, 0]], dtype=np.float32)),
        radius=radius,
        copy=True,
    )
    assert result.connectivities.nnz == result.distances.nnz == 0


@pytest.mark.parametrize("interval", [False, True])
def test_percentile_counts_stored_zeros(interval):
    coords = np.zeros((4, 2))
    coords[:, 0] = [0, 1, 3, 7] if interval else [0, 1, 4, 10]
    method = "radius" if interval else "knn"
    kwargs = {"radius": (2, 6)} if interval else {"n_neighs": 1}
    result = _call(method, _adata(coords), **kwargs, percentile=50, copy=True)
    expected = np.zeros((4, 4))
    if interval:
        expected[1, 2] = expected[2, 1] = 1
    # Diagonal zeros and interval-pruned entries both count toward the median.
    np.testing.assert_array_equal(result.connectivities.toarray(), expected)
    np.testing.assert_array_equal(result.distances.toarray(), 2 * expected)


@pytest.mark.parametrize("method,kwargs", _MODES)
def test_libraries_restore_order_and_process_independently(method, kwargs):
    coords = np.random.default_rng(76).normal(size=(36, 2))
    coords[1::2] *= 10
    labels = np.array(["a", "b"] * 18)
    adata = _adata(coords)
    adata.obs["library"] = pd.Categorical(labels, categories=["b", "unused", "a"])
    kwargs = dict(kwargs, transform="cosine")
    if method != "grid":
        kwargs["percentile"] = 60
    result = _call(method, adata, **kwargs, library_key="library", copy=True)
    _assert_graphs(result, _library_reference(coords, labels, method, **kwargs))
    assert not result.connectivities.toarray()[labels[:, None] != labels[None, :]].any()


@pytest.mark.parametrize("method", ["delaunay", "grid"])
def test_gpu_delaunay_integration(monkeypatch, method):
    from rapids_singlecell.squidpy_gpu import _spatial_delaunay as backend

    monkeypatch.setattr(backend, "_GPU_MIN_POINTS", 3)

    def unexpected_qhull(*args, **kwargs):
        pytest.fail("Nondegenerate input must use GPU triangulation.")

    monkeypatch.setattr(backend, "Delaunay", unexpected_qhull)
    coords = np.random.default_rng(1989).normal(size=(256, 2))
    labels = np.array(["a", "b"] * 128)
    adata = _adata(coords)
    adata.obs["library"] = pd.Categorical(labels, categories=["b", "a"])
    kwargs = (
        {"radius": (0.05, 1.0), "percentile": 75}
        if method == "delaunay"
        else {"delaunay": True, "n_rings": 2}
    )
    kwargs.update(transform="cosine", set_diag=True)
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
    _assert_graphs(
        copied, (view.obsp["custom_connectivities"], view.obsp["custom_distances"])
    )


@pytest.mark.parametrize("kind", ["cupy", "dataframe", "integer", "noncontiguous"])
def test_coordinate_containers(kind):
    coords = np.array([[0, 0], [1, 0], [4, 0], [10, 0]], dtype=float)
    adata = _adata(coords)
    if kind == "cupy":
        adata.obsm["spatial"] = cp.asarray(coords)
    elif kind == "dataframe":
        adata.obsm["spatial"] = pd.DataFrame(coords, index=adata.obs_names)
    elif kind == "integer":
        adata.obsm["spatial"] = coords.astype(np.int32)
    else:
        adata.obsm["spatial"] = np.asfortranarray(coords)
    result = _call("knn", adata, n_neighs=1, copy=True)
    _assert_graphs(result, _reference(coords, "knn", n_neighs=1))


def test_no_legacy_dispatch():
    assert not hasattr(rsc.gr, "spatial_neighbors")


def test_custom_builder_protocol():
    class Builder:
        def build(self, coords):
            assert isinstance(coords, cp.ndarray)
            rows = cp.arange(len(coords))
            adj = cp_sparse.csr_matrix(
                (cp.ones(len(coords)), (rows, (rows + 1) % len(coords))),
                shape=(len(coords), len(coords)),
            )
            return adj, adj * 5

        def uns_params(self):
            return {"builder": "cycle"}

    adata = _adata(np.zeros((4, 2)))
    rsc.gr.spatial_neighbors_from_builder(adata, Builder())
    expected = np.roll(np.eye(4), 1, axis=1)
    np.testing.assert_array_equal(
        adata.obsp["spatial_connectivities"].toarray(), expected
    )
    np.testing.assert_array_equal(
        adata.obsp["spatial_distances"].toarray(), expected * 5
    )
    assert adata.uns["spatial_neighbors"]["params"] == {"builder": "cycle"}


def test_builtin_subclass_build_override():
    calls = []

    class CustomKNN(rsc.gr.neighbors.KNNBuilder):
        def build(self, coords):
            calls.append(len(coords))
            adj, dst = super().build(coords)
            return adj * 2, dst

    adata = _adata(np.random.default_rng(31).normal(size=(24, 2)))
    adata.obs["library"] = pd.Categorical(["a", "b"] * 12)
    expected = _call("knn", adata, library_key="library", copy=True)
    result = rsc.gr.spatial_neighbors_from_builder(
        adata, CustomKNN(), library_key="library", copy=True
    )
    assert calls == [12, 12]
    _assert_graphs(result, (expected.connectivities * 2, expected.distances))


@pytest.mark.parametrize(
    "method,kwargs,scale",
    [("knn", {"n_neighs": 0}, 0), ("radius", {"radius": np.nan}, 0)]
    + [(method, kwargs, 1e200) for method, kwargs in _MODES],
)
def test_invalid_parameters_do_not_write(method, kwargs, scale):
    adata = _adata(np.full((4, 2), scale))
    with pytest.raises(ValueError):
        _call(method, adata, **kwargs)
    assert not adata.obsp and not adata.uns
