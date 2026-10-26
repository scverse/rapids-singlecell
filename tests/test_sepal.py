from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from cupyx.scipy import sparse as sparse_gpu
from scipy import sparse
from scipy.spatial.distance import cdist

import rapids_singlecell as rsc
from rapids_singlecell._cuda import _sepal_cuda as _sep_cuda
from rapids_singlecell.squidpy_gpu import _sepal

PLANS = [("smem2", 1), ("smem1", 1), ("global", 1), ("smem2", 2), ("smem1", 4)]


def _grid(kind: str, nx: int, ny: int, *, holes: float, n_genes: int, seed: int):
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    if kind == "hex":
        xs = 2.0 * xs + (ys % 2)
        ys = ys * np.sqrt(3)
    coords = np.c_[xs.ravel(), ys.ravel()]
    coords = coords[rng.random(len(coords)) >= holes]
    n = len(coords)
    X = rng.poisson(1.0, (n, n_genes)).astype(np.float32)
    for j in range(0, n_genes, 2):  # spatial patterns in half of the genes
        cx, cy = rng.uniform(coords.min(0), coords.max(0))
        r2 = (coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2
        X[:, j] += rng.poisson(5 * np.exp(-r2 / (2 * (4 + j) ** 2)))
    adata = AnnData(sparse.csr_matrix(X), obsm={"spatial": coords})
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    rsc.gr.spatial_neighbors_grid(adata, n_neighs=6 if kind == "hex" else 4)
    return adata


def _reference(adata: AnnData, max_neighs: int, *, n_iter=30000, dt=0.001):
    """NumPy port of squidpy.gr.sepal (BSD-3-Clause, scverse)."""
    g = sparse.csr_matrix(adata.obsp["spatial_connectivities"])
    g.eliminate_zeros()
    deg = np.diff(g.indptr)
    sat = np.flatnonzero(deg == max_neighs)
    unsat = np.flatnonzero(deg < max_neighs)
    sat_idx = np.stack([g.indices[g.indptr[i] : g.indptr[i + 1]] for i in sat])
    is_sat = deg == max_neighs
    nearest = np.full(len(unsat), -1)
    for k, i in enumerate(unsat):
        nb = g.indices[g.indptr[i] : g.indptr[i + 1]]
        hit = nb[is_sat[nb]]
        if len(hit):
            nearest[k] = hit[0]
    far = nearest < 0
    if far.any():
        spatial = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        dist = cdist(spatial[unsat[far]], spatial[sat], metric="cityblock")
        nearest[far] = sat[np.argmin(dist, axis=1)]

    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    scores = []
    for j in range(X.shape[1]):
        conc = X[:, j].astype(np.float64)
        prev, it_conv = 1.0, np.nan
        for it in range(n_iter):
            nhood = conc[sat_idx].sum(axis=1)
            if max_neighs == 4:
                d2 = nhood - 4 * conc[sat]
            else:  # squidpy's numba fastmath multiplies by the reciprocal
                d2 = (2.0 * nhood - 12.0 * conc[sat]) * (1.0 / 3.0)
            dcdt = np.zeros_like(conc)
            dcdt[sat] = d2
            conc[sat] += dcdt[sat] * dt
            conc[unsat] += dcdt[nearest] * dt
            conc[conc < 0] = 0
            x = conc[sat]
            x = x[x > 0]
            xs = x.sum()
            if xs < np.finfo(np.float64).eps:
                ent = 0.0
            else:
                p = x / xs
                ent = float(
                    (-np.log(np.maximum(p, np.finfo(np.float64).eps)) * p).sum()
                )
            ent /= len(sat)
            if abs(ent - prev) <= 1e-8:
                it_conv = it
                break
            prev = ent
        scores.append(dt * it_conv)
    return pd.Series(scores, index=adata.var_names)


@pytest.fixture(params=PLANS, ids=lambda p: f"{p[0]}-{p[1]}")
def plan(request, monkeypatch):
    """Pin where the state lives and the blocks per gene."""
    name, cluster_size = request.param
    store = {"smem2": _sepal._SMEM2, "smem1": _sepal._SMEM1, "global": _sepal._GLOBAL}[
        name
    ]

    def pinned(n_cells, max_neighs, dtype):
        n = _sep_cuda.occupancy(
            f32=dtype == np.float32,
            max_neighs=max_neighs,
            store=store,
            cluster_size=cluster_size,
            n_cells=n_cells,
            block_size=_sepal._BLOCK,
        )
        if n == 0:
            pytest.skip("configuration not supported on this GPU")
        return store, cluster_size, n

    monkeypatch.setattr(_sepal, "_plan", pinned)


@pytest.mark.parametrize(
    ("kind", "max_neighs", "holes"),
    [("hex", 6, 0.0), ("square", 4, 0.0), ("hex", 6, 0.08), ("square", 4, 0.15)],
)
def test_sepal_float64_matches_squidpy(plan, kind, max_neighs, holes):
    adata = _grid(kind, 24, 18, holes=holes, n_genes=7, seed=0)
    ref = _reference(adata, max_neighs)
    res = rsc.gr.sepal(adata, max_neighs, copy=True, dtype=np.float64)
    np.testing.assert_array_equal(res["sepal_score"].reindex(ref.index), ref)


@pytest.mark.parametrize(("kind", "max_neighs"), [("hex", 6), ("square", 4)])
def test_sepal_float32_matches_squidpy(plan, kind, max_neighs):
    adata = _grid(kind, 24, 18, holes=0.05, n_genes=8, seed=1)
    ref = _reference(adata, max_neighs)
    res = rsc.gr.sepal(adata, max_neighs, copy=True)["sepal_score"]
    np.testing.assert_array_equal(res.reindex(ref.index), ref)


def test_sepal_far_unsaturated_cells():
    # isolated cells without saturated neighbours use the L1-nearest one
    adata = _grid("square", 20, 20, holes=0.35, n_genes=4, seed=3)
    ref = _reference(adata, 4)
    res = rsc.gr.sepal(adata, 4, copy=True, dtype=np.float64)
    np.testing.assert_array_equal(res["sepal_score"].reindex(ref.index), ref)


@pytest.mark.parametrize(
    "to_input",
    [
        lambda X: X,
        lambda X: X.toarray(),
        lambda X: cp.asarray(X.toarray()),
        lambda X: sparse_gpu.csr_matrix(X),
        lambda X: sparse.csc_matrix(X),
    ],
    ids=["scipy_csr", "numpy", "cupy", "cupy_csr", "scipy_csc"],
)
def test_sepal_input_types(to_input):
    adata = _grid("square", 14, 12, holes=0.0, n_genes=4, seed=5)
    ref = rsc.gr.sepal(adata, 4, copy=True)
    adata.X = to_input(adata.X)
    pd.testing.assert_frame_equal(rsc.gr.sepal(adata, 4, copy=True), ref)


def test_sepal_layer_raw_and_genes():
    adata = _grid("hex", 14, 12, holes=0.0, n_genes=6, seed=6)
    ref = rsc.gr.sepal(adata, 6, copy=True)
    adata.layers["counts"] = adata.X.copy()
    adata.raw = adata.copy()
    adata.X = sparse.csr_matrix(adata.X.shape, dtype=np.float32)
    pd.testing.assert_frame_equal(
        rsc.gr.sepal(adata, 6, layer="counts", copy=True), ref
    )
    pd.testing.assert_frame_equal(rsc.gr.sepal(adata, 6, use_raw=True, copy=True), ref)
    one = rsc.gr.sepal(adata, 6, genes="g2", layer="counts", copy=True)
    assert list(one.index) == ["g2"]
    assert one.iloc[0, 0] == ref.loc["g2", "sepal_score"]


@pytest.mark.parametrize(
    "to_graph",
    [lambda g: g.astype(np.int64), lambda g: sparse_gpu.csr_matrix(g)],
    ids=["int64", "cupy"],
)
def test_sepal_graph_types(to_graph):
    adata = _grid("hex", 12, 10, holes=0.05, n_genes=3, seed=11)
    ref = rsc.gr.sepal(adata, 6, copy=True)
    key = "spatial_connectivities"
    adata.obsp[key] = to_graph(adata.obsp[key])
    pd.testing.assert_frame_equal(rsc.gr.sepal(adata, 6, copy=True), ref)


def test_sepal_cupy_coordinates():
    # far unsaturated cells use the coordinates
    adata = _grid("square", 20, 20, holes=0.35, n_genes=3, seed=12)
    ref = rsc.gr.sepal(adata, 4, copy=True)
    adata.obsm["spatial"] = cp.asarray(adata.obsm["spatial"])
    pd.testing.assert_frame_equal(rsc.gr.sepal(adata, 4, copy=True), ref)


def test_sepal_highly_variable_and_uns():
    adata = _grid("square", 14, 12, holes=0.0, n_genes=6, seed=7)
    adata.var["highly_variable"] = [True, False, True, False, True, False]
    assert rsc.gr.sepal(adata, 4) is None
    res = adata.uns["sepal_score"]
    assert set(res.index) == {"g0", "g2", "g4"}
    assert res["sepal_score"].is_monotonic_decreasing


def test_sepal_not_converged_is_nan():
    adata = _grid("square", 14, 12, holes=0.0, n_genes=3, seed=8)
    res = rsc.gr.sepal(adata, 4, n_iter=2, copy=True)
    assert res["sepal_score"].isna().all()


def test_sepal_errors():
    adata = _grid("square", 10, 10, holes=0.0, n_genes=2, seed=9)
    with pytest.raises(ValueError, match="max_neighs"):
        rsc.gr.sepal(adata, 5)
    with pytest.raises(ValueError, match="max_neighs=6"):
        rsc.gr.sepal(adata, 6)
    with pytest.raises(ValueError, match="dtype"):
        rsc.gr.sepal(adata, 4, dtype=np.float16)
    with pytest.raises(KeyError, match="not_there"):
        rsc.gr.sepal(adata, 4, connectivity_key="not_there")
    with pytest.raises(KeyError, match="Layer"):
        rsc.gr.sepal(adata, 4, layer="nope")
    with pytest.raises(ValueError, match="No genes"):
        rsc.gr.sepal(adata, 4, genes=[])


def test_sepal_spatialdata():
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import TableModel

    adata = _grid("square", 12, 10, holes=0.0, n_genes=3, seed=10)
    ref = rsc.gr.sepal(adata, 4, copy=True)
    sdata = sd.SpatialData(tables={"cells": TableModel.parse(adata.copy())})
    pd.testing.assert_frame_equal(
        rsc.gr.sepal(sdata, 4, table_key="cells", copy=True), ref
    )
    assert rsc.gr.sepal(sdata, 4, table_key="cells") is None
    pd.testing.assert_frame_equal(sdata.tables["cells"].uns["sepal_score"], ref)
    with pytest.raises(TypeError, match="table_key"):
        rsc.gr.sepal(sdata, 4)
