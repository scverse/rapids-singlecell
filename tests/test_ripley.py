from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

import rapids_singlecell as rsc
from rapids_singlecell.squidpy_gpu._ripley import _ppp, _query
from rapids_singlecell.squidpy_gpu._spatial_neighbors_backend import _build_kdtree

CLUSTER_KEY = "leiden"
MODES = ["F", "G", "L"]


@pytest.fixture
def adata() -> AnnData:
    adata = read_h5ad(Path(__file__).parent / "_data/dummy.h5ad")
    adata.obs[CLUSTER_KEY] = adata.obs.cluster.astype("category")
    return adata


@pytest.fixture
def clustered() -> AnnData:
    rng = np.random.default_rng(0)
    n, k = 3000, 4
    centers = rng.uniform(0, 1000, (k, 2))
    labels = rng.integers(0, k, n)
    coords = np.where(
        rng.random((n, 1)) < 0.5,
        centers[labels] + rng.normal(0, 40, (n, 2)),
        rng.uniform(0, 1000, (n, 2)),
    )
    adata = AnnData(
        obs=pd.DataFrame(
            {CLUSTER_KEY: pd.Categorical([f"c{i}" for i in labels])},
            index=np.arange(n).astype(str),
        )
    )
    adata.obsm["spatial"] = coords
    return adata


def _support(coords: np.ndarray, n_steps: int = 50) -> np.ndarray:
    return np.linspace(0, (ConvexHull(coords).volume / 2) ** 0.5, n_steps)


def _f_g_reference(distances: np.ndarray, support: np.ndarray) -> np.ndarray:
    # squidpy's `_f_g_function`
    counts, _ = np.histogram(distances, bins=support)
    return np.concatenate(([0.0], np.cumsum(counts) / counts.sum()))


@pytest.mark.parametrize("mode", MODES)
def test_ripley_modes(adata: AnnData, mode: str):
    """Adapted from squidpy: keys and shapes of the stored result."""
    rsc.gr.ripley(adata, cluster_key=CLUSTER_KEY, mode=mode)

    uns_key = f"{CLUSTER_KEY}_ripley_{mode}"
    res = adata.uns[uns_key]
    assert set(res) == {f"{mode}_stat", "sims_stat", "bins", "pvalues"}
    obs_df = res[f"{mode}_stat"]
    np.testing.assert_array_equal(
        adata.obs[CLUSTER_KEY].cat.categories,
        obs_df[CLUSTER_KEY].cat.categories,
    )
    assert obs_df.shape[1] == res["sims_stat"].shape[1]
    assert res["pvalues"].shape[0] == adata.obs[CLUSTER_KEY].cat.categories.shape[0]
    assert res["bins"].shape[0] == 50


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n_simulations", [20, 50])
@pytest.mark.parametrize("n_observations", [10, 100])
@pytest.mark.parametrize("max_dist", [None, 1000])
@pytest.mark.parametrize("n_steps", [2, 50, 100])
def test_ripley_results(
    adata: AnnData,
    *,
    mode: str,
    n_simulations: int,
    n_observations: int,
    max_dist: float | None,
    n_steps: int,
):
    """Adapted from squidpy: result shapes and zero first bins."""
    n_clusters = adata.obs[CLUSTER_KEY].cat.categories.shape[0]
    res = rsc.gr.ripley(
        adata,
        cluster_key=CLUSTER_KEY,
        mode=mode,
        n_simulations=n_simulations,
        n_observations=n_observations,
        max_dist=max_dist,
        n_steps=n_steps,
        copy=True,
    )
    obs_df = res[f"{mode}_stat"]
    sims_df = res["sims_stat"]

    assert obs_df.shape == (n_steps * n_clusters, 3)
    assert res["bins"].shape == (n_steps,)
    assert sims_df.shape == (n_steps * n_simulations, 3)
    assert res["pvalues"].shape == (n_clusters, n_steps)
    assert sims_df.bins.values[0] == obs_df.bins.values[0] == 0.0
    assert sims_df.stats.values[0] == obs_df.stats.values[0] == 0.0
    assert np.count_nonzero(obs_df.bins.values) == n_steps * n_clusters - n_clusters
    assert ((res["pvalues"] >= 0) & (res["pvalues"] <= 0.5)).all()


@pytest.mark.parametrize("mode", MODES)
def test_ripley_rng(adata: AnnData, mode: str):
    """Adapted from squidpy: seeds reproduce, differ, and vary per simulation."""
    kw = {"cluster_key": CLUSTER_KEY, "mode": mode, "n_simulations": 20, "copy": True}

    def sims(rng):
        res = rsc.gr.ripley(adata, rng=rng, **kw)
        return res["sims_stat"].pivot(
            index="bins", columns="simulations", values="stats"
        ).to_numpy(), res[f"{mode}_stat"]

    sims1, obs1 = sims(np.random.default_rng(42))
    sims2, obs2 = sims(42)
    sims3, _ = sims(43)
    np.testing.assert_array_equal(sims1, sims2)
    pd.testing.assert_frame_equal(obs1, obs2)
    assert not np.array_equal(sims1, sims3)
    assert not np.allclose(sims1, sims1[:, [0]])


def test_ripley_g_matches_reference(clustered: AnnData):
    coords = clustered.obsm["spatial"]
    labels = clustered.obs[CLUSTER_KEY].to_numpy()
    support = _support(coords)
    res = rsc.gr.ripley(clustered, CLUSTER_KEY, mode="G", n_neigh=3, copy=True)
    observed = res["G_stat"].pivot(index="bins", columns=CLUSTER_KEY, values="stats")
    for cluster in observed.columns:
        tree = NearestNeighbors(n_neighbors=3).fit(coords[labels == cluster])
        distances, _ = tree.kneighbors(coords[labels != cluster])
        np.testing.assert_allclose(
            observed[cluster].to_numpy(), _f_g_reference(distances, support)
        )


def test_ripley_l_matches_reference(clustered: AnnData):
    coords = clustered.obsm["spatial"]
    labels = clustered.obs[CLUSTER_KEY].to_numpy()
    support = _support(coords)
    area = ConvexHull(coords).volume
    n = len(coords)
    res = rsc.gr.ripley(clustered, CLUSTER_KEY, mode="L", copy=True)
    observed = res["L_stat"].pivot(index="bins", columns=CLUSTER_KEY, values="stats")
    for cluster in observed.columns:
        distances = pdist(coords[labels == cluster])
        pairs = 2 * (distances <= support[:, None]).sum(axis=1)
        expected = np.sqrt(pairs / n / (n / area) / np.pi)
        # Pairs are counted in float32.
        np.testing.assert_allclose(observed[cluster].to_numpy(), expected, rtol=1e-4)


def test_ripley_f_matches_reference(clustered: AnnData):
    """Distances from external queries match sklearn."""
    coords = clustered.obsm["spatial"]
    codes = clustered.obs[CLUSTER_KEY].cat.codes.to_numpy().astype(np.int32)
    queries = np.random.default_rng(1).uniform(0, 1000, (500, 2))
    tree = _build_kdtree(cp.asarray(coords), cp.asarray(codes))
    for cluster in range(codes.max() + 1):
        distances = _query(cp.asarray(queries), tree, cluster, 2)
        expected, _ = (
            NearestNeighbors(n_neighbors=2)
            .fit(coords[codes == cluster])
            .kneighbors(queries)
        )
        np.testing.assert_allclose(distances.get(), expected)


def test_ppp_uniform_in_hull():
    # The hull is the triangle (0, 0), (2, 0), (0, 2), half its bounding box.
    corners = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    inner = np.random.default_rng(0).uniform(0, 1, (50, 2))
    hull = ConvexHull(np.vstack([corners, inner]))
    points = _ppp(hull, 20000, cp.random.default_rng(0)).get()
    assert points.shape == (20000, 2)
    assert (points.min(axis=0) >= 0).all()
    assert (points.sum(axis=1) <= 2).all()
    # A quarter of the triangle's area lies at x > 1.
    assert abs((points[:, 0] > 1).mean() - 0.25) < 0.02


def test_ripley_cupy_coordinates(adata: AnnData):
    expected = rsc.gr.ripley(adata, CLUSTER_KEY, mode="G", copy=True, rng=0)
    adata.obsm["spatial"] = cp.asarray(adata.obsm["spatial"])
    actual = rsc.gr.ripley(adata, CLUSTER_KEY, mode="G", copy=True, rng=0)
    pd.testing.assert_frame_equal(actual["G_stat"], expected["G_stat"])


def test_ripley_invalid_inputs(adata: AnnData):
    with pytest.raises(ValueError, match="Unsupported metric"):
        rsc.gr.ripley(adata, CLUSTER_KEY, mode="L", metric="cosine")
    with pytest.raises(ValueError, match="mode"):
        rsc.gr.ripley(adata, CLUSTER_KEY, mode="K")
    with pytest.raises(ValueError, match="missing labels"):
        missing = adata.copy()
        missing.obs[CLUSTER_KEY] = missing.obs[CLUSTER_KEY].astype(object)
        missing.obs.iloc[0, missing.obs.columns.get_loc(CLUSTER_KEY)] = np.nan
        missing.obs[CLUSTER_KEY] = missing.obs[CLUSTER_KEY].astype("category")
        rsc.gr.ripley(missing, CLUSTER_KEY)
    with pytest.raises(ValueError, match="n_neigh"):
        rsc.gr.ripley(adata, CLUSTER_KEY, mode="G", n_neigh=500)
    adata.obsm["spatial"] = np.hstack([adata.obsm["spatial"]] * 2)[:, :3]
    with pytest.raises(ValueError, match="2D"):
        rsc.gr.ripley(adata, CLUSTER_KEY)
