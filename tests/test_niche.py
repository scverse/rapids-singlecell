from __future__ import annotations

import inspect
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import read_h5ad
from cupyx.scipy import sparse as sparse_gpu
from scipy import sparse

from rapids_singlecell.gr import calculate_niche
from rapids_singlecell.squidpy_gpu._niche import (
    _neighborhood_profile,
    _utag_features,
)

DATA = Path(__file__).parent / "_data" / "dummy.h5ad"
SPATIAL_CONNECTIVITIES_KEY = "spatial_connectivities"
GROUPS = "cluster"


@pytest.fixture
def adata():
    a = read_h5ad(DATA)
    # _neighborhood_profile uses pd.Categorical on this column
    a.obs[GROUPS] = pd.Categorical(a.obs[GROUPS])
    return a


# -- semantic tests adapted from squidpy/tests/graph/test_niche.py (BSD-3) --


def test_niche_calc_nhood(adata):
    """Adapted from squidpy: profile shape, normalization, min_niche_size labels."""
    calculate_niche(
        adata,
        flavor="neighborhood",
        groups=GROUPS,
        n_neighbors=10,
        resolutions=[0.1],
        min_niche_size=20,
    )
    niches = adata.obs["nhood_niche_res=0.1"]

    # no NaNs, more cells in real niches than in 'not_a_niche'
    assert niches.isna().sum() == 0
    assert len(niches[niches != "not_a_niche"]) > len(niches[niches == "not_a_niche"])
    for label in niches.unique():
        if label != "not_a_niche":
            assert (niches == label).sum() >= 20

    # profile shape
    n_cats = len(adata.obs[GROUPS].cat.categories)
    rel = cp.asnumpy(
        _neighborhood_profile(
            adata,
            groups=GROUPS,
            distance=1,
            weights=None,
            abs_nhood=False,
            key=SPATIAL_CONNECTIVITIES_KEY,
        )
    )
    abs_ = cp.asnumpy(
        _neighborhood_profile(
            adata,
            groups=GROUPS,
            distance=1,
            weights=None,
            abs_nhood=True,
            key=SPATIAL_CONNECTIVITIES_KEY,
        )
    )
    assert rel.shape == (adata.n_obs, n_cats)
    assert abs_.shape == rel.shape

    # relative profile: each row sums to 1 when the cell has neighbors (all do here),
    # so total sum == n_obs and the per-row max sum is 1.
    np.testing.assert_allclose(rel.sum(axis=1).sum(), adata.n_obs, atol=1e-4)
    assert rel.sum(axis=1).max() == pytest.approx(1.0, abs=1e-5)

    # absolute profile: per-row sum equals that cell's degree in the spatial graph
    deg = np.asarray((adata.obsp[SPATIAL_CONNECTIVITIES_KEY] != 0).sum(axis=1)).ravel()
    np.testing.assert_array_equal(abs_.sum(axis=1).astype(int), deg)


def test_niche_calc_utag(adata):
    """Adapted from squidpy: utag output shape, sparsity, sensitivity to graph."""
    calculate_niche(adata, flavor="utag", n_neighbors=10, resolutions=[0.1, 1.0])

    niches_high = adata.obs["utag_niche_res=1.0"]
    niches_low = adata.obs["utag_niche_res=0.1"]
    assert niches_high.isna().sum() == 0
    # higher resolution → strictly more (or at least as many) clusters
    assert niches_high.nunique() >= niches_low.nunique()

    # output shape matches X (returns cupy.ndarray for dense X)
    feat = _utag_features(adata, SPATIAL_CONNECTIVITIES_KEY)
    assert feat.shape == adata.X.shape

    # sparsity preserved when input X is sparse (returns cupyx sparse)
    a_sparse = adata.copy()
    a_sparse.X = sparse.csr_matrix(adata.X)
    feat_sparse = _utag_features(a_sparse, SPATIAL_CONNECTIVITIES_KEY)
    assert sparse_gpu.issparse(feat_sparse)
    assert feat_sparse.shape == adata.X.shape

    # different spatial graph structure → different feature matrix
    # (uniform value scaling is invisible after row-normalization, so we drop edges)
    a2 = adata.copy()
    G = a2.obsp[SPATIAL_CONNECTIVITIES_KEY].tolil()
    G[0, :] = 0
    G[1, :] = 0
    G = G.tocsr()
    G.eliminate_zeros()
    a2.obsp[SPATIAL_CONNECTIVITIES_KEY] = G
    feat2 = _utag_features(a2, SPATIAL_CONNECTIVITIES_KEY)
    assert not cp.allclose(feat, feat2)


# -- additional rsc-specific tests --


@pytest.mark.parametrize("flavor", ["neighborhood", "utag"])
def test_basic_runs_inplace(adata, flavor):
    kw = {"groups": GROUPS} if flavor == "neighborhood" else {}
    out = calculate_niche(adata, flavor=flavor, n_neighbors=10, resolutions=0.5, **kw)
    assert out is None
    prefix = "nhood_niche" if flavor == "neighborhood" else "utag_niche"
    col = f"{prefix}_res=0.5"
    assert col in adata.obs.columns
    assert isinstance(adata.obs[col].dtype, pd.CategoricalDtype)


def test_copy_returns_new_object(adata):
    before = list(adata.obs.columns)
    out = calculate_niche(
        adata,
        flavor="neighborhood",
        groups=GROUPS,
        n_neighbors=10,
        resolutions=0.5,
        copy=True,
    )
    assert out is not None
    assert "nhood_niche_res=0.5" in out.obs.columns
    assert list(adata.obs.columns) == before


def test_multiple_resolutions(adata):
    calculate_niche(
        adata,
        flavor="neighborhood",
        groups=GROUPS,
        n_neighbors=10,
        resolutions=[0.3, 0.7],
    )
    assert "nhood_niche_res=0.3" in adata.obs.columns
    assert "nhood_niche_res=0.7" in adata.obs.columns


def test_n_hop_neighbors(adata):
    calculate_niche(
        adata,
        flavor="neighborhood",
        groups=GROUPS,
        n_neighbors=10,
        resolutions=0.5,
        distance=3,
        n_hop_weights=[1.0, 0.5, 0.25],
    )
    assert "nhood_niche_res=0.5" in adata.obs.columns


def test_min_niche_size_relabels_all(adata):
    """min_niche_size > n_obs should send every cell to 'not_a_niche'."""
    calculate_niche(
        adata,
        flavor="neighborhood",
        groups=GROUPS,
        n_neighbors=10,
        resolutions=2.0,
        min_niche_size=adata.n_obs + 1,
    )
    labels = adata.obs["nhood_niche_res=2.0"].astype(str)
    assert (labels == "not_a_niche").all()


def test_determinism_same_seed(adata):
    a1, a2 = adata.copy(), adata.copy()
    calculate_niche(
        a1,
        flavor="neighborhood",
        groups=GROUPS,
        n_neighbors=10,
        resolutions=0.5,
        random_state=42,
    )
    calculate_niche(
        a2,
        flavor="neighborhood",
        groups=GROUPS,
        n_neighbors=10,
        resolutions=0.5,
        random_state=42,
    )
    np.testing.assert_array_equal(
        a1.obs["nhood_niche_res=0.5"].astype(str).values,
        a2.obs["nhood_niche_res=0.5"].astype(str).values,
    )


def test_determinism_utag_same_seed(adata):
    a1, a2 = adata.copy(), adata.copy()
    calculate_niche(a1, flavor="utag", n_neighbors=10, resolutions=0.5, random_state=7)
    calculate_niche(a2, flavor="utag", n_neighbors=10, resolutions=0.5, random_state=7)
    np.testing.assert_array_equal(
        a1.obs["utag_niche_res=0.5"].astype(str).values,
        a2.obs["utag_niche_res=0.5"].astype(str).values,
    )


def test_unknown_flavor_raises(adata):
    with pytest.raises(ValueError, match="Unknown flavor"):
        calculate_niche(adata, flavor="bogus", n_neighbors=10, resolutions=0.5)


def test_neighborhood_requires_groups(adata):
    with pytest.raises(ValueError, match="`groups` is required"):
        calculate_niche(adata, flavor="neighborhood", n_neighbors=10, resolutions=0.5)


def test_groups_not_in_obs_raises(adata):
    with pytest.raises(KeyError):
        calculate_niche(
            adata,
            flavor="neighborhood",
            groups="missing_col",
            n_neighbors=10,
            resolutions=0.5,
        )


def test_missing_connectivities_raises(adata):
    del adata.obsp["spatial_connectivities"]
    with pytest.raises(KeyError, match="spatial_connectivities"):
        calculate_niche(
            adata, flavor="neighborhood", groups=GROUPS, n_neighbors=10, resolutions=0.5
        )


def test_invalid_distance_raises(adata):
    with pytest.raises(ValueError, match="distance"):
        calculate_niche(
            adata,
            flavor="neighborhood",
            groups=GROUPS,
            n_neighbors=10,
            resolutions=0.5,
            distance=0,
        )


def test_custom_connectivity_key(adata):
    adata.obsp["my_graph"] = adata.obsp["spatial_connectivities"]
    del adata.obsp["spatial_connectivities"]
    calculate_niche(
        adata,
        flavor="neighborhood",
        groups=GROUPS,
        n_neighbors=10,
        resolutions=0.5,
        spatial_connectivities_key="my_graph",
    )
    assert "nhood_niche_res=0.5" in adata.obs.columns


# -- cellcharter flavor tests --


def test_cellcharter_basic(adata):
    calculate_niche(adata, flavor="cellcharter", n_components=4)
    assert "cellcharter_niche" in adata.obs.columns
    col = adata.obs["cellcharter_niche"]
    assert isinstance(col.dtype, pd.CategoricalDtype)
    assert col.isna().sum() == 0
    assert col.nunique() <= 4


def test_calculate_niche_exposes_only_gmm_init_extension():
    params = inspect.signature(calculate_niche).parameters
    assert "init" not in params
    assert "kmeans_n_init" not in params
    assert "gmm_init" in params
    assert params["gmm_init"].default == "random_from_data"
    assert params["random_state"].default == 42


def test_cellcharter_distance_zero(adata):
    """distance=0 falls back to PCA + GMM on raw X (no shell aggregation)."""
    calculate_niche(adata, flavor="cellcharter", n_components=3, distance=0)
    assert adata.obs["cellcharter_niche"].nunique() <= 3


def test_cellcharter_use_rep(adata):
    """use_rep skips shell-aggregation and PCA; uses adata.obsm[key] directly."""
    rng = np.random.default_rng(0)
    adata.obsm["X_test"] = rng.standard_normal((adata.n_obs, 10)).astype(np.float32)
    calculate_niche(adata, flavor="cellcharter", n_components=4, use_rep="X_test")
    assert "cellcharter_niche" in adata.obs.columns


@pytest.mark.parametrize("gmm_init", ["random_from_data", "kmeans", "sklearn_kmeans"])
def test_cellcharter_gmm_init_options(adata, gmm_init):
    rng = np.random.default_rng(0)
    adata.obsm["X_test"] = rng.standard_normal((adata.n_obs, 10)).astype(np.float32)
    calculate_niche(
        adata,
        flavor="cellcharter",
        n_components=4,
        use_rep="X_test",
        gmm_init=gmm_init,
        random_state=0,
    )
    assert "cellcharter_niche" in adata.obs.columns


def test_cellcharter_determinism(adata):
    a1 = adata.copy()
    a2 = adata.copy()
    calculate_niche(a1, flavor="cellcharter", n_components=4, random_state=42)
    calculate_niche(a2, flavor="cellcharter", n_components=4, random_state=42)
    np.testing.assert_array_equal(
        a1.obs["cellcharter_niche"].astype(str).values,
        a2.obs["cellcharter_niche"].astype(str).values,
    )


def test_cellcharter_variance(adata):
    """`aggregation="variance"` runs and produces a categorical column."""
    calculate_niche(adata, flavor="cellcharter", n_components=4, aggregation="variance")
    assert "cellcharter_niche" in adata.obs.columns
    assert isinstance(adata.obs["cellcharter_niche"].dtype, pd.CategoricalDtype)


def test_cellcharter_invalid_aggregation(adata):
    with pytest.raises(ValueError, match="aggregation"):
        calculate_niche(
            adata, flavor="cellcharter", n_components=4, aggregation="bogus"
        )


def test_cellcharter_invalid_gmm_init(adata):
    with pytest.raises(ValueError, match="gmm_init"):
        calculate_niche(adata, flavor="cellcharter", n_components=4, gmm_init="bogus")


def test_cellcharter_bad_n_components(adata):
    with pytest.raises(ValueError, match="n_components"):
        calculate_niche(adata, flavor="cellcharter", n_components=0)


def test_cellcharter_missing_use_rep(adata):
    with pytest.raises(KeyError):
        calculate_niche(
            adata, flavor="cellcharter", n_components=4, use_rep="not_there"
        )


def test_cellcharter_use_rep_too_few_dims(adata):
    adata.obsm["X_small"] = np.zeros((adata.n_obs, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="at least"):
        calculate_niche(adata, flavor="cellcharter", n_components=10, use_rep="X_small")


def test_cellcharter_invalid_distance_negative(adata):
    with pytest.raises(ValueError, match="distance"):
        calculate_niche(adata, flavor="cellcharter", n_components=4, distance=-1)


def test_cellcharter_handles_zero_columns_after_shell_aggregation(adata):
    """Regression: shell-aggregated features can contain all-zero columns when
    a gene's expression never propagates into a given shell. The sparse PCA
    path rejects zero columns, so `_run_cellcharter` must drop them before PCA.
    Use a sparse X with an injected zero gene to force the sparse PCA route."""
    X = adata.X
    X = X.toarray() if sparse.issparse(X) else np.array(X, copy=True)
    X[:, 0] = 0
    adata.X = sparse.csr_matrix(X.astype(np.float32))
    calculate_niche(adata, flavor="cellcharter", n_components=4, distance=2)
    assert "cellcharter_niche" in adata.obs.columns
    assert adata.obs["cellcharter_niche"].isna().sum() == 0
