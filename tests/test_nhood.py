from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import read_h5ad
from scipy import sparse

import rapids_singlecell as rsc
from rapids_singlecell._cuda import _nhood_cuda as _nh


@pytest.fixture
def adata():
    adata = read_h5ad(Path(__file__).parent / "_data/dummy.h5ad")
    adata.obs["leiden"] = adata.obs["cluster"].astype("category")
    return adata


def _reference_counts(adata, key, *, weights=False):
    """Squidpy's interaction matrix, with NaN cells and their edges removed."""
    cats = adata.obs[key]
    mask = ~cats.isna().to_numpy()
    g = sparse.csr_matrix(adata.obsp["spatial_connectivities"])[mask][:, mask].tocoo()
    codes = cats.cat.codes.to_numpy()[mask]
    k = len(cats.cat.categories)
    out = np.zeros((k, k))
    np.add.at(out, (codes[g.row], codes[g.col]), g.data if weights else 1)
    return out


def _reference_zscore(adata, key, n_perms, seed):
    g = sparse.csr_matrix(adata.obsp["spatial_connectivities"]).tocoo()
    codes = adata.obs[key].cat.codes.to_numpy()
    k = len(adata.obs[key].cat.categories)

    def count(labels):
        return np.bincount(labels[g.row] * k + labels[g.col], minlength=k * k).reshape(
            k, k
        )

    rng = np.random.default_rng(seed)
    perms = np.stack([count(rng.permutation(codes)) for _ in range(n_perms)])
    return (count(codes) - perms.mean(0)) / perms.std(0)


@pytest.mark.parametrize("weights", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
def test_interaction_matrix(adata, weights, normalized):
    g = adata.obsp["spatial_connectivities"].tocsr()
    g.data = np.random.default_rng(0).random(g.nnz)
    adata.obsp["spatial_connectivities"] = g

    out = rsc.gr.interaction_matrix(
        adata, "leiden", weights=weights, normalized=normalized, copy=True
    )
    expected = _reference_counts(adata, "leiden", weights=weights)
    if normalized:
        expected = expected / expected.sum(axis=1, keepdims=True)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, expected)

    assert (
        rsc.gr.interaction_matrix(
            adata, "leiden", weights=weights, normalized=normalized
        )
        is None
    )
    np.testing.assert_allclose(adata.uns["leiden_interactions"], out)


@pytest.mark.parametrize("dtype", [bool, np.int32])
def test_interaction_matrix_integer_graph(adata, dtype):
    adata.obsp["spatial_connectivities"] = adata.obsp["spatial_connectivities"].astype(
        dtype
    )
    out = rsc.gr.interaction_matrix(adata, "leiden", copy=True)
    assert np.issubdtype(out.dtype, np.integer)
    np.testing.assert_array_equal(out, _reference_counts(adata, "leiden"))


def test_missing_labels_are_dropped(adata):
    """Cells without a label are removed with their edges, as in squidpy."""
    labels = adata.obs["leiden"].astype(object)
    labels.iloc[::7] = np.nan
    adata.obs["leiden"] = pd.Categorical(labels)
    expected = _reference_counts(adata, "leiden")

    np.testing.assert_array_equal(
        rsc.gr.interaction_matrix(adata, "leiden", copy=True), expected
    )
    res = rsc.gr.nhood_enrichment(adata, "leiden", n_perms=50, seed=0, copy=True)
    np.testing.assert_array_equal(res.counts, expected)

    adata.obs["leiden"] = pd.Categorical([np.nan] * adata.n_obs, categories=["a"])
    with pytest.raises(RuntimeError, match="none remain"):
        rsc.gr.interaction_matrix(adata, "leiden")


def test_nhood_enrichment(adata):
    rsc.gr.nhood_enrichment(adata, "leiden", n_perms=2000, seed=0)
    res = adata.uns["leiden_nhood_enrichment"]
    np.testing.assert_array_equal(res["count"], _reference_counts(adata, "leiden"))

    expected = _reference_zscore(adata, "leiden", n_perms=2000, seed=0)
    finite = np.isfinite(expected)
    assert np.isfinite(res["zscore"][finite]).all()
    # Independent permutations: only Monte Carlo error separates the two.
    np.testing.assert_allclose(
        res["zscore"][finite], expected[finite], rtol=0.1, atol=0.25
    )


def test_nhood_enrichment_seed(adata):
    kw = {"n_perms": 100, "copy": True}
    a = rsc.gr.nhood_enrichment(adata, "leiden", seed=1, **kw)
    b = rsc.gr.nhood_enrichment(adata, "leiden", seed=1, **kw)
    c = rsc.gr.nhood_enrichment(adata, "leiden", seed=2, **kw)
    np.testing.assert_array_equal(a.zscore, b.zscore)
    assert not np.array_equal(a.zscore, c.zscore)
    np.testing.assert_array_equal(a.counts, c.counts)


def test_nhood_enrichment_library_key(adata):
    """Labels only move within a library: one cluster per library never changes."""
    adata.obs["lib"] = adata.obs["leiden"].copy()
    res = rsc.gr.nhood_enrichment(
        adata, "leiden", library_key="lib", n_perms=20, seed=0, copy=True
    )
    # Every permutation reproduces the observed counts, so the std is zero.
    assert np.isnan(res.zscore).all()

    adata.obs["lib"] = pd.Categorical(np.arange(adata.n_obs) % 3)
    res = rsc.gr.nhood_enrichment(
        adata, "leiden", library_key="lib", n_perms=100, seed=0, copy=True
    )
    assert np.isfinite(res.zscore).any()


def test_nhood_enrichment_errors(adata):
    with pytest.raises(ValueError, match="n_perms"):
        rsc.gr.nhood_enrichment(adata, "leiden", n_perms=0)
    adata.obs["one"] = pd.Categorical(["a"] * adata.n_obs)
    with pytest.raises(ValueError, match="at least `2` clusters"):
        rsc.gr.nhood_enrichment(adata, "one")
    with pytest.raises(TypeError, match="categorical"):
        rsc.gr.nhood_enrichment(adata, "cluster")
    with pytest.raises(KeyError, match="not found"):
        rsc.gr.interaction_matrix(adata, "leiden", connectivity_key="missing")


@pytest.mark.parametrize("k", [3, 120, 200])
def test_permuted_counts_kernel(k):
    """Shuffles stay within libraries; k=120/200 use opt-in shared or global bins."""
    rng = np.random.default_rng(k)
    n, nnz, batch = 3000, 20000, 4
    rows = cp.asarray(rng.integers(0, n, nnz), dtype=cp.int32)
    cols = cp.asarray(rng.integers(0, n, nnz), dtype=cp.int32)
    labels = cp.asarray(rng.integers(0, k, n), dtype=cp.int32)
    lib = cp.asarray(rng.integers(0, 3, n))
    pos = cp.argsort(lib).astype(cp.int32)
    group_off = cp.searchsorted(lib[pos], cp.arange(4)).astype(cp.int32)
    seeds = cp.asarray(rng.integers(0, 2**64, batch, dtype=np.uint64))
    buf = cp.empty((batch, n), dtype=cp.int32)
    out = cp.zeros((batch, k, k), dtype=cp.uint64)
    # batch=3 runs two batches, the second one partial.
    _nh.permuted_counts(
        rows,
        cols,
        labels,
        pos,
        group_off,
        seeds,
        out=out,
        k=k,
        batch=3,
        labels_out=buf,
    )
    for b in range(batch):
        for g in range(3):
            cp.testing.assert_array_equal(
                cp.sort(buf[b][lib == g]), cp.sort(labels[lib == g])
            )
        expected = cp.bincount(
            buf[b, rows] * k + buf[b, cols], minlength=k * k
        ).reshape(k, k)
        cp.testing.assert_array_equal(out[b], expected)
    assert not (buf[0] == buf[1]).all()


@pytest.mark.parametrize("m", [2, 3, 4])
def test_small_groups_are_shuffled_uniformly(m):
    """Every permutation of a tiny library is equally likely (chi-square test)."""
    from scipy.stats import chisquare

    n_perms = 60_000
    labels = cp.arange(m, dtype=cp.int32)
    seeds = cp.asarray(
        np.random.default_rng(m).integers(0, 2**64, n_perms, dtype=np.uint64)
    )
    buf = cp.empty((n_perms, m), dtype=cp.int32)
    out = cp.zeros((n_perms, m, m), dtype=cp.uint64)
    edge = cp.zeros(1, dtype=cp.int32)
    _nh.permuted_counts(
        edge,
        edge,
        labels,
        labels,
        cp.asarray([0, m], dtype=cp.int32),
        seeds,
        out=out,
        k=m,
        batch=4096,
        labels_out=buf,
    )
    _, counts = np.unique(
        (buf * m ** cp.arange(m)).sum(axis=1).get(), return_counts=True
    )
    assert counts.size == np.prod(np.arange(1, m + 1))
    assert chisquare(counts).pvalue > 1e-4
