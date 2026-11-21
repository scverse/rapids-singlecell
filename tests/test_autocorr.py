from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad
from cupyx.scipy import sparse as sparse_gpu
from scipy import sparse, stats

from rapids_singlecell.gr import spatial_autocorr

MULTI_GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() >= 2

MORAN_I = "moranI"
GEARY_C = "gearyC"


@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_autocorr_consistency(mode):
    file = Path(__file__).parent / Path("_data/dummy.h5ad")
    dummy_adata = read_h5ad(file)

    spatial_autocorr(dummy_adata, mode=mode)
    df1 = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_perms=50)
    df2 = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_perms=50)

    idx_df = df1.index.values
    idx_adata = dummy_adata[:, dummy_adata.var.highly_variable.values].var_names.values

    if mode == "moran":
        UNS_KEY = MORAN_I
    elif mode == "geary":
        UNS_KEY = GEARY_C

    assert UNS_KEY in dummy_adata.uns.keys()
    assert "pval_sim_fdr_bh" in df1
    assert "pval_norm_fdr_bh" in dummy_adata.uns[UNS_KEY]
    assert dummy_adata.uns[UNS_KEY].columns.shape == (4,)
    assert df1.columns.shape == (9,)
    # test pval_norm same
    np.testing.assert_allclose(df1["pval_norm"].values, df2["pval_norm"].values, atol=1e-5, rtol=1e-5)
    # test highly variable
    assert dummy_adata.uns[UNS_KEY].shape != df1.shape
    # assert idx are sorted and contain same elements
    assert not np.array_equal(idx_df, idx_adata)


@pytest.mark.parametrize("mode", ["moran", "geary"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_autocorr_sparse(mode, dtype):
    """Test spatial_autocorr with sparse data and different dtypes."""
    file = Path(__file__).parent / Path("_data/dummy.h5ad")
    dummy_adata = read_h5ad(file)

    # Convert to sparse with specified dtype
    dummy_adata.X = sparse.csr_matrix(dummy_adata.X, dtype=dtype)

    df = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_perms=None)

    stat_col = "I" if mode == "moran" else "C"
    assert stat_col in df.columns
    assert "pval_norm" in df.columns
    # Check no inf or nan values in the statistic
    assert not np.any(np.isinf(df[stat_col].values))
    # Some nan is expected for zero-variance genes, but not all
    assert not np.all(np.isnan(df[stat_col].values))


@pytest.mark.parametrize("mode", ["moran", "geary"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_autocorr_dense(mode, dtype):
    """Test spatial_autocorr with dense data and different dtypes."""
    file = Path(__file__).parent / Path("_data/dummy.h5ad")
    dummy_adata = read_h5ad(file)

    # Convert to dense with specified dtype
    if sparse.issparse(dummy_adata.X):
        dummy_adata.X = dummy_adata.X.toarray().astype(dtype)
    else:
        dummy_adata.X = dummy_adata.X.astype(dtype)

    df = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_perms=None)

    stat_col = "I" if mode == "moran" else "C"
    assert stat_col in df.columns
    assert "pval_norm" in df.columns
    # Check no inf or nan values in the statistic
    assert not np.any(np.isinf(df[stat_col].values))
    assert not np.all(np.isnan(df[stat_col].values))


@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_autocorr_sparse_dense_consistency(mode):
    """Test that sparse and dense give consistent results."""
    file = Path(__file__).parent / Path("_data/dummy.h5ad")
    adata_dense = read_h5ad(file)
    adata_sparse = read_h5ad(file)

    # Use float64 for both
    adata_dense.X = adata_dense.X.astype(np.float64)
    adata_sparse.X = sparse.csr_matrix(adata_sparse.X, dtype=np.float64)

    df_dense = spatial_autocorr(adata_dense, mode=mode, copy=True, n_perms=None)
    df_sparse = spatial_autocorr(adata_sparse, mode=mode, copy=True, n_perms=None)

    stat_col = "I" if mode == "moran" else "C"

    # Results should be very close between sparse and dense
    np.testing.assert_allclose(
        df_dense[stat_col].values,
        df_sparse[stat_col].values,
        rtol=1e-5,
        atol=1e-5,
    )


def test_autocorr_dtype_parameter():
    """Test that the dtype parameter works correctly."""
    file = Path(__file__).parent / Path("_data/dummy.h5ad")
    adata = read_h5ad(file)

    # Input is float64, but force float32 computation
    adata.X = adata.X.astype(np.float64)

    df_f32 = spatial_autocorr(adata, mode="moran", copy=True, n_perms=None, dtype=np.float32)
    df_f64 = spatial_autocorr(adata, mode="moran", copy=True, n_perms=None, dtype=np.float64)

    # Both should produce valid results
    assert not np.any(np.isinf(df_f32["I"].values))
    assert not np.any(np.isinf(df_f64["I"].values))

    # Results should be close but not identical due to precision differences
    np.testing.assert_allclose(
        df_f32["I"].values,
        df_f64["I"].values,
        rtol=1e-4,
        atol=1e-4,
    )


def test_autocorr_float32_nan_raises_error():
    """Test that float32 nan/inf raises an error with helpful message."""
    from anndata import AnnData

    # Create data with a constant gene (zero variance) that causes nan in float32
    np.random.seed(42)
    n_cells = 100
    n_genes = 5

    X = np.random.rand(n_cells, n_genes).astype(np.float32)
    # Make one gene constant - this will cause division by zero (nan)
    X[:, 0] = 1.0

    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    # Create simple connectivity
    from sklearn.neighbors import kneighbors_graph

    positions = np.random.rand(n_cells, 2)
    adj = kneighbors_graph(positions, n_neighbors=5, mode="connectivity")
    adata.obsp["spatial_connectivities"] = adj

    # Run with float32 - should raise error suggesting float64
    with pytest.raises(ValueError, match="float64"):
        spatial_autocorr(adata, mode="moran", copy=True, n_perms=None)

    # With float64, should also raise error (for constant genes) but with bug report message
    with pytest.raises(ValueError, match="bug report"):
        spatial_autocorr(adata, mode="moran", copy=True, n_perms=None, dtype=np.float64)


@pytest.mark.skipif(not MULTI_GPU_AVAILABLE, reason="Requires >= 2 GPUs")
@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_autocorr_multi_gpu_sparse(mode):
    """Test that multi-GPU gives same results as single GPU for sparse data."""
    file = Path(__file__).parent / Path("_data/dummy.h5ad")
    adata_single = read_h5ad(file)
    adata_multi = read_h5ad(file)

    # Use sparse float64
    adata_single.X = sparse.csr_matrix(adata_single.X, dtype=np.float64)
    adata_multi.X = sparse.csr_matrix(adata_multi.X, dtype=np.float64)

    # Single GPU
    df_single = spatial_autocorr(adata_single, mode=mode, copy=True, n_perms=50, multi_gpu=False)
    # Multi GPU
    df_multi = spatial_autocorr(adata_multi, mode=mode, copy=True, n_perms=50, multi_gpu=True)

    stat_col = "I" if mode == "moran" else "C"

    # Statistics should be identical (same computation, just parallelized)
    np.testing.assert_allclose(
        df_single[stat_col].values,
        df_multi[stat_col].values,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.skipif(not MULTI_GPU_AVAILABLE, reason="Requires >= 2 GPUs")
@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_autocorr_multi_gpu_dense(mode):
    """Test that multi-GPU gives same results as single GPU for dense data."""
    file = Path(__file__).parent / Path("_data/dummy.h5ad")
    adata_single = read_h5ad(file)
    adata_multi = read_h5ad(file)

    # Use dense float64
    adata_single.X = adata_single.X.astype(np.float64)
    adata_multi.X = adata_multi.X.astype(np.float64)

    # Single GPU
    df_single = spatial_autocorr(adata_single, mode=mode, copy=True, n_perms=50, multi_gpu=False)
    # Multi GPU
    df_multi = spatial_autocorr(adata_multi, mode=mode, copy=True, n_perms=50, multi_gpu=True)

    stat_col = "I" if mode == "moran" else "C"

    # Statistics should be identical
    np.testing.assert_allclose(
        df_single[stat_col].values,
        df_multi[stat_col].values,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.skipif(not MULTI_GPU_AVAILABLE, reason="Requires >= 2 GPUs")
@pytest.mark.parametrize("mode", ["moran", "geary"])
@pytest.mark.parametrize("n_perms", [49, 50, 51])
def test_autocorr_multi_gpu_odd_perms(mode, n_perms):
    """Test multi-GPU with odd number of permutations."""
    file = Path(__file__).parent / Path("_data/dummy.h5ad")
    adata = read_h5ad(file)
    adata.X = sparse.csr_matrix(adata.X, dtype=np.float64)

    df = spatial_autocorr(adata, mode=mode, copy=True, n_perms=n_perms, multi_gpu=True)

    stat_col = "I" if mode == "moran" else "C"
    assert not np.any(np.isinf(df[stat_col].values))
    # Check pval_sim exists (computed from permutations)
    assert "pval_sim" in df.columns


def _reference(x, w, mode):
    """Moran's I / Geary's C of every column of x under weights w (NumPy, float64)."""
    n = x.shape[0]
    if mode == "moran":
        xc = x - x.mean(axis=0)
        return (xc * (w @ xc)).sum(axis=0) / (xc * xc).sum(axis=0)
    rows, cols = w.nonzero()
    diff = x[rows] - x[cols]
    num = (np.asarray(w[rows, cols]).ravel()[:, None] * diff * diff).sum(axis=0)
    ss = ((x - x.mean(axis=0)) ** 2).sum(axis=0)
    return (n - 1) * num / (2 * w.sum() * ss)


@pytest.mark.parametrize("mode", ["moran", "geary"])
@pytest.mark.parametrize("n_perms", [None, 1, 16, 37])
@pytest.mark.parametrize("density", [0.3, 1.0])
def test_block_scores_match_reference(mode, n_perms, density):
    """Observed and permuted scores match NumPy with the kernel's permutations."""
    from rapids_singlecell._cuda import _autocorr_cuda as _ac
    from rapids_singlecell.squidpy_gpu._autocorr import _block_scores

    rng = np.random.default_rng(0)
    n, n_genes = 400, 21
    x = rng.random((n, n_genes)) * (rng.random((n, n_genes)) < density)
    x[:5] = 0  # rows without expression are skipped
    w = sparse.random(n, n, density=0.02, format="csr", random_state=1)
    w.setdiag(0)
    w.eliminate_zeros()
    keys = None
    if n_perms:
        keys = rng.integers(0, 2**32, (n_perms, _ac.ROUNDS), dtype=np.uint32)
    adj = sparse_gpu.csr_matrix(w)
    observed, changes = _block_scores(
        cp.asarray(x),
        (adj.indptr, adj.indices, adj.data),
        cp.asarray(np.asarray(w.sum(1)).ravel()),
        cp.asarray(np.asarray(w.sum(0)).ravel()),
        None if keys is None else cp.asarray(keys),
        total=w.sum(),
        uniform=False,
        geary=mode == "geary",
    )
    np.testing.assert_allclose(observed.get(), _reference(x, w, mode), rtol=1e-10)
    if n_perms is None:
        assert changes is None
        return
    perms = observed + changes
    assert perms.shape == (n_perms, n_genes)
    for p in range(n_perms):
        pi = cp.empty(n, dtype=cp.int32)
        _ac.permutation(cp.asarray(keys[p]), out=pi)
        pi = pi.get()
        assert np.array_equal(np.sort(pi), np.arange(n))
        np.testing.assert_allclose(perms[p].get(), _reference(x, w[pi], mode), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_autocorr_seed_blocks_and_dask(mode):
    """Seeded permutations are reproducible, over gene blocks and from Dask."""
    import dask.array as da

    rng = np.random.default_rng(3)
    n, n_genes = 600, 70  # three blocks of up to 32 genes
    x = sparse.random(n, n_genes, density=0.2, format="csr", random_state=4)
    pos = rng.random((n, 2))
    from sklearn.neighbors import kneighbors_graph

    adata = AnnData(x.astype(np.float32))
    adata.obsp["spatial_connectivities"] = kneighbors_graph(pos, 6)
    kw = {"mode": mode, "copy": True, "n_perms": 40, "seed": 0, "corr_method": None}
    df1 = spatial_autocorr(adata, **kw)
    df2 = spatial_autocorr(adata, **kw)
    pd.testing.assert_frame_equal(df1, df2)
    stat = "I" if mode == "moran" else "C"
    expected = _reference(
        x.toarray().astype(np.float64),
        adata.obsp["spatial_connectivities"].multiply(1 / 6).tocsr(),
        mode,
    )
    np.testing.assert_allclose(
        df1[stat].sort_index(),
        pd.Series(expected, index=adata.var_names).sort_index(),
        rtol=1e-5,
    )
    lazy = AnnData(da.from_array(x.astype(np.float32), chunks=(128, -1)))
    lazy.obsp["spatial_connectivities"] = adata.obsp["spatial_connectivities"]
    pd.testing.assert_frame_equal(df1, spatial_autocorr(lazy, **kw))


@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_autocorr_permutation_ties(mode):
    # Every row of W is the same. Permuting rows leaves every score unchanged.
    rng = np.random.default_rng(41)
    x = rng.normal(size=(25, 9)).astype(np.float32)
    w = np.tile(rng.random(len(x)).astype(np.float32), (len(x), 1))
    adata = AnnData(x)
    adata.obsp["spatial_connectivities"] = w
    df = spatial_autocorr(adata, mode=mode, n_perms=63, seed=17, copy=True)
    np.testing.assert_array_equal(df["pval_sim"], np.full(x.shape[1], 1 / 64))
    np.testing.assert_allclose(df["var_sim"], 0, atol=1e-29)


@pytest.mark.parametrize(("mode", "stat"), [("moran", "I"), ("geary", "C")])
def test_autocorr_one_permutation_sparse(mode, stat, monkeypatch):
    from rapids_singlecell._cuda import _autocorr_cuda as _ac

    rng = np.random.default_rng(12)
    n, n_genes = 128, 321
    monkeypatch.setattr("rapids_singlecell.squidpy_gpu._autocorr._CHUNK_BYTES", 32 << 10)
    x = rng.integers(1, 12, (n, n_genes)) * (rng.random((n, n_genes)) < 0.12)
    x[:, 0] = 0
    x[-1, 0] = 1  # No edge points to this cell. Gene 0 ties under the shuffle.
    csr = sparse.csr_matrix(x, dtype=np.float32)
    parts = np.column_stack((0.25 * csr.data, 0.75 * csr.data))
    adata = AnnData(sparse.csr_matrix((parts.ravel(), csr.indices.repeat(2), csr.indptr * 2), shape=x.shape))
    cells = np.arange(n)
    w = sparse.csr_matrix((np.ones(n), (cells, cells // 2)), shape=(n, n))
    adata.obsp["spatial_connectivities"] = w
    keys = np.random.default_rng(17).integers(0, 2**32, _ac.ROUNDS, dtype=np.uint32)
    pi = cp.empty(n, dtype=cp.int32)
    _ac.permutation(cp.asarray(keys), out=pi)
    observed = _reference(x, w, mode)
    tied = _reference(x, w[pi.get()], mode) == observed
    assert tied[0] and not tied.all()

    for cols in (np.arange(n_genes), np.arange(0, n_genes, 7)[::-1]):
        genes = adata.var_names[cols]
        result = spatial_autocorr(adata, genes=genes, mode=mode, n_perms=1, seed=17, multi_gpu=False, copy=True)
        result = result.loc[genes]
        np.testing.assert_allclose(result[stat], observed[cols], atol=1e-12)
        np.testing.assert_array_equal(result["var_sim"], 0)
        np.testing.assert_array_equal(result["pval_sim"], 0.5)
        np.testing.assert_equal(result["pval_z_sim"].to_numpy(), np.where(tied[cols], np.nan, 0.0))


@pytest.mark.parametrize(("mode", "stat"), [("moran", "I"), ("geary", "C")])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_autocorr_large_offset(mode, stat, dtype):
    rng = np.random.default_rng(17)
    x = rng.integers(-4, 5, (251, 3)).astype(dtype)
    shifted = x + dtype(2**23 if dtype == np.float32 else 2**50)
    w = sparse.random(len(x), len(x), density=0.04, format="csr", random_state=7).astype(dtype)
    adata = AnnData(shifted)
    adata.obsp["spatial_connectivities"] = w
    result = spatial_autocorr(adata, mode=mode, copy=True, transformation=False)
    np.testing.assert_allclose(
        result.loc[adata.var_names, stat],
        _reference(x.astype(np.float64), w.astype(np.float64), mode),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_array_equal(adata.X, shifted)


@pytest.mark.parametrize(("mode", "stat"), [("moran", "I"), ("geary", "C")])
def test_autocorr_wide_sparse(mode, stat):
    # A sparse matrix can have dense, nearly constant columns. Within each
    # connected component gene 1 differs only in its least significant bits.
    n, n_genes = 256, 1025
    rng = np.random.default_rng(52)
    x = rng.normal(size=(n, n_genes)) * (rng.random((n, n_genes)) < 0.04)
    x[:, 0] = 2**50 + rng.integers(-4, 5, n)
    x[: n // 2, 1] = x[: n // 2, 0]
    x[n // 2 :, 1] = 0
    cells = np.arange(n)
    neighbor = (cells + 1) % (n // 2) + cells // (n // 2) * (n // 2)
    w = sparse.csr_matrix((rng.uniform(0.1, 2, n), (cells, neighbor)), shape=(n, n))
    adata = AnnData(sparse.csr_matrix(x))
    adata.obsp["spatial_connectivities"] = w
    df = spatial_autocorr(adata, mode=mode, transformation=False, multi_gpu=False, copy=True)
    x[:, 0] -= x[0, 0]
    expected = _reference(x, w, mode)
    np.testing.assert_allclose(df.loc[adata.var_names, stat], expected, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(df.loc[adata.var_names[1], stat], expected[1], atol=0, rtol=1e-10)


@pytest.mark.parametrize(("mode", "stat"), [("moran", "I"), ("geary", "C")])
def test_autocorr_analytic_variance(mode, stat):
    """Analytic variance follows Cliff & Ord 1981 for each statistic (esda Moran/Geary)."""
    adata = read_h5ad(Path(__file__).parent / Path("_data/dummy.h5ad"))
    w = sparse.csr_matrix(adata.obsp["spatial_connectivities"], dtype=np.float64)
    n, s0 = w.shape[0], w.sum()
    s1 = (w + w.T).multiply(w + w.T).sum() / 2
    deg = np.asarray(w.sum(0)).ravel() + np.asarray(w.sum(1)).ravel()
    s2 = deg @ deg
    if mode == "moran":
        var = (n * n * s1 - n * s2 + 3 * s0**2) / ((n * n - 1) * s0**2)
        var -= 1 / (n - 1) ** 2
    else:
        var = ((2 * s1 + s2) * (n - 1) - 4 * s0**2) / (2 * (n + 1) * s0**2)
    df = spatial_autocorr(adata, mode=mode, transformation=False, copy=True, corr_method=None)
    np.testing.assert_allclose(df["var_norm"], var, rtol=1e-10)
    expected = -1 / (n - 1) if mode == "moran" else 1.0
    z = (df[stat] - expected) / np.sqrt(var)
    np.testing.assert_allclose(df["pval_norm"], stats.norm.sf(np.abs(z)), rtol=1e-6)


@pytest.mark.parametrize("source", ["cupy", "csr", "dask"])
def test_autocorr_selected_genes(source, monkeypatch):
    import dask.array as da

    rng = np.random.default_rng(41)
    shape = (71, 323)
    x = rng.normal(size=shape) * (rng.random(shape) < 0.4)
    monkeypatch.setattr("rapids_singlecell.squidpy_gpu._autocorr._CHUNK_BYTES", 4096)
    if source == "cupy":
        values = cp.asarray(x)
    else:
        csr = sparse.csr_matrix(x)
        fraction = rng.uniform(size=csr.nnz)
        entries = np.column_stack((fraction * csr.data, (1 - fraction) * csr.data))
        values = sparse.csr_matrix((entries.ravel(), csr.indices.repeat(2), csr.indptr * 2), shape=x.shape)
        if source == "dask":
            values = da.from_array(values, chunks=(20, 23))
    adata = AnnData(values)
    w = sparse.diags(np.ones(x.shape[0] - 1), offsets=1, format="csr")
    adata.obsp["spatial_connectivities"] = w
    cols = [*range(99, 65, -1), 2, 87]
    genes = adata.var_names[cols]
    result = spatial_autocorr(adata, genes=genes, copy=True, transformation=False, multi_gpu=False)
    expected = pd.Series(_reference(x[:, cols], w, "moran"), index=genes, name="I")
    pd.testing.assert_series_equal(result["I"].sort_index(), expected.sort_index(), rtol=1e-10, atol=1e-12)


def test_host_csr_chunks_share_entries():
    from rapids_singlecell.squidpy_gpu._autocorr import _csr_chunks

    x = sparse.csr_matrix(np.ones((80, 40), dtype=np.float32))
    for start, chunk in _csr_chunks(x, 4096):
        assert np.shares_memory(chunk.data, x.data)
        assert np.shares_memory(chunk.indices, x.indices)
        np.testing.assert_array_equal(chunk.toarray(), x[start : start + chunk.shape[0]].toarray())
