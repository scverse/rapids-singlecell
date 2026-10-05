from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np
import pytest
from anndata import read_h5ad
from scipy import sparse

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
    np.testing.assert_allclose(
        df1["pval_norm"].values, df2["pval_norm"].values, atol=1e-5, rtol=1e-5
    )
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

    df_f32 = spatial_autocorr(
        adata, mode="moran", copy=True, n_perms=None, dtype=np.float32
    )
    df_f64 = spatial_autocorr(
        adata, mode="moran", copy=True, n_perms=None, dtype=np.float64
    )

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


def _adata_with_constant_genes(*, sparse_x: bool):
    """Random data where gene_0 is all zeros and gene_1 is a nonzero constant."""
    from anndata import AnnData
    from sklearn.neighbors import kneighbors_graph

    rng = np.random.default_rng(42)
    n_cells, n_genes = 100, 6
    X = rng.random((n_cells, n_genes)).astype(np.float32)
    X[:, 0] = 0.0
    X[:, 1] = 1.0
    adata = AnnData(sparse.csr_matrix(X) if sparse_x else X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    positions = rng.random((n_cells, 2))
    adata.obsp["spatial_connectivities"] = kneighbors_graph(
        positions, n_neighbors=5, mode="connectivity"
    )
    return adata


@pytest.mark.parametrize("mode", ["moran", "geary"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse_x", [True, False])
@pytest.mark.parametrize("n_perms", [None, 20])
def test_autocorr_constant_genes_are_nan(mode, dtype, sparse_x, n_perms):
    """Constant genes get NaN rows and a warning; the other genes are unaffected."""
    adata = _adata_with_constant_genes(sparse_x=sparse_x)
    stat = "I" if mode == "moran" else "C"

    with pytest.warns(UserWarning, match=r"2 gene\(s\) are constant.*gene_0, gene_1"):
        df = spatial_autocorr(adata, mode=mode, copy=True, n_perms=n_perms, dtype=dtype)

    assert set(df.index) == set(adata.var_names)
    assert df.loc[["gene_0", "gene_1"]].isna().all().all()
    others = [f"gene_{i}" for i in range(2, 6)]
    assert np.isfinite(df.loc[others, stat]).all()
    assert df.index[-2:].isin(["gene_0", "gene_1"]).all()  # NaN rows sort last

    # Scores and corrected p-values match a run without the constant genes.
    ref = spatial_autocorr(
        adata[:, others].copy(), mode=mode, copy=True, n_perms=None, dtype=dtype
    )
    np.testing.assert_allclose(
        df.loc[others, stat], ref.loc[others, stat], rtol=1e-5, atol=1e-7
    )
    if n_perms is None:
        np.testing.assert_allclose(
            df.loc[others, "pval_norm_fdr_bh"],
            ref.loc[others, "pval_norm_fdr_bh"],
            rtol=1e-4,
        )


@pytest.mark.parametrize("mode", ["moran", "geary"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_perms", [None, 20])
def test_autocorr_dense_int64_graph_indices(mode, dtype, n_perms):
    """Dense data with an int64-indexed graph matches the int32-indexed result."""
    from anndata import AnnData
    from sklearn.neighbors import kneighbors_graph

    rng = np.random.default_rng(0)
    X = rng.random((100, 4)).astype(np.float32)
    graph = sparse.csr_matrix(kneighbors_graph(rng.random((100, 2)), n_neighbors=5))
    stat = "I" if mode == "moran" else "C"
    scores = {}
    for idx_dtype in (np.int32, np.int64):
        g = graph.copy()
        g.indices, g.indptr = g.indices.astype(idx_dtype), g.indptr.astype(idx_dtype)
        adata = AnnData(X)
        adata.obsp["spatial_connectivities"] = g
        df = spatial_autocorr(adata, mode=mode, copy=True, n_perms=n_perms, dtype=dtype)
        scores[idx_dtype] = df[stat].sort_index().to_numpy()
    assert np.all(scores[np.int32] != 0)
    # float32 scores use atomics, so reruns differ by ~1e-6 relative.
    np.testing.assert_allclose(scores[np.int64], scores[np.int32], rtol=1e-5, atol=1e-7)


def test_autocorr_all_constant_genes_raise():
    adata = _adata_with_constant_genes(sparse_x=True)
    with pytest.raises(ValueError, match="Every selected gene is constant"):
        spatial_autocorr(adata, genes=["gene_0", "gene_1"], mode="moran", copy=True)


@pytest.mark.parametrize(
    ("dtype", "match"), [(np.float32, "dtype=np.float64"), (np.float64, "bug report")]
)
def test_check_precision_issues_messages(dtype, match):
    """The guard still raises for nan/inf that is not explained by constant genes."""
    from rapids_singlecell.squidpy_gpu._utils import _check_precision_issues

    with pytest.raises(ValueError, match=match):
        _check_precision_issues(cp.array([0.1, cp.nan], dtype=dtype), dtype)


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
    df_single = spatial_autocorr(
        adata_single, mode=mode, copy=True, n_perms=50, multi_gpu=False
    )
    # Multi GPU
    df_multi = spatial_autocorr(
        adata_multi, mode=mode, copy=True, n_perms=50, multi_gpu=True
    )

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
    df_single = spatial_autocorr(
        adata_single, mode=mode, copy=True, n_perms=50, multi_gpu=False
    )
    # Multi GPU
    df_multi = spatial_autocorr(
        adata_multi, mode=mode, copy=True, n_perms=50, multi_gpu=True
    )

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


@pytest.mark.parametrize("mode", ["moran", "geary"])
@pytest.mark.parametrize("n_perms", [49, 50, 51, 99, 100, 101])
@pytest.mark.parametrize("use_sparse", [True, False])
def test_autocorr_permutation_shape(mode, n_perms, use_sparse):
    """Test that permutation arrays have exact correct shape for any n_perms."""
    from cupyx.scipy import sparse as gpu_sparse

    from rapids_singlecell.squidpy_gpu._gearysc import _gearys_C_cupy
    from rapids_singlecell.squidpy_gpu._moransi import _morans_I_cupy

    # Create small test data
    np.random.seed(42)
    n_cells, n_genes = 50, 10
    X = np.random.rand(n_cells, n_genes).astype(np.float64)

    if use_sparse:
        data = gpu_sparse.csr_matrix(cp.array(X))
    else:
        data = cp.array(X)

    # Create simple adjacency matrix
    adj = gpu_sparse.random(
        n_cells, n_cells, density=0.1, format="csr", dtype=np.float64
    )
    adj = adj + adj.T  # Make symmetric

    if mode == "moran":
        _, perms = _morans_I_cupy(data, adj, n_permutations=n_perms)
    else:
        _, perms = _gearys_C_cupy(data, adj, n_permutations=n_perms)

    # Verify exact shape
    assert perms.shape == (n_perms, n_genes), (
        f"Expected ({n_perms}, {n_genes}), got {perms.shape}"
    )
