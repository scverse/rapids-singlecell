from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from cupyx.scipy import sparse as cusparse
from scanpy.datasets import pbmc3k, pbmc3k_processed
from scipy import sparse

import rapids_singlecell as rsc
from rapids_singlecell.preprocessing._sparse_pca._block_lanczos import randomized_svd
from rapids_singlecell.preprocessing._sparse_pca._svd_lanczos import lanczos_svd

A = pytest.importorskip("anndata.acc").A

# Reference values computed with sklearn PCA
A_list = [
    [0, 0, 7, 0, 0],
    [8, 5, 0, 2, 0],
    [6, 0, 0, 2, 5],
    [0, 0, 0, 1, 0],
    [8, 8, 2, 1, 0],
    [0, 0, 0, 4, 5],
]

A_pca = np.array(
    [
        [-4.4783009, 5.55508466, 1.73111572, -0.06029139, 0.17292555],
        [5.4855141, -0.42651191, -0.74776055, -0.74532146, 0.74633582],
        [0.01161428, -4.0156662, 2.37252748, -1.33122372, -0.29044446],
        [-3.61934397, 0.48525412, -2.96861931, -1.16312545, -0.33230607],
        [7.14050048, 1.86330409, -0.05786325, 1.25045782, -0.50213107],
        [-4.53998399, -3.46146476, -0.32940009, 2.04950419, 0.20562023],
    ]
)

A_svd = np.array(
    [
        [-0.77034038, -2.00750922, 6.64603489, -0.39669256, -0.22212097],
        [-9.47135856, -0.6326006, -1.33787112, -0.24894361, -1.02044665],
        [-5.90007339, 4.99658727, 0.70712592, -2.15188849, 0.30430008],
        [-0.19132409, 0.42172251, 0.11169531, 0.50977966, -0.71637566],
        [-11.1286238, -2.73045559, 0.08040596, 1.06850585, 0.74173764],
        [-1.50180389, 5.56886849, 1.64034442, 2.24476032, -0.05109001],
    ]
)


def _pbmc3k_normalized() -> AnnData:
    pbmc = pbmc3k()
    pbmc.X = pbmc.X.astype("float64")
    sc.pp.filter_genes(pbmc, min_counts=1)
    sc.pp.log1p(pbmc)
    sc.pp.normalize_total(pbmc)
    sc.pp.highly_variable_genes(pbmc)
    return pbmc


# =============================================================================
# Basic PCA correctness tests
# =============================================================================


@pytest.mark.parametrize("use_array", [False, True])
@pytest.mark.parametrize("use_sparse", [True, False])
def test_pca_correctness_zero_center(use_sparse, use_array):
    """Test PCA correctness against reference values (zero_center=True)."""
    A = np.array(A_list).astype("float64")
    if use_sparse:
        A = sparse.csr_matrix(A)

    adata = AnnData(A)
    if use_array:
        # feeding the matrix directly should return the embedding
        X_pca = rsc.pp.pca(adata.X, n_comps=4, zero_center=True)
    else:
        rsc.pp.pca(adata, n_comps=4, zero_center=True)
        X_pca = adata.obsm["X_pca"]

    # Compare absolute values (signs can flip)
    np.testing.assert_allclose(
        np.abs(cp.asnumpy(X_pca)),
        np.abs(A_pca[:, :4]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_pca_return_info_array():
    """`return_info=True` returns the 4-tuple for matrix input (Scanpy parity)."""
    A = np.array(A_list).astype("float64")
    out = rsc.pp.pca(cp.asarray(A), n_comps=4, return_info=True)
    assert isinstance(out, tuple) and len(out) == 4
    X_pca, components, variance_ratio, variance = out
    assert cp.asnumpy(X_pca).shape == (6, 4)
    assert cp.asnumpy(components).shape == (4, 5)
    assert cp.asnumpy(variance_ratio).shape == (4,)
    assert cp.asnumpy(variance).shape == (4,)
    np.testing.assert_allclose(np.abs(cp.asnumpy(X_pca)), np.abs(A_pca[:, :4]), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("use_sparse", [True, False])
def test_pca_correctness_no_center(use_sparse):
    """Test truncated SVD correctness (zero_center=False)."""
    A = np.array(A_list).astype("float64")
    if use_sparse:
        A = sparse.csr_matrix(A)

    adata = AnnData(A)
    rsc.pp.pca(adata, n_comps=4, zero_center=False)

    np.testing.assert_allclose(
        np.abs(adata.obsm["X_pca"]),
        np.abs(A_svd[:, :4]),
        rtol=1e-5,
        atol=1e-5,
    )


# =============================================================================
# Sparse solver correctness tests
# =============================================================================


@pytest.mark.parametrize("svd_solver", ["lanczos", "randomized", "covariance_eigh"])
@pytest.mark.parametrize("zero_center", [True, False])
def test_sparse_solver_correctness(svd_solver, zero_center, monkeypatch):
    """Test sparse solvers produce correct results compared to covariance_eigh."""
    rng = np.random.RandomState(42)
    X = sparse.random(200, 50, density=0.2, random_state=rng, format="csr")
    X = X.astype(np.float64)

    # Reference: covariance_eigh (exact method)
    ref = AnnData(X.copy())
    rsc.pp.pca(ref, n_comps=10, svd_solver="covariance_eigh", zero_center=zero_center)

    if svd_solver in ("lanczos", "randomized"):

        def reject_conversion(*args, **kwargs):
            pytest.fail("Sparse SVD materialized the transpose of its CSR input")

        monkeypatch.setattr(cusparse.csr_matrix, "tocsc", reject_conversion)
        monkeypatch.setattr(cusparse.csc_matrix, "tocsr", reject_conversion)

    # Test solver
    test = AnnData(X.copy())
    rsc.pp.pca(test, n_comps=10, svd_solver=svd_solver, zero_center=zero_center, random_state=0)

    # Check shapes
    assert test.obsm["X_pca"].shape == (200, 10)
    assert test.varm["PCs"].shape == (50, 10)

    # Variance should be positive and decreasing
    var = test.uns["pca"]["variance"]
    assert np.all(var > 0)
    assert np.all(var[:-1] >= var[1:])

    # Variance ratios should match closely
    if svd_solver == "randomized":
        # Randomized is approximate
        np.testing.assert_allclose(
            test.uns["pca"]["variance_ratio"][:5],
            ref.uns["pca"]["variance_ratio"][:5],
            rtol=0.05,
        )
    else:
        # Exact methods should match very closely
        np.testing.assert_allclose(
            test.uns["pca"]["variance_ratio"],
            ref.uns["pca"]["variance_ratio"],
            rtol=1e-5,
            atol=1e-8,
        )


@pytest.mark.parametrize("svd_solver", ["lanczos", "randomized"])
@pytest.mark.parametrize("zero_center", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sparse_solver_reproducibility(svd_solver, zero_center, dtype):
    X = sparse.random(200, 50, density=0.2, random_state=42, format="csr", dtype=dtype)
    # Duplicate coordinates must be combined before computing column variances.
    values = np.stack((X.data, X.data * 0.5, X.data * -0.25), axis=1).ravel()
    X = sparse.csr_matrix((values, X.indices.repeat(3), X.indptr * 3), shape=X.shape)

    def run():
        return rsc.pp.pca(
            X, n_comps=10, svd_solver=svd_solver, zero_center=zero_center, dtype=dtype, rng=42, return_info=True
        )

    first = run()
    for a, b in zip(first, run(), strict=True):
        assert cp.asnumpy(a).tobytes() == cp.asnumpy(b).tobytes()

    scores, components, variance_ratio, variance = first
    dense = X.toarray()
    if zero_center:
        dense -= dense.mean(axis=0)
    tol = 64 * np.finfo(dtype).eps
    np.testing.assert_allclose(cp.asnumpy(scores), dense @ cp.asnumpy(components).T, rtol=tol, atol=tol)
    total_variance = np.square(dense).sum() / (X.shape[0] - 1)
    np.testing.assert_allclose(cp.asnumpy(variance_ratio), cp.asnumpy(variance) / total_variance, rtol=tol)


@pytest.mark.parametrize("svd_solver", ["lanczos", "randomized"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_sparse_solver_complex_projection_and_variance(svd_solver, dtype):
    rng = np.random.default_rng(4510)
    X = sparse.random(54, 24, density=0.5, random_state=rng, format="csr").astype(dtype)
    X.data += 1j * rng.standard_normal(X.nnz)
    mask = np.arange(X.shape[1]) % 3 != 1
    selected = X[:, mask]
    kwargs = {"n_comps": 5, "svd_solver": svd_solver, "zero_center": False, "dtype": X.dtype, "rng": 42}
    scores, components, ratio, variance = map(cp.asnumpy, rsc.pp.pca(selected, return_info=True, **kwargs))
    tol = 64 * np.finfo(dtype).eps
    np.testing.assert_allclose(scores, selected @ components.conj().T, rtol=tol, atol=tol)
    assert np.isrealobj(ratio)
    values = selected.data.astype(np.complex128, copy=False)
    total_variance = np.vdot(values, values).real / (X.shape[0] - 1)
    np.testing.assert_allclose(ratio, variance / total_variance, rtol=tol)

    adata = AnnData(X)
    rsc.pp.pca(adata, mask=mask, **kwargs)
    loadings = cp.asnumpy(adata.varm["PCs"])
    masked_scores = cp.asnumpy(adata.obsm["X_pca"])
    assert loadings.dtype == np.result_type(dtype, np.float64)
    np.testing.assert_array_equal(loadings[~mask], 0)
    np.testing.assert_allclose(X @ loadings, masked_scores, rtol=tol, atol=tol)
    assert masked_scores.tobytes() == scores.tobytes()
    assert loadings[mask].tobytes() == components.conj().T.astype(loadings.dtype, copy=False).tobytes()
    assert adata.uns["pca"]["variance_ratio"].tobytes() == ratio.tobytes()
    assert adata.uns["pca"]["variance"].tobytes() == variance.tobytes()

    with pytest.raises(TypeError, match="real-valued"):
        rsc.pp.pca(X, n_comps=5, svd_solver=svd_solver, zero_center=True, rng=42)


@pytest.mark.parametrize("svd_solver", ["lanczos", "randomized"])
def test_centered_sparse_solver_variance_large_offset(svd_solver):
    rng = np.random.default_rng(43)
    X = rng.binomial(1, 0.25, (48, 16)).astype(np.float64)
    X[:, 0] = 1e10 + np.tile([-1, 0, 1], 16)
    X[:, 1] = 1e10
    csr = sparse.csr_matrix(X)
    _, _, ratio, variance = rsc.pp.pca(csr, n_comps=4, svd_solver=svd_solver, dtype=X.dtype, rng=42, return_info=True)
    total_variance = X.var(axis=0, ddof=1).sum()
    variance = cp.asnumpy(variance)
    assert np.all(np.isfinite(variance) & (variance > 0))
    np.testing.assert_allclose(cp.asnumpy(ratio), variance / total_variance, rtol=1e-12)


# =============================================================================
# Dtype and input format tests
# =============================================================================


@pytest.mark.parametrize("svd_solver", ["lanczos", "randomized", "covariance_eigh"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sparse_solver_dtypes(svd_solver, dtype):
    """Test sparse solvers with different dtypes produce correct results."""
    rng = np.random.RandomState(42)
    X = sparse.random(100, 50, density=0.3, random_state=rng, format="csr")
    X = X.astype(dtype)

    # Reference
    ref = AnnData(X.copy())
    rsc.pp.pca(ref, svd_solver="covariance_eigh", n_comps=10, dtype=dtype)

    # Test
    test = AnnData(X.copy())
    rsc.pp.pca(test, svd_solver=svd_solver, n_comps=10, dtype=dtype, random_state=0)

    # Check dtype preserved
    assert test.obsm["X_pca"].dtype == np.dtype(dtype)

    # Check correctness (looser tolerance for float32)
    rtol = 0.1 if svd_solver == "randomized" else (1e-3 if dtype == "float32" else 1e-5)
    np.testing.assert_allclose(
        test.uns["pca"]["variance_ratio"][:5],
        ref.uns["pca"]["variance_ratio"][:5],
        rtol=rtol,
    )


@pytest.mark.parametrize("svd_solver", ["lanczos", "randomized", "covariance_eigh"])
def test_cupy_sparse_input_correctness(svd_solver):
    """Test sparse solvers with CuPy sparse input produce correct results."""
    rng = np.random.RandomState(42)
    X_scipy = sparse.random(200, 100, density=0.2, random_state=rng, format="csr")
    X_scipy = X_scipy.astype(np.float64)

    # Reference with scipy sparse
    ref = AnnData(X_scipy.copy())
    rsc.pp.pca(ref, svd_solver="covariance_eigh", n_comps=10)

    # Test with cupy sparse
    X_cupy = cusparse.csr_matrix(X_scipy)
    test = AnnData(X_cupy)
    rsc.pp.pca(test, svd_solver=svd_solver, n_comps=10, random_state=0)

    assert test.obsm["X_pca"].shape == (200, 10)

    # Check correctness
    rtol = 0.05 if svd_solver == "randomized" else 1e-5
    np.testing.assert_allclose(
        test.uns["pca"]["variance_ratio"][:5],
        ref.uns["pca"]["variance_ratio"][:5],
        rtol=rtol,
    )


# =============================================================================
# Shape and edge case tests
# =============================================================================


def test_pca_shapes():
    """Test PCA output shapes."""
    adata = AnnData(np.random.randn(30, 20))
    rsc.pp.pca(adata)
    assert adata.obsm["X_pca"].shape == (30, 19)

    adata = AnnData(np.random.randn(20, 30))
    rsc.pp.pca(adata)
    assert adata.obsm["X_pca"].shape == (20, 19)

    with pytest.raises(ValueError):
        rsc.pp.pca(adata, n_comps=100)


def test_pca_chunked():
    """Test chunked PCA matches default PCA."""
    chunked = pbmc3k_processed()
    default = pbmc3k_processed()
    chunked.X = chunked.X.astype(np.float64)
    default.X = default.X.astype(np.float64)

    rsc.pp.pca(chunked, chunked=True, chunk_size=chunked.shape[0])
    rsc.pp.pca(default)

    np.testing.assert_allclose(
        np.abs(chunked.obsm["X_pca"]),
        np.abs(default.obsm["X_pca"]),
        rtol=1e-7,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.abs(chunked.varm["PCs"]), np.abs(default.varm["PCs"]), rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(
        chunked.uns["pca"]["variance_ratio"],
        default.uns["pca"]["variance_ratio"],
        rtol=1e-6,
    )


def test_pca_chunked_sparse():
    """Test chunked PCA works with sparse input (densifies per chunk)."""
    adata = sc.datasets.blobs(n_variables=50, n_centers=3, n_observations=200)
    adata.X = sparse.csr_matrix(adata.X.astype(np.float64))

    # Run chunked PCA on sparse - should work without errors
    rsc.pp.pca(adata, chunked=True, chunk_size=50)

    # Verify outputs are valid
    assert adata.obsm["X_pca"].shape == (200, 49)  # 50 vars, so max 49 comps
    assert adata.varm["PCs"].shape == (50, 49)
    assert len(adata.uns["pca"]["variance_ratio"]) == 49

    # Variance ratios should sum to <= 1 and be positive descending
    var_ratio = adata.uns["pca"]["variance_ratio"]
    assert np.all(var_ratio >= 0)
    assert np.all(var_ratio[:-1] >= var_ratio[1:])
    assert np.sum(var_ratio) <= 1.0 + 1e-6


def test_pca_reproducible():
    """Test PCA is reproducible across runs."""
    pbmc = pbmc3k_processed()
    pbmc.X = pbmc.X.astype(np.float32)

    rsc.tl.pca(pbmc)
    a = pbmc.obsm["X_pca"].copy()

    rsc.tl.pca(pbmc)
    b = pbmc.obsm["X_pca"]

    assert np.array_equal(a, b)

    # Test with cupy array input
    cpbmc = pbmc.copy()
    cpbmc.X = cp.array(cpbmc.X)
    rsc.tl.pca(cpbmc)
    c = cpbmc.obsm["X_pca"]

    np.testing.assert_allclose(a, c, rtol=1e-5, atol=1e-6)


def test_pca_sparse_vs_dense():
    """Test sparse and dense inputs produce same results."""
    pbmc = pbmc3k_processed()

    sparse_ad = pbmc.copy()
    dense_ad = pbmc.copy()

    sparse_ad.X = sparse.csr_matrix(sparse_ad.X.astype(np.float64))
    dense_ad.X = dense_ad.X.astype(np.float64)

    rsc.pp.pca(sparse_ad)
    rsc.pp.pca(dense_ad)

    np.testing.assert_allclose(
        np.abs(sparse_ad.obsm["X_pca"]),
        np.abs(dense_ad.obsm["X_pca"]),
        rtol=1e-7,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        sparse_ad.uns["pca"]["variance_ratio"],
        dense_ad.uns["pca"]["variance_ratio"],
        rtol=1e-6,
    )


# =============================================================================
# Mask tests
# =============================================================================


def test_mask_length_error():
    """Check error for mask length mismatch."""
    adata = AnnData(np.array(A_list).astype("float32"))
    mask_var = np.random.choice([True, False], adata.shape[1] + 1)
    with pytest.raises(ValueError, match=r"shape of the mask do not match"):
        rsc.pp.pca(adata, mask=mask_var)


def test_mask_var_non_boolean_error():
    """Non-boolean mask_var is rejected (Scanpy parity via `_check_mask`)."""
    adata = AnnData(np.array(A_list).astype("float32"))
    with pytest.raises(ValueError, match="boolean"):
        rsc.pp.pca(adata, mask=np.arange(adata.shape[1]))


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("array_type", ["array", cusparse.csr_matrix, cusparse.csc_matrix])
def test_mask_var_argument_equivalence(float_dtype, array_type):
    """Test mask as bool array vs string produces same results."""
    rng = np.random.RandomState(42)
    X = cp.random.random((100, 10), dtype=float_dtype)
    if array_type != "array":
        X = array_type(X)
    adata_base = AnnData(X)
    # Ensure at least 2 columns selected (cuML requires n_cols >= 2)
    mask_var = rng.choice([True, False], adata_base.shape[1])
    while mask_var.sum() < 2:
        mask_var = rng.choice([True, False], adata_base.shape[1])

    adata = adata_base.copy()
    rsc.pp.pca(adata, mask=mask_var, dtype=float_dtype)

    adata_w_mask = adata_base.copy()
    adata_w_mask.var["mask"] = mask_var
    rsc.pp.pca(adata_w_mask, mask="var.mask", dtype=float_dtype)

    assert cp.allclose(
        adata.X.toarray() if cusparse.issparse(adata.X) else adata.X,
        adata_w_mask.X.toarray() if cusparse.issparse(adata_w_mask.X) else adata_w_mask.X,
    )


def test_mask_correctness():
    """Test masked PCA matches PCA on subset."""
    adata = sc.datasets.blobs(n_variables=10, n_centers=3, n_observations=100)
    # fixed: a random mask selects fewer than the 2 variables cuML needs for 1% of seeds
    mask_var = np.arange(adata.shape[1]) % 2 == 0

    adata_masked = adata[:, mask_var].copy()
    rsc.pp.pca(adata, mask=mask_var)
    rsc.pp.pca(adata_masked)

    # Masked variables should have zero loadings
    masked_var_loadings = adata.varm["PCs"][~mask_var]
    np.testing.assert_equal(masked_var_loadings, np.zeros_like(masked_var_loadings))

    # Results should match subset
    np.testing.assert_equal(adata.obsm["X_pca"], adata_masked.obsm["X_pca"])
    np.testing.assert_allclose(adata.varm["PCs"][mask_var], adata_masked.varm["PCs"], rtol=1e-11)


@pytest.mark.parametrize("svd_solver", ["lanczos", "randomized", "covariance_eigh"])
def test_mask_sparse_solvers(svd_solver):
    """Test sparse solvers with variable mask produce correct results."""
    adata = sc.datasets.blobs(n_variables=30, n_centers=3, n_observations=100)
    adata.X = sparse.csr_matrix(adata.X.astype(np.float64))
    mask_var = np.zeros(adata.shape[1], dtype=bool)
    mask_var[:20] = True

    # Reference: subset without mask
    adata_subset = adata[:, mask_var].copy()
    rsc.pp.pca(adata_subset, svd_solver="covariance_eigh", n_comps=5)

    # Test: full data with mask
    adata_masked = adata.copy()
    rsc.pp.pca(adata_masked, svd_solver=svd_solver, mask=mask_var, n_comps=5)

    # Masked variables should have zero loadings
    masked_loadings = adata_masked.varm["PCs"][~mask_var]
    np.testing.assert_equal(masked_loadings, np.zeros_like(masked_loadings))

    # Variance ratios should match
    rtol = 0.05 if svd_solver == "randomized" else 1e-5
    np.testing.assert_allclose(
        adata_masked.uns["pca"]["variance_ratio"],
        adata_subset.uns["pca"]["variance_ratio"],
        rtol=rtol,
    )


@pytest.mark.parametrize("float_dtype", ["float32", "float64"])
def test_mask_defaults(float_dtype):
    """Test highly_variable is used as default mask."""
    A = cp.array(A_list).astype("float32")
    adata = AnnData(A)

    without_var = adata.copy()
    rsc.pp.pca(without_var, dtype=float_dtype)

    mask = np.array([True, True, False, True, False])
    adata.var["highly_variable"] = mask

    with_var = adata.copy()
    rsc.pp.pca(with_var, dtype=float_dtype)

    assert without_var.uns["pca"]["params"]["mask_var"] is None
    assert with_var.uns["pca"]["params"]["mask_var"] == "highly_variable"
    assert not np.array_equal(without_var.obsm["X_pca"], with_var.obsm["X_pca"])

    with_no_mask = adata.copy()
    rsc.pp.pca(with_no_mask, mask=None, dtype=float_dtype)
    assert np.array_equal(without_var.obsm["X_pca"], with_no_mask.obsm["X_pca"])


def _mask_var_adata() -> AnnData:
    adata = AnnData(np.random.default_rng(0).normal(size=(20, 5)).astype(np.float32))
    adata.var["highly_variable"] = [True, True, False, False, True]
    return adata


def test_mask_var_default_uses_highly_variable():
    """The default picks up `.var['highly_variable']` when it exists."""
    adata = _mask_var_adata()
    rsc.pp.pca(adata, n_comps=2)
    assert adata.uns["pca"]["params"]["mask_var"] == "highly_variable"


def test_mask_var_ndarray():
    """Passing an ndarray mask_var records no name and masks the loadings."""
    adata = _mask_var_adata()
    mask = np.array([True, False, True, False, True])
    rsc.pp.pca(adata, n_comps=2, mask=mask)
    assert adata.uns["pca"]["params"]["mask_var"] is None
    assert np.array_equal(adata.varm["PCs"][~mask], np.zeros((2, 2)))


# =============================================================================
# Layer tests
# =============================================================================


def test_pca_layer():
    """Test layer input produces same results as X."""
    X_adata = _pbmc3k_normalized()
    X_adata.X = X_adata.X.astype(np.float64)

    layer_adata = X_adata.copy()
    layer_adata.layers["counts"] = X_adata.X.copy()
    del layer_adata.X

    rsc.pp.pca(X_adata)
    rsc.pp.pca(layer_adata, use="layers.counts")

    assert layer_adata.uns["pca"]["params"]["layer"] == "counts"
    assert "layer" not in X_adata.uns["pca"]["params"]

    np.testing.assert_almost_equal(X_adata.uns["pca"]["variance"], layer_adata.uns["pca"]["variance"])
    np.testing.assert_almost_equal(X_adata.uns["pca"]["variance_ratio"], layer_adata.uns["pca"]["variance_ratio"])
    np.testing.assert_almost_equal(X_adata.obsm["X_pca"], layer_adata.obsm["X_pca"])
    np.testing.assert_almost_equal(X_adata.varm["PCs"], layer_adata.varm["PCs"])


def test_pca_deprecated_params():
    adata = _mask_var_adata()
    adata.layers["counts"] = adata.X.copy()
    with pytest.warns(FutureWarning, match="layer"):
        rsc.pp.pca(adata, n_comps=2, layer="counts")
    with pytest.warns(FutureWarning, match="key_added"):
        rsc.pp.pca(adata, n_comps=2, key_added="k")
    assert {"k"} <= set(adata.obsm) & set(adata.varm) & set(adata.uns)
    with pytest.warns(FutureWarning, match="copy"):
        copied = rsc.pp.pca(adata, n_comps=2, copy=True)
    assert "X_pca" in copied.obsm
    with pytest.warns(FutureWarning, match="mask_var"):
        rsc.pp.pca(adata, n_comps=2, mask_var="highly_variable", out=A.obsm["m"])
    assert adata.uns["m"]["params"]["mask_var"] == "highly_variable"
    np.testing.assert_allclose(np.abs(adata.obsm["m"]), np.abs(adata.obsm["X_pca"]), atol=1e-5)
    with pytest.warns(FutureWarning, match="mask_var"), pytest.raises(TypeError, match="either `mask` or `mask_var`"):
        rsc.pp.pca(adata, mask="var.highly_variable", mask_var="highly_variable")


def test_pca_mask_ref():
    adata = _mask_var_adata()
    rsc.pp.pca(adata, n_comps=2, mask=A.var["highly_variable"])
    assert adata.uns["pca"]["params"]["mask_var"] == ['["var", "highly_variable"]']
    with pytest.raises(ValueError, match="incompatible with `.obsm`"):
        rsc.pp.pca(adata, use=A.obsm["X_pca"], mask="var.highly_variable", out=A.obsm["x"])


def test_pca_out():
    adata = _mask_var_adata()
    res = rsc.pp.pca(adata, n_comps=2, out=True)
    assert res.embedding.shape == (20, 2)
    assert res.loadings.shape == (5, 2)
    assert not ({"X_pca"} & set(adata.obsm) or "pca" in adata.uns)

    rsc.pp.pca(adata, n_comps=2, out=A.obsm["p"])
    assert "p" in adata.obsm and "p" in adata.varm and "p" in adata.uns
    np.testing.assert_allclose(np.abs(cp.asnumpy(adata.obsm["p"])), np.abs(cp.asnumpy(res.embedding)), atol=1e-5)

    rsc.pp.pca(adata, n_comps=2, out={"embedding": "obsm.e"})
    assert "e" in adata.obsm and "e" not in adata.varm and "e" in adata.uns

    rsc.pp.pca(adata, n_comps=2, use=A.obsm["p"], out=A.obsm["pp"])
    assert "pp" not in adata.varm
    assert adata.uns["pp"]["components"].shape == (2, 2)
    assert adata.uns["pp"]["params"]["obsm"] == "p"


@pytest.mark.filterwarnings("ignore:Some cells have zero counts:UserWarning")
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
@pytest.mark.parametrize("svd_solver", [None, "lanczos", "randomized"])
def test_pca_layer_mask(svd_solver):
    """Test error when layer has zero-expression genes."""
    adata = sc.datasets.pbmc3k()[:, 1000].copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    with pytest.raises(
        ValueError,
        match="There are genes with zero expression. Please remove them before running PCA.",
    ):
        rsc.pp.pca(adata, svd_solver=svd_solver)


# =============================================================================
# Accuracy benchmark tests
# =============================================================================


def test_lanczos_accuracy():
    """Test Lanczos matches covariance_eigh at high precision."""
    pbmc = pbmc3k_processed()
    pbmc.X = sparse.csr_matrix(pbmc.X.astype(np.float64))

    ref = pbmc.copy()
    rsc.pp.pca(ref, svd_solver="covariance_eigh", n_comps=10)

    test = pbmc.copy()
    rsc.pp.pca(test, svd_solver="lanczos", random_state=0, n_comps=10)

    ref_pca = ref.obsm["X_pca"]
    test_pca = test.obsm["X_pca"]
    if hasattr(ref_pca, "get"):
        ref_pca = ref_pca.get()
    if hasattr(test_pca, "get"):
        test_pca = test_pca.get()

    # First 7 components (well-separated singular values) should match precisely
    np.testing.assert_allclose(
        np.abs(test_pca[:, :7]),
        np.abs(ref_pca[:, :7]),
        rtol=1e-7,
        atol=1e-7,
    )

    # Variance ratios should match very closely
    np.testing.assert_allclose(
        test.uns["pca"]["variance_ratio"],
        ref.uns["pca"]["variance_ratio"],
        rtol=1e-6,
        atol=1e-9,
    )


def test_randomized_accuracy():
    """Test randomized SVD captures variance accurately."""
    pbmc = pbmc3k_processed()
    pbmc.X = sparse.csr_matrix(pbmc.X.astype(np.float64))

    ref = pbmc.copy()
    rsc.pp.pca(ref, svd_solver="covariance_eigh", n_comps=10)

    test = pbmc.copy()
    rsc.pp.pca(test, svd_solver="randomized", random_state=0, n_comps=10)

    # Randomized should capture at least 95% of the reference variance
    ref_total = np.sum(ref.uns["pca"]["variance_ratio"])
    test_total = np.sum(test.uns["pca"]["variance_ratio"])
    assert test_total > 0.95 * ref_total

    # First few variance ratios should be reasonably close
    np.testing.assert_allclose(
        test.uns["pca"]["variance_ratio"][:3],
        ref.uns["pca"]["variance_ratio"][:3],
        rtol=0.05,
    )


def test_randomized_kwargs():
    """Test n_oversamples and n_iter kwargs improve accuracy."""
    rng = np.random.RandomState(42)
    X = sparse.random(500, 100, density=0.1, random_state=rng, format="csr")
    X = X.astype(np.float64)

    # Reference
    ref = AnnData(X.copy())
    rsc.pp.pca(ref, svd_solver="covariance_eigh", n_comps=20)

    # Default kwargs
    test_default = AnnData(X.copy())
    rsc.pp.pca(test_default, svd_solver="randomized", n_comps=20, random_state=0)

    # More oversamples and iterations
    test_accurate = AnnData(X.copy())
    rsc.pp.pca(
        test_accurate,
        svd_solver="randomized",
        n_comps=20,
        random_state=0,
        n_oversamples=30,
        n_iter=5,
    )

    # More oversamples/iterations should give better accuracy
    error_default = np.sum(np.abs(test_default.uns["pca"]["variance_ratio"] - ref.uns["pca"]["variance_ratio"]))
    error_accurate = np.sum(np.abs(test_accurate.uns["pca"]["variance_ratio"] - ref.uns["pca"]["variance_ratio"]))
    assert error_accurate <= error_default


# =============================================================================
# SVD solver unit tests (following sklearn/scipy patterns)
# =============================================================================


def _check_svd_results(U, s, Vt, A, k, *, atol=1e-10, rtol=1e-7):
    """Helper to check SVD results for correctness.

    Based on scipy.sparse.linalg tests.
    """
    n, m = A.shape

    # Check shapes
    assert U.shape == (n, k)
    assert s.shape == (k,)
    assert Vt.shape == (k, m)

    # Check that U is semi-orthogonal (U.T @ U = I)
    UtU = U.T @ U
    np.testing.assert_allclose(UtU, np.eye(k), atol=atol, rtol=rtol)

    # Check that Vt is semi-orthogonal (Vt @ Vt.T = I)
    VtV = Vt @ Vt.T
    np.testing.assert_allclose(VtV, np.eye(k), atol=atol, rtol=rtol)

    # Check singular values are positive and sorted descending
    assert np.all(s >= 0)
    assert np.all(s[:-1] >= s[1:])


class TestSVDSolvers:
    """Unit tests for SVD solvers following sklearn/scipy test patterns."""

    @pytest.mark.parametrize("svd_solver", ["lanczos", "randomized"])
    @pytest.mark.parametrize("shape", [(100, 50), (50, 100), (100, 100)])
    def test_svd_shapes(self, svd_solver, shape):
        """Test SVD with different matrix shapes (tall, wide, square)."""
        from rapids_singlecell.preprocessing._sparse_pca._sparse_svd_pca import (
            PCA_sparse_svd,
        )

        rng = np.random.RandomState(42)
        X = sparse.random(*shape, density=0.2, random_state=rng, format="csr")
        X = cusparse.csr_matrix(X.astype(np.float64))

        k = 10
        pca = PCA_sparse_svd(n_components=k, svd_solver=svd_solver, zero_center=True, rng=0)
        pca.fit(X)

        # Check components shape
        assert pca.components_.shape == (k, shape[1])
        assert pca.explained_variance_.shape == (k,)

    @pytest.mark.parametrize("svd_func", [lanczos_svd, randomized_svd], ids=["lanczos", "randomized"])
    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    def test_svd_low_rank_reconstruction(self, svd_func, dtype):
        """Test that SVD can reconstruct a low-rank matrix.

        Following sklearn's test_randomized_svd_low_rank_all_dtypes.
        """
        # Create a low-rank matrix: rank 5 in 100x50 space
        rng = np.random.RandomState(42)
        rank = 5
        U_true = rng.randn(100, rank)
        s_true = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
        Vt_true = rng.randn(rank, 50)
        if dtype == np.complex128:
            U_true = U_true + 1j * rng.randn(*U_true.shape)
            Vt_true = Vt_true + 1j * rng.randn(*Vt_true.shape)
        # Orthogonalize
        U_true, _ = np.linalg.qr(U_true)
        Vt_true, _ = np.linalg.qr(Vt_true.T)
        Vt_true = Vt_true.conj().T

        A = U_true @ np.diag(s_true) @ Vt_true
        A_gpu = cp.asarray(A)

        k = 5
        U, s, Vt = svd_func(A_gpu, k=k, rng=0)

        U = U.get()
        s = s.get()
        Vt = Vt.get()

        # Check singular values match
        np.testing.assert_allclose(s, s_true, rtol=1e-5, atol=1e-6)

        # Check reconstruction
        A_reconstructed = U @ np.diag(s) @ Vt
        np.testing.assert_allclose(A_reconstructed, A, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(A @ Vt.conj().T, U * s, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(A.conj().T @ U, Vt.conj().T * s, rtol=1e-5, atol=1e-6)

    def test_svd_residual_norm(self):
        """Test the fundamental SVD property: Av = σu and A^T u = σv.

        The residual norms ||Av_i - σ_i u_i|| and ||A^T u_i - σ_i v_i||
        should be near zero. This is THE defining property of SVD.

        Note: Only tested for Lanczos (exact method). Randomized SVD is approximate
        and does not satisfy this property - it optimizes reconstruction error,
        not individual singular triplet accuracy.
        """
        rng = np.random.RandomState(42)
        X = sparse.random(100, 50, density=0.2, random_state=rng, format="csr")
        X_gpu = cusparse.csr_matrix(X.astype(np.float64))
        X_dense = X.toarray()

        k = 10
        U, s, Vt = lanczos_svd(X_gpu, k=k, rng=0)

        U = U.get()
        s = s.get()
        Vt = Vt.get()
        V = Vt.T

        atol = 1e-10

        # Check Av_i = σ_i u_i for each singular triplet
        for i in range(k):
            Av = X_dense @ V[:, i]
            sigma_u = s[i] * U[:, i]
            residual = np.linalg.norm(Av - sigma_u)
            assert residual < atol, f"Residual ||Av_{i} - σ_{i}u_{i}|| = {residual:.2e} > {atol:.0e}"

        # Check A^T u_i = σ_i v_i for each singular triplet
        for i in range(k):
            Atu = X_dense.T @ U[:, i]
            sigma_v = s[i] * V[:, i]
            residual = np.linalg.norm(Atu - sigma_v)
            assert residual < atol, f"Residual ||A^T u_{i} - σ_{i}v_{i}|| = {residual:.2e} > {atol:.0e}"

    @pytest.mark.parametrize(
        "svd_func,atol",
        [
            (lanczos_svd, 1e-4),  # Lanczos achieves ~3e-5 orthogonality error
            (randomized_svd, 1e-7),
        ],
        ids=["lanczos", "randomized"],
    )
    def test_svd_orthonormality(self, svd_func, atol):
        """Test that U and V are orthonormal.

        Following scipy's _check_svds.
        """
        rng = np.random.RandomState(42)
        X = sparse.random(200, 100, density=0.1, random_state=rng, format="csr")
        X_gpu = cusparse.csr_matrix(X.astype(np.float64))

        k = 20
        U, s, Vt = svd_func(X_gpu, k=k, rng=0)

        U = U.get()
        s = s.get()
        Vt = Vt.get()

        _check_svd_results(U, s, Vt, X.toarray(), k, atol=atol, rtol=1e-6)

    @pytest.mark.parametrize(
        "svd_func,rtol,atol_excess",
        [
            (lanczos_svd, 1e-6, 1e-6),
            (randomized_svd, 0.05, 0.1),
        ],
        ids=["lanczos", "randomized"],
    )
    def test_svd_vs_numpy(self, svd_func, rtol, atol_excess):
        """Test SVD results match numpy.linalg.svd.

        Following sklearn's test_randomized_svd_low_rank_all_dtypes.
        """
        rng = np.random.RandomState(42)
        A = rng.randn(50, 30).astype(np.float64)
        A_gpu = cp.asarray(A)

        # Reference: numpy SVD
        U_np, s_np, Vt_np = np.linalg.svd(A, full_matrices=False)

        k = 10
        U, s, Vt = svd_func(A_gpu, k=k, rng=0)

        U = U.get()
        s = s.get()
        Vt = Vt.get()

        # Singular values should match
        np.testing.assert_allclose(s, s_np[:k], rtol=rtol)

        # Reconstruction error should be comparable to numpy's optimal rank-k approximation
        # (not comparing reconstructions directly due to sign ambiguity in SVD)
        A_reconstructed = U @ np.diag(s) @ Vt
        A_np_reconstructed = U_np[:, :k] @ np.diag(s_np[:k]) @ Vt_np[:k, :]

        # Both should approximate A with similar accuracy (Frobenius norm)
        error_gpu = np.linalg.norm(A - A_reconstructed)
        error_np = np.linalg.norm(A - A_np_reconstructed)

        # Check reconstruction error is within tolerance of optimal
        relative_excess_error = (error_gpu - error_np) / error_np
        assert relative_excess_error < atol_excess, (
            f"Reconstruction error too high: {error_gpu:.4f} vs optimal {error_np:.4f} "
            f"(excess: {relative_excess_error:.2%})"
        )

    def test_randomized_power_iterations(self):
        """Test that power iterations improve accuracy for noisy matrices.

        Following sklearn's test_randomized_svd_low_rank_with_noise.
        """
        # Create a matrix with noise
        rng = np.random.RandomState(42)
        rank = 5
        U_true = rng.randn(100, rank)
        s_true = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
        Vt_true = rng.randn(rank, 50)
        U_true, _ = np.linalg.qr(U_true)
        Vt_true, _ = np.linalg.qr(Vt_true.T)
        Vt_true = Vt_true.T

        A_clean = U_true @ np.diag(s_true) @ Vt_true
        noise = 0.1 * rng.randn(100, 50)
        A_noisy = A_clean + noise
        A_gpu = cp.asarray(A_noisy)

        k = 5

        # With few iterations, accuracy is worse
        _, s_few, _ = randomized_svd(A_gpu, k=k, n_iter=0, rng=0)
        s_few = s_few.get()

        # With more iterations, accuracy improves
        _, s_many, _ = randomized_svd(A_gpu, k=k, n_iter=5, rng=0)
        s_many = s_many.get()

        # Reference
        _, s_np, _ = np.linalg.svd(A_noisy, full_matrices=False)

        error_few = np.sum(np.abs(s_few - s_np[:k]))
        error_many = np.sum(np.abs(s_many - s_np[:k]))

        # More iterations should give better accuracy
        assert error_many < error_few

    @pytest.mark.parametrize("svd_func", [lanczos_svd, randomized_svd], ids=["lanczos", "randomized"])
    def test_svd_sign_determinism(self, svd_func):
        """Test that SVD results are deterministic (sign-flipped consistently).

        Following sklearn's test_randomized_svd_sign_flip.
        """
        rng = np.random.RandomState(42)
        X = sparse.random(100, 50, density=0.2, random_state=rng, format="csr")
        X_gpu = cusparse.csr_matrix(X.astype(np.float64))

        k = 10

        # Run twice with same random state
        U1, s1, Vt1 = svd_func(X_gpu, k=k, rng=0)
        U2, s2, Vt2 = svd_func(X_gpu, k=k, rng=0)

        for a, b in ((U1, U2), (s1, s2), (Vt1, Vt2)):
            assert a.get().tobytes() == b.get().tobytes()

    def test_mean_centered_operator(self):
        """Test that the mean-centered operator works correctly."""
        from rapids_singlecell.preprocessing._sparse_pca._operators import (
            mean_centered_operator,
        )

        rng = np.random.RandomState(42)
        X = sparse.random(100, 50, density=0.2, random_state=rng, format="csr")
        X_gpu = cusparse.csr_matrix(X.astype(np.float64))

        # Compute mean
        mean = cp.asarray(X.toarray().mean(axis=0))

        # Create operator
        op = mean_centered_operator(X_gpu, mean)

        # Test matvec
        v = cp.ones(50, dtype=np.float64)
        result_op = op.dot(v)

        # Expected: X @ v - mean @ v (broadcast)
        X_dense = cp.asarray(X.toarray())
        expected = X_dense @ v - cp.dot(mean, v)

        np.testing.assert_allclose(result_op.get(), expected.get(), rtol=1e-10)

        # Test matmat
        V = cp.asarray(rng.standard_normal((50, 69)))
        result_op = op.dot(V)
        expected = X_dense @ V - cp.outer(cp.ones(100), mean @ V)

        np.testing.assert_allclose(result_op.get(), expected.get(), rtol=1e-10)

    @pytest.mark.parametrize("width", [1, 35])
    def test_centered_adjoint_long_column(self, width):
        from rapids_singlecell.preprocessing._sparse_pca._operators import mean_centered_operator

        n = 100_000
        X = cusparse.csr_matrix(cp.ones((n, 1), dtype=cp.float32))
        op = mean_centered_operator(X, cp.ones(1, dtype=cp.float32))
        rhs = cp.full((n, width), 1 / np.sqrt(n), dtype=cp.float32)
        result = op.T @ (rhs[:, 0] if width == 1 else rhs)
        atol = 4 * np.spacing(np.float32(np.sqrt(n)))
        np.testing.assert_allclose(result.get(), 0, atol=atol)

    def test_csr_adjoint_int64(self):
        from rapids_singlecell.preprocessing._sparse_pca._csr_adjoint import _CSRAdjoint

        rng = np.random.default_rng(18)
        X = sparse.vstack(
            (
                sparse.random(193, 777, density=0.125, random_state=rng),
                sparse.csr_matrix((71, 777)),
                sparse.random(1, 777, density=0.95, random_state=rng),
            ),
            format="csr",
        )
        X = sparse.hstack((X, sparse.csr_matrix((X.shape[0], 3))), format="csr")
        X_gpu = cusparse.csr_matrix(X)
        X_gpu.indptr = X_gpu.indptr.astype(cp.int64)
        X_gpu.indices = X_gpu.indices.astype(cp.int64)
        adjoint = _CSRAdjoint(X_gpu)

        rhs = rng.standard_normal((X.shape[0], 7))
        for v in (rhs[:, 0], rhs):
            np.testing.assert_allclose(adjoint.dot(cp.asarray(v)).get(), X.T @ v, atol=1e-12)
        mean, centered_ss = adjoint.centered_stats()
        dense = X.toarray()
        np.testing.assert_allclose(mean.get(), dense.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(centered_ss.get(), np.square(dense - dense.mean(axis=0)).sum(), rtol=1e-12)

    @pytest.mark.parametrize(
        ("width", "rhs_dtype", "order"), [(24, np.float32, "C"), (71, np.float32, "C"), (71, np.float64, "F")]
    )
    def test_wide_csr_adjoint(self, width, rhs_dtype, order):
        from rapids_singlecell.preprocessing._sparse_pca._csr_adjoint import _CSRAdjoint

        rng = np.random.default_rng(63)
        X = sparse.random(193, 778, density=0.007, format="csr", random_state=rng, dtype=np.float32)
        X = sparse.hstack((np.ones((193, 1), dtype=np.float32), X), format="csr")
        X_gpu = cusparse.csr_matrix(X)
        X_gpu.indptr = X_gpu.indptr.astype(cp.int64)
        X_gpu.indices = X_gpu.indices.astype(cp.int64)
        adjoint = _CSRAdjoint(X_gpu)

        rhs = rng.standard_normal((193, width)).astype(rhs_dtype, order=order)
        operand = cp.array(rhs, order=order)
        result = adjoint.dot(operand).get()
        expected = X.astype(np.float64).T @ rhs
        np.testing.assert_allclose(result, expected, rtol=4 * np.finfo(rhs_dtype).eps, atol=1e-12)
        assert result.tobytes() == adjoint.dot(operand).get().tobytes()

    @pytest.mark.parametrize("dense_row", [False, True])
    def test_csr_adjoint_shared_peaks(self, dense_row):
        from rapids_singlecell.preprocessing._sparse_pca._csr_adjoint import _CSRAdjoint

        rng = np.random.default_rng(31)
        peaks = 131_073
        X = sparse.random(97, peaks, density=0.009, random_state=rng, format="csr")
        if dense_row:
            # A row offset can exceed uint16 even when most cells have few peaks.
            X = sparse.vstack((X, np.ones((1, peaks))), format="csr")
        # Shared peaks exceed the packed prefix limit in the first tile.
        X = sparse.hstack((np.ones((X.shape[0], 64)), X), format="csr")
        X.sort_indices()
        adjoint = _CSRAdjoint(cusparse.csr_matrix(X))

        rhs = rng.standard_normal((X.shape[0], 7))
        for v in (rhs[:, 0], rhs):
            operand = cp.asarray(v)
            result = adjoint.dot(operand).get()
            np.testing.assert_allclose(result, X.T @ v, rtol=1e-12, atol=1e-12)
            assert result.tobytes() == adjoint.dot(operand).get().tobytes()
        mean, centered_ss = adjoint.centered_stats()
        expected_mean = np.asarray(X.mean(axis=0)).ravel()
        np.testing.assert_allclose(mean.get(), expected_mean, atol=1e-12)
        np.testing.assert_allclose(centered_ss.get(), X.power(2).sum() - X.shape[0] * (expected_mean @ expected_mean))

    @pytest.mark.parametrize("svd_func", [lanczos_svd, randomized_svd], ids=["lanczos", "randomized"])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_svd_dtype_preservation(self, svd_func, dtype):
        """Test that SVD preserves input dtype."""
        rng = np.random.RandomState(42)
        X = sparse.random(100, 50, density=0.2, random_state=rng, format="csr")
        X_gpu = cusparse.csr_matrix(X.astype(dtype))

        k = 10
        U, s, Vt = svd_func(X_gpu, k=k, rng=0)

        assert U.dtype == dtype
        assert s.dtype == dtype
        assert Vt.dtype == dtype

    def test_svd_sparse_efficiency(self):
        """Test that sparse SVD doesn't densify the matrix."""
        rng = np.random.RandomState(42)
        # Create a large sparse matrix
        X = sparse.random(1000, 500, density=0.01, random_state=rng, format="csr")
        X_gpu = cusparse.csr_matrix(X.astype(np.float64))

        # Memory for dense would be ~4GB, sparse is ~80MB
        # This should complete quickly without OOM
        k = 10
        U, s, Vt = lanczos_svd(X_gpu, k=k, rng=0)

        assert U.shape == (1000, k)
        assert s.shape == (k,)
        assert Vt.shape == (k, 500)

    @pytest.mark.parametrize("svd_func", [lanczos_svd, randomized_svd], ids=["lanczos", "randomized"])
    def test_svd_small_k(self, svd_func):
        """Test SVD with small k values."""
        rng = np.random.RandomState(42)
        X = sparse.random(100, 50, density=0.2, random_state=rng, format="csr")
        X_gpu = cusparse.csr_matrix(X.astype(np.float64))

        for k in [1, 2, 3]:
            U, s, Vt = svd_func(X_gpu, k=k, rng=0)

            assert U.shape == (100, k)
            assert s.shape == (k,)
            assert Vt.shape == (k, 50)

            # Check orthonormality
            U_np = U.get()
            Vt_np = Vt.get()
            np.testing.assert_allclose(U_np.T @ U_np, np.eye(k), atol=1e-6)
            np.testing.assert_allclose(Vt_np @ Vt_np.T, np.eye(k), atol=1e-6)


# =============================================================================
# Multi-vector locking restart path
# =============================================================================


def _rotated_block_matrix(m, n, spectrum):
    """Block-diagonal matrix of 2x2 rotated blocks with an exact prescribed spectrum.

    Block ``b`` is ``U(theta_b) diag(s_2b, s_2b+1) V(phi_b)^T`` with ``U``/``V`` plane
    rotations, so the singular values are exactly ``spectrum`` and the right singular
    vectors are exactly the columns of ``V(phi_b)`` -- an analytic reference, no dense
    solve needed. Kept deliberately simple so the identical construction can be used by
    the C++ test in RAFT (``LanczosHardening*.MultiLockRestart``).
    """
    rows, cols, vals = [], [], []
    for b in range(n // 2):
        s0, s1 = spectrum[2 * b], spectrum[2 * b + 1]
        theta, phi = 0.17 + 0.11 * b, 0.31 + 0.07 * b
        cu, su = np.cos(theta), np.sin(theta)
        cv, sv = np.cos(phi), np.sin(phi)
        for r, c, v in [
            (2 * b, 2 * b, cu * s0 * cv + su * s1 * sv),
            (2 * b, 2 * b + 1, cu * s0 * sv - su * s1 * cv),
            (2 * b + 1, 2 * b, su * s0 * cv - cu * s1 * sv),
            (2 * b + 1, 2 * b + 1, su * s0 * sv + cu * s1 * cv),
        ]:
            rows.append(r)
            cols.append(c)
            vals.append(v)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(m, n))


def _rotated_block_reference_V(n, k):
    """Analytic right singular vectors of :func:`_rotated_block_matrix`, shape (n, k)."""
    V = np.zeros((n, k))
    for b in range(n // 2):
        phi = 0.31 + 0.07 * b
        cv, sv = np.cos(phi), np.sin(phi)
        if 2 * b < k:
            V[2 * b, 2 * b], V[2 * b + 1, 2 * b] = cv, sv
        if 2 * b + 1 < k:
            V[2 * b, 2 * b + 1], V[2 * b + 1, 2 * b + 1] = -sv, cv
    return V


def _largest_principal_angle(V_ref, V):
    """Largest principal angle between two orthonormal column blocks, in radians.

    Used instead of a column-wise comparison because with clustered singular values the
    individual singular vectors are not unique, but their span is.
    """
    k = min(V_ref.shape[1], V.shape[1])
    s = np.linalg.svd(V_ref[:, :k].T @ V[:, :k], compute_uv=False)
    return float(np.arccos(np.clip(s.min(), -1.0, 1.0)))


@pytest.mark.parametrize("separated", [False, True])
def test_lanczos_restart_checkpoint(*, separated):
    from rapids_singlecell.preprocessing._sparse_pca._svd_lanczos import _lanczos_bidiag

    n, ncv = 144, 80
    values = cp.linspace(1, 0.01, n, dtype=cp.float32)
    values[:2] = cp.asarray([20, 19])
    if separated:
        values[2:6] = cp.asarray([8, 6, 4, 2])

    # Start a restart with the two largest singular vectors already locked.
    U = cp.zeros((n, ncv + 3), dtype=values.dtype, order="F")
    U[:, :2] = cp.eye(n, 2, dtype=values.dtype)
    rng = cp.random.default_rng(2)
    a, b = _lanczos_bidiag(
        cp.diag(values),
        ncv=ncv,
        v_start=rng.standard_normal(n).astype(values.dtype),
        U_full=U,
        V_full=U.copy(order="F"),
        n_locked=2,
        rng=rng,
        ortho_buf=cp.empty(ncv + 3, dtype=values.dtype),
        remaining=2,
    )
    assert (a.size < ncv) == separated
    projected = np.diag(a.get()) + np.diag(b[1:-1].get(), 1)
    np.testing.assert_allclose(np.linalg.svd(projected, compute_uv=False)[:2], values[2:4].get(), rtol=1e-5)


def test_lanczos_restart_convergence_guard():
    from rapids_singlecell.preprocessing._sparse_pca._svd_lanczos import _restart_converged

    alphas = cp.linspace(8, 1, 48, dtype=cp.float32)
    betas = cp.full(49, 1e-12, dtype=cp.float32)
    betas[-1] = 1
    tied = alphas.copy()
    tied[2] = tied[1]
    assert not _restart_converged(tied, betas, 2)
    betas[20] = 0
    assert not _restart_converged(alphas, betas, 2)


@pytest.mark.parametrize(
    "dtype,tol,sv_rtol,angle_tol,resid_tol,orth_tol",
    [
        (np.float64, 1e-10, 1e-12, 1e-6, 1e-9, 1e-8),
        (np.float32, 1e-5, 1e-4, 1e-2, 1e-4, 1e-3),
    ],
    ids=["float64", "float32"],
)
def test_lanczos_multilock_restart(dtype, *, tol, sv_rtol, angle_tol, resid_tol, orth_tol, monkeypatch):
    """Check the leading unlocked Ritz vector seeds the next sweep after multi-vector locking."""
    from rapids_singlecell.preprocessing._sparse_pca import _svd_lanczos

    m, n, k = 96, 64, 20
    ncv = 32
    assert ncv < min(m, n)

    spectrum = 20.0 * 0.97 ** np.arange(n)
    A_host = _rotated_block_matrix(m, n, spectrum)
    A_gpu = cusparse.csr_matrix(A_host.astype(dtype))

    # `_lanczos_bidiag` is called once per sweep with the *pre-lock* n_locked, so the
    # diffs of that sequence are exactly the per-restart d (num_found).
    n_locked_seq = []
    ritz = None
    original = _svd_lanczos._lanczos_bidiag

    def recording_bidiag(*args, **kwargs):
        nonlocal ritz
        start = kwargs["n_locked"]
        basis = kwargs["V_full"]
        if n_locked_seq and start > n_locked_seq[-1]:
            locked = basis[:, n_locked_seq[-1] : start].get()
            locked_indices = np.abs(ritz @ locked).argmax(axis=0)
            best = next(i for i in range(ncv) if i not in locked_indices)
            alignment = np.abs(ritz[best] @ kwargs["v_start"].get())
            np.testing.assert_allclose(alignment, 1, atol=orth_tol, rtol=0)
        n_locked_seq.append(start)
        a, b = original(*args, **kwargs)
        projected = np.diag(a.get()) + np.diag(b[1:-1].get(), 1)
        _, _, vt = np.linalg.svd(projected)
        ritz = vt @ basis[:, start : start + a.size].get().T
        return a, b

    monkeypatch.setattr(_svd_lanczos, "_lanczos_bidiag", recording_bidiag)

    U, s, Vt = _svd_lanczos.lanczos_svd(A_gpu, k=k, ncv=ncv, tol=tol, max_iter=300, rng=0)

    # --- guard: the test must actually exercise multi-vector locking ------------
    d_per_restart = np.diff(np.asarray(n_locked_seq))
    assert len(n_locked_seq) >= 3, (
        f"expected at least 2 restarts, saw {len(n_locked_seq)} sweep(s); "
        "this configuration no longer exercises the restart path"
    )
    assert d_per_restart.max() >= 2, (
        f"expected at least one restart with d >= 2, saw d={d_per_restart.tolist()}; "
        "this configuration no longer exercises multi-vector locking"
    )

    U_h = cp.asnumpy(U).astype(np.float64)
    s_h = cp.asnumpy(s).astype(np.float64)
    V_h = cp.asnumpy(Vt).astype(np.float64).T

    # --- singular values against the known spectrum -----------------------------
    np.testing.assert_allclose(s_h, spectrum[:k], rtol=sv_rtol)

    # --- largest principal angle against the analytic reference V block ---------
    angle = _largest_principal_angle(_rotated_block_reference_V(n, k), V_h)
    assert angle < angle_tol, f"largest principal angle {angle:.3e} >= {angle_tol:.0e}"

    # --- right residual ||A^T u_i - s_i v_i|| / s_0 ------------------------------
    # The left residual ||A v_i - s_i u_i|| is identically zero after the refinement
    # step (U = A V; S = colnorms; U /= S) and carries no information, so it is not
    # asserted here.
    A_dense = A_host.toarray()
    resid = np.linalg.norm(A_dense.T @ U_h - V_h * s_h[None, :], axis=0) / spectrum[0]
    assert resid.max() < resid_tol, f"max right residual {resid.max():.3e}"

    # --- orthonormality ----------------------------------------------------------
    np.testing.assert_allclose(U_h.T @ U_h, np.eye(k), atol=orth_tol)
    np.testing.assert_allclose(V_h.T @ V_h, np.eye(k), atol=orth_tol)
