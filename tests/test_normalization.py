from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from cupyx.scipy.sparse import csc_matrix, csr_matrix

import rapids_singlecell as rsc

X_total = cp.array([[1, 0], [3, 0], [5, 6]], dtype=np.float64)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse", [True, False])
def test_normalize_total(dtype, sparse):
    if sparse:
        X = csr_matrix(X_total, dtype=dtype)
    else:
        X = X_total.copy().astype(dtype)
    cudata = AnnData(X)

    rsc.pp.normalize_total(cudata, target_sum=1)
    cp.testing.assert_allclose(
        cp.ravel(cudata.X.sum(axis=1)), np.ones(cudata.shape[0], dtype=dtype)
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_normalize_total_layers(dtype):
    cudata = AnnData(csr_matrix(X_total, dtype=dtype))
    cudata.layers["layer"] = cudata.X.copy()

    rsc.pp.normalize_total(cudata, target_sum=1, layer="layer")
    assert np.allclose(
        cudata.layers["layer"].sum(axis=1), np.ones(cudata.shape[0], dtype=dtype)
    )


@pytest.mark.parametrize(
    "sparsity_func", [cp.array, csr_matrix, csc_matrix], ids=lambda x: x.__name__
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("theta", [0.01, 1.0, 100, np.inf])
@pytest.mark.parametrize("clip", [None, 1.0, np.inf])
def test_normalize_pearson_residuals_values(sparsity_func, dtype, theta, clip):
    # toy data
    X = cp.array([[3, 6], [2, 4], [1, 0]], dtype=dtype)
    ns = cp.sum(X, axis=1)
    ps = cp.sum(X, axis=0) / cp.sum(X)
    mu = cp.outer(ns, ps)

    # compute reference residuals
    if np.isinf(theta):
        # Poisson case
        residuals_reference = (X - mu) / cp.sqrt(mu)
    else:
        # NB case
        residuals_reference = (X - mu) / cp.sqrt(mu + mu**2 / theta)

    # compute output to test
    cudata = AnnData(X=sparsity_func(X, dtype=dtype))
    output_X = rsc.pp.normalize_pearson_residuals(
        cudata, theta=theta, clip=clip, inplace=False
    )

    rsc.pp.normalize_pearson_residuals(cudata, theta=theta, clip=clip, inplace=True)
    assert np.all(np.isin(["pearson_residuals_normalization"], list(cudata.uns.keys())))
    assert np.all(
        np.isin(
            ["theta", "clip", "computed_on"],
            list(cudata.uns["pearson_residuals_normalization"].keys()),
        )
    )

    # test against inplace
    cp.testing.assert_array_equal(cudata.X, output_X)
    if clip is None:
        # default clipping: compare to sqrt(n) threshold
        clipping_threshold = np.sqrt(cudata.shape[0]).astype(dtype)
        assert np.max(output_X) <= clipping_threshold
        assert np.min(output_X) >= -clipping_threshold
    elif np.isinf(clip):
        # no clipping: compare to raw residuals
        assert np.allclose(output_X, residuals_reference)
    else:
        # custom clipping: compare to custom threshold
        assert np.max(output_X) <= clip
        assert np.min(output_X) >= -clip


@pytest.mark.parametrize(
    "sparsity_func", [csr_matrix, csc_matrix], ids=lambda x: x.__name__
)
@pytest.mark.parametrize("theta", [100.0, np.inf])
def test_normalize_pearson_residuals_float64_precision(sparsity_func, theta):
    """Regression test: float64 precision of the sparse Pearson-residual kernels.

    ``sparse_norm_res_csr_kernel`` / ``sparse_norm_res_csc_kernel`` (in
    ``_cuda/pr/kernels_pr.cuh``) previously divided by the single-precision
    intrinsic ``sqrtf``. Because the kernels are templated on the element
    type, a ``float64`` instantiation silently narrowed the variance term
    to ``float32``, capping accuracy at ~7 significant digits regardless of
    the requested dtype. The ``rtol``/``atol`` of 1e-9 below is tight enough
    to fail on a single-precision result and pass on a genuine float64 one.
    """
    rng = np.random.default_rng(0)
    counts = rng.poisson(0.3, size=(300, 200)).astype(np.float64)
    # ensure every gene and cell has a nonzero total so mu > 0 everywhere
    counts[0, :] += 1
    counts[:, 0] += 1
    X = cp.asarray(counts)

    # analytic float64 reference residuals (no clipping)
    ns = cp.sum(X, axis=1)
    ps = cp.sum(X, axis=0) / cp.sum(X)
    mu = cp.outer(ns, ps)
    if np.isinf(theta):
        reference = (X - mu) / cp.sqrt(mu)
    else:
        reference = (X - mu) / cp.sqrt(mu + mu**2 / theta)

    cudata = AnnData(X=sparsity_func(X, dtype=np.float64))
    output = rsc.pp.normalize_pearson_residuals(
        cudata, theta=theta, clip=np.inf, inplace=False
    )

    # the buggy `sqrtf` path is only ~1e-7 accurate; 1e-9 cleanly separates it
    cp.testing.assert_allclose(output, reference, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("base", [None, 2, 10])
def test_log1p_base(dtype, sparse, base):
    X = cp.array([[1.0, 2.0], [3.0, 4.0], [0.0, 5.0]], dtype=dtype)
    if sparse:
        X = csr_matrix(X)
    cudata = AnnData(X.copy())

    rsc.pp.log1p(cudata, base=base)

    # Compute reference
    X_ref = cp.array([[1.0, 2.0], [3.0, 4.0], [0.0, 5.0]], dtype=dtype)
    X_ref = cp.log1p(X_ref)
    if base is not None:
        X_ref /= cp.log(base)

    if sparse:
        result = cudata.X.toarray()
    else:
        result = cudata.X

    cp.testing.assert_allclose(result, X_ref, rtol=1e-5)
    assert cudata.uns["log1p"]["base"] == base


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse", [True, False])
def test_normalize_total_exclude_highly_expressed(dtype, sparse):
    """Cross-validate against scanpy's normalize_total with exclude_highly_expressed."""
    from scanpy.datasets import pbmc3k

    adata = pbmc3k()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)

    # scanpy reference
    adata_sc = adata.copy()
    adata_sc.X = adata_sc.X.astype(dtype)
    sc.pp.normalize_total(adata_sc, exclude_highly_expressed=True, max_fraction=0.05)

    # rapids_singlecell
    adata_rsc = adata.copy()
    if sparse:
        adata_rsc.X = csr_matrix(adata_rsc.X.astype(dtype))
    else:
        adata_rsc.X = cp.array(adata_rsc.X.toarray(), dtype=dtype)
    rsc.pp.normalize_total(adata_rsc, exclude_highly_expressed=True, max_fraction=0.05)

    if sparse:
        result = cp.asnumpy(adata_rsc.X.toarray())
    else:
        result = cp.asnumpy(adata_rsc.X)

    if sparse:
        expected = adata_sc.X.toarray()
    else:
        expected = adata_sc.X.toarray()

    np.testing.assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.parametrize("sparse", [True, False])
def test_normalize_total_exclude_none_highly_expressed(sparse):
    """When no genes are highly expressed, result matches normal normalize_total."""
    # Use data where no gene dominates any cell
    X = cp.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]], dtype=np.float64)

    if sparse:
        X1 = csr_matrix(X.copy())
        X2 = csr_matrix(X.copy())
    else:
        X1 = X.copy()
        X2 = X.copy()

    adata1 = AnnData(X1)
    adata2 = AnnData(X2)

    # Normal normalize
    rsc.pp.normalize_total(adata1, target_sum=1e4)
    # With exclude_highly_expressed but max_fraction high enough that nothing is excluded
    rsc.pp.normalize_total(
        adata2, target_sum=1e4, exclude_highly_expressed=True, max_fraction=0.99
    )

    if sparse:
        r1 = adata1.X.toarray()
        r2 = adata2.X.toarray()
    else:
        r1 = adata1.X
        r2 = adata2.X

    cp.testing.assert_allclose(r1, r2)


def test_normalize_total_max_fraction_validation():
    """Invalid max_fraction raises ValueError."""
    X = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    cudata = AnnData(X)

    with pytest.raises(ValueError, match="`max_fraction` must be between 0 and 1"):
        rsc.pp.normalize_total(cudata, exclude_highly_expressed=True, max_fraction=0.0)

    with pytest.raises(ValueError, match="`max_fraction` must be between 0 and 1"):
        rsc.pp.normalize_total(cudata, exclude_highly_expressed=True, max_fraction=1.0)

    with pytest.raises(ValueError, match="`max_fraction` must be between 0 and 1"):
        rsc.pp.normalize_total(cudata, exclude_highly_expressed=True, max_fraction=-0.1)
