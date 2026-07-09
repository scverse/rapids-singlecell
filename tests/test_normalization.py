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


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_normalize_total_promotes_dense_integers(dtype):
    cudata = AnnData(cp.array([[1, 1], [2, 4]], dtype=dtype))

    rsc.pp.normalize_total(cudata, target_sum=10)

    assert cudata.X.dtype == cp.float32
    cp.testing.assert_allclose(
        cudata.X.sum(axis=1), cp.full(cudata.n_obs, 10, dtype=cp.float32)
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


@pytest.mark.parametrize("use_array", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("base", [None, 2, 10])
def test_log1p_base(use_array, dtype, sparse, base):
    X = cp.array([[1.0, 2.0], [3.0, 4.0], [0.0, 5.0]], dtype=dtype)
    if sparse:
        X = csr_matrix(X)
    cudata = AnnData(X.copy())

    if use_array:
        # feeding the matrix directly should match feeding the AnnData
        out = rsc.pp.log1p(X.copy(), base=base)
        result = out.toarray() if hasattr(out, "toarray") else out
    else:
        rsc.pp.log1p(cudata, base=base)
        result = cudata.X.toarray() if sparse else cudata.X

    # Compute reference on the host (CPU) to validate the GPU result
    X_ref = np.log1p(np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 5.0]], dtype=dtype))
    if base is not None:
        X_ref /= np.log(base)

    cp.testing.assert_allclose(result, X_ref, rtol=1e-5)
    if not use_array:
        assert cudata.uns["log1p"]["base"] == base


def test_log1p_inplace_false_does_not_write_metadata():
    X = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
    adata = AnnData(X.copy())

    result = rsc.pp.log1p(adata, inplace=False)

    cp.testing.assert_array_equal(adata.X, X)
    cp.testing.assert_allclose(result, cp.log1p(X))
    assert "log1p" not in adata.uns


@pytest.mark.parametrize("use_array", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse", [True, False])
def test_sqrt(use_array, dtype, sparse):
    X = cp.array([[1.0, 4.0], [9.0, 16.0], [0.0, 25.0]], dtype=dtype)
    if sparse:
        X = csr_matrix(X)
    cudata = AnnData(X.copy())

    if use_array:
        # feeding the matrix directly should match feeding the AnnData
        out = rsc.pp.sqrt(X.copy())
        result = out.toarray() if hasattr(out, "toarray") else out
    else:
        rsc.pp.sqrt(cudata)
        result = cudata.X.toarray() if sparse else cudata.X

    X_ref = np.sqrt(np.array([[1.0, 4.0], [9.0, 16.0], [0.0, 25.0]], dtype=dtype))
    cp.testing.assert_allclose(result, X_ref, rtol=1e-5)


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


# ------------------------------------------------------------------------------
# normalize_clr (shifted CLR / PFlog1pPF)
# ------------------------------------------------------------------------------

# A small count matrix with no empty cells, used for the value/equivalence tests.
X_clr = np.array(
    [[5, 0, 3, 2], [1, 1, 0, 4], [0, 7, 2, 1], [3, 3, 3, 3]], dtype="float32"
)

CLR_ARRAY_TYPES = [cp.array, csr_matrix, csc_matrix]


def _to_np(x):
    """cupy dense / cupy sparse / numpy -> dense numpy."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    return cp.asnumpy(x)


def _estimate_alpha_reference(x) -> float:
    """Closed-form OLS overdispersion, independent of the implementation."""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=0)
    var = (x**2).mean(axis=0) - mu**2
    mu2 = mu**2
    return float(np.sum((var - mu) * mu2) / np.sum(mu2 * mu2))


def _clr_reference(x, *, target_sum=None, alpha=None) -> np.ndarray:
    """Self-contained dense shifted-CLR, independent of the implementation.

    PF to a target depth, log1p, then subtract the per-cell mean. Empty cells
    (zero depth) are left as all-zero rows, matching `normalize_clr`.
    """
    x = np.asarray(x, dtype=np.float64)
    depths = x.sum(axis=1)
    if alpha is not None:
        if alpha == "auto":
            alpha = _estimate_alpha_reference(x)
        target_sum = 4.0 * alpha * depths.mean()
    elif target_sum is None:
        target_sum = depths.mean()
    safe_depths = np.where(depths == 0, 1.0, depths)
    u = x * (target_sum / safe_depths)[:, None]
    log_u = np.log1p(u)
    return log_u - log_u.mean(axis=1, keepdims=True)


def _reconstruct_clr(X, residuals) -> np.ndarray:
    """rsc keeps X = log1p(PF) sparse and the per-cell centering offset apart.

    The centered CLR is `X - offset[:, None]`; only the test materializes it.
    """
    return _to_np(X) - cp.asnumpy(residuals).reshape(-1, 1)


def _clr_result(adata) -> np.ndarray:
    """Centered CLR from the factored output written to `layers["clr"]`."""
    return _reconstruct_clr(adata.layers["clr"], adata.obsm["clr_residuals"])


@pytest.mark.parametrize("array_type", CLR_ARRAY_TYPES, ids=lambda f: f.__name__)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_normalize_clr_values(array_type, dtype):
    """Reconstructed CLR matches the reference and every cell sums to zero.

    Asserting that dense / csr / csc inputs all equal the same dense reference
    also proves the sparse "offset trick" matches the dense path.
    """
    adata = AnnData(array_type(cp.asarray(X_clr).astype(dtype)))
    x_before = _to_np(adata.X).copy()
    rsc.pp.normalize_clr(adata)

    result = _clr_result(adata)
    np.testing.assert_allclose(result, _clr_reference(X_clr), rtol=1e-5, atol=1e-5)
    # zero-sum (Aitchison) hyperplane
    np.testing.assert_allclose(result.sum(axis=1), 0.0, atol=1e-5)
    # source matrix is left untouched (result goes to layers["clr"])
    np.testing.assert_array_equal(_to_np(adata.X), x_before)


@pytest.mark.parametrize("array_type", CLR_ARRAY_TYPES, ids=lambda f: f.__name__)
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"target_sum": 1e4}, {"alpha": 0.5}, {"alpha": "auto"}],
    ids=["default", "target_sum", "alpha", "alpha_auto"],
)
def test_normalize_clr_params(array_type, kwargs):
    adata = AnnData(array_type(cp.asarray(X_clr)))
    rsc.pp.normalize_clr(adata, **kwargs)
    np.testing.assert_allclose(
        _clr_result(adata),
        _clr_reference(X_clr, **kwargs),
        rtol=1e-5,
        atol=1e-5,
    )


def test_normalize_clr_alpha_overrides_target_sum():
    """`alpha` sets target_sum = 4*alpha*scale and overrides any given `target_sum`."""
    alpha = 0.5
    scale = X_clr.sum(axis=1).mean()

    via_alpha = AnnData(csr_matrix(cp.asarray(X_clr)))
    rsc.pp.normalize_clr(via_alpha, alpha=alpha)

    via_target = AnnData(csr_matrix(cp.asarray(X_clr)))
    rsc.pp.normalize_clr(via_target, target_sum=4.0 * alpha * scale)
    np.testing.assert_allclose(
        _clr_result(via_alpha),
        _clr_result(via_target),
        rtol=1e-5,
        atol=1e-5,
    )

    # passing both -> alpha wins, target_sum ignored
    both = AnnData(csr_matrix(cp.asarray(X_clr)))
    rsc.pp.normalize_clr(both, alpha=alpha, target_sum=999.0)
    np.testing.assert_allclose(
        _clr_result(both),
        _clr_result(via_alpha),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("array_type", CLR_ARRAY_TYPES, ids=lambda f: f.__name__)
def test_normalize_clr_alpha_auto(array_type):
    """`alpha="auto"` estimates the overdispersion and matches an explicit alpha."""
    estimated = _estimate_alpha_reference(X_clr)
    assert estimated > 0

    auto = AnnData(array_type(cp.asarray(X_clr)))
    rsc.pp.normalize_clr(auto, alpha="auto")

    explicit = AnnData(array_type(cp.asarray(X_clr)))
    rsc.pp.normalize_clr(explicit, alpha=estimated)
    np.testing.assert_allclose(
        _clr_result(auto),
        _clr_result(explicit),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("alpha", [0.0, -0.5], ids=["zero", "negative"])
def test_normalize_clr_nonpositive_alpha_raises(alpha):
    """A non-positive `alpha` cannot derive K = 4*alpha*s and raises."""
    adata = AnnData(csr_matrix(cp.asarray(X_clr)))
    with pytest.raises(ValueError, match=r"alpha.*positive"):
        rsc.pp.normalize_clr(adata, alpha=alpha)


def test_normalize_clr_alpha_auto_zero_mean_raises():
    """`alpha="auto"` cannot estimate overdispersion when every gene mean is zero."""
    adata = AnnData(cp.zeros((3, 4), dtype=cp.float32))
    with pytest.raises(ValueError, match="Cannot estimate overdispersion"):
        rsc.pp.normalize_clr(adata, alpha="auto")


@pytest.mark.parametrize("array_type", CLR_ARRAY_TYPES, ids=lambda f: f.__name__)
def test_normalize_clr_zero_cell(array_type):
    """An empty cell stays all-zero, stays finite, and triggers a warning."""
    x = X_clr.copy()
    x[1] = 0  # make the second cell empty
    adata = AnnData(array_type(cp.asarray(x)))
    with pytest.warns(UserWarning, match="zero counts"):
        rsc.pp.normalize_clr(adata)
    result = _clr_result(adata)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[1], 0.0, atol=1e-6)


def test_normalize_clr_inplace_false():
    adata = AnnData(csr_matrix(cp.asarray(X_clr)))
    x_before = _to_np(adata.X).copy()
    out = rsc.pp.normalize_clr(adata, inplace=False)

    # factored design: inplace=False returns (X, cell_depths, residuals)
    X, _cell_depths, residuals = out
    np.testing.assert_allclose(
        _reconstruct_clr(X, residuals), _clr_reference(X_clr), rtol=1e-5, atol=1e-5
    )
    # input is left untouched and nothing is written to the object
    np.testing.assert_array_equal(_to_np(adata.X), x_before)
    assert "clr" not in adata.layers
    assert "clr_residuals" not in adata.obsm


def test_normalize_clr_copy():
    adata = AnnData(csr_matrix(cp.asarray(X_clr)))
    x_before = _to_np(adata.X).copy()
    returned = rsc.pp.normalize_clr(adata, copy=True)

    assert isinstance(returned, AnnData)
    assert returned is not adata
    np.testing.assert_allclose(
        _clr_result(returned),
        _clr_reference(X_clr),
        rtol=1e-5,
        atol=1e-5,
    )
    # source matrix on the copy is preserved; original object untouched
    np.testing.assert_array_equal(_to_np(returned.X), x_before)
    assert "clr" not in adata.layers


def test_normalize_clr_copy_inplace_error():
    adata = AnnData(csr_matrix(cp.asarray(X_clr)))
    with pytest.raises(
        ValueError, match="`copy=True` cannot be used with `inplace=False`"
    ):
        rsc.pp.normalize_clr(adata, copy=True, inplace=False)


def test_normalize_clr_layer():
    """`layer` selects the input; output always goes to layers["clr"], sources kept."""
    adata = AnnData(
        csr_matrix(cp.asarray(X_clr)),
        layers={"counts": csr_matrix(cp.asarray(X_clr))},
    )
    x_before = _to_np(adata.X).copy()
    counts_before = _to_np(adata.layers["counts"]).copy()
    rsc.pp.normalize_clr(adata, layer="counts")

    # both X and the source layer are untouched; result lands in layers["clr"]
    np.testing.assert_array_equal(_to_np(adata.X), x_before)
    np.testing.assert_array_equal(_to_np(adata.layers["counts"]), counts_before)
    np.testing.assert_allclose(
        _clr_result(adata),
        _clr_reference(X_clr),
        rtol=1e-5,
        atol=1e-5,
    )
