from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from cupyx.scipy.sparse import csc_matrix, csr_matrix

import rapids_singlecell as rsc
from rapids_singlecell.preprocessing._normalize import _estimate_nb_overdispersion

ARRAY_TYPES = [cp.asarray, csr_matrix, csc_matrix]

# TODO: drop this skip once a scanpy release ships `normalize_shifted_clr` (scverse/scanpy#4160);
# until then only the `dev` CI env, which installs scanpy main, compares against scanpy.
needs_scanpy_shifted_clr = pytest.mark.skipif(
    not hasattr(sc.pp, "normalize_shifted_clr"),
    reason="installed scanpy has no `normalize_shifted_clr` (scverse/scanpy#4160)",
)


def _counts() -> np.ndarray:
    """RNA-like overdispersed counts with an empty cell and an empty gene."""
    rng = np.random.default_rng(0)
    mu = rng.lognormal(0.5, 1.2, size=30)
    x = rng.negative_binomial(2, 2 / (2 + mu), size=(40, 30)).astype(np.float64)
    x[3] = 0
    x[:, 7] = 0
    return x


def _scanpy_shifted_clr(x: np.ndarray, **kwargs) -> np.ndarray:
    adata = AnnData(x.copy())
    with pytest.warns(UserWarning, match="Some cells have zero counts"):
        sc.pp.normalize_shifted_clr(adata, **kwargs)
    return np.asarray(adata.X)


# Same hand-computed cases as scanpy's tests
X_clr = np.array([[5, 0, 3, 2], [1, 1, 0, 4], [0, 7, 2, 1], [3, 3, 3, 3]], dtype="float32")
# Each gene has mean 2 and variance 4, so alpha = (4 - 2) / 2**2 = 0.5;
# log1p(4 * 0.5 * [0, 4]) = [0, ln 9], centered on its mean ln 3.
X_clr_estimated = np.array([[0, 4], [4, 0]], dtype="float32")
X_clr_estimated_expected = np.log(3) * np.array([[-1.0, 1.0], [1.0, -1.0]])
# alpha = 0.25: log1p([0, 1, 3]) = [0, ln 2, 2 ln 2], centered on its mean ln 2.
X_clr_explicit = np.array([[0, 1, 3]], dtype="float32")
X_clr_explicit_expected = np.log(2) * np.array([[-1.0, 0.0, 1.0]])


def _dense(X):
    return cp.asnumpy(X.toarray() if hasattr(X, "toarray") else X)


def _clr(x, array_type=cp.asarray, **kwargs) -> np.ndarray:
    adata = AnnData(array_type(cp.asarray(x, dtype=cp.float32)))
    rsc.pp.normalize_shifted_clr(adata, **kwargs)
    assert isinstance(adata.X, cp.ndarray)
    return cp.asnumpy(adata.X)


@needs_scanpy_shifted_clr
@pytest.mark.parametrize("array_type", ARRAY_TYPES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("alpha", [None, 0.3], ids=["estimated", "explicit"])
def test_matches_scanpy(array_type, dtype, alpha):
    x = _counts().astype(dtype)
    expected = _scanpy_shifted_clr(x, alpha=alpha)
    adata = AnnData(array_type(cp.asarray(x)))
    with pytest.warns(UserWarning, match="Some cells have zero counts"):
        rsc.pp.normalize_shifted_clr(adata, alpha=alpha)

    assert isinstance(adata.X, cp.ndarray)
    assert adata.X.dtype == dtype
    tol = 1e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(cp.asnumpy(adata.X), expected, rtol=tol, atol=tol)


@needs_scanpy_shifted_clr
def test_estimate_alpha_matches_scanpy():
    from scanpy.preprocessing._normalization import _estimate_nb_overdispersion as sc_estimate

    x = _counts()
    alpha = _estimate_nb_overdispersion(csr_matrix(cp.asarray(x)))
    assert alpha == pytest.approx(sc_estimate(x), rel=1e-12)


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
def test_values(array_type):
    """Check values with the estimated `alpha` against hand-computed results."""
    np.testing.assert_allclose(_clr(X_clr_estimated, array_type), X_clr_estimated_expected, rtol=1e-5, atol=1e-6)


def test_values_integer_input():
    adata = AnnData(cp.asarray(X_clr_estimated, dtype=cp.int64))
    rsc.pp.normalize_shifted_clr(adata)
    assert adata.X.dtype == cp.float64
    np.testing.assert_allclose(cp.asnumpy(adata.X), X_clr_estimated_expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
def test_explicit_alpha(array_type):
    """Check values with an explicit `alpha` against hand-computed results."""
    np.testing.assert_allclose(
        _clr(X_clr_explicit, array_type, alpha=0.25), X_clr_explicit_expected, rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("alpha", [None, 0.5], ids=["estimated", "explicit"])
def test_zero_sum(alpha):
    """Every cell lies on the zero-sum (Aitchison) hyperplane."""
    result = _clr(X_clr, alpha=alpha)
    np.testing.assert_allclose(result.sum(axis=1), 0.0, atol=1e-5)
    # a cell with equal counts for every gene maps to all zeros
    np.testing.assert_allclose(result[3], 0.0, atol=1e-6)


def test_within_cell_rank_order():
    """Higher counts in a cell give higher values, and equal counts equal values."""
    result = _clr(X_clr, csr_matrix)
    order = np.argsort(X_clr, axis=1, kind="stable")
    count_steps = np.diff(np.take_along_axis(X_clr, order, axis=1), axis=1)
    value_steps = np.diff(np.take_along_axis(result, order, axis=1), axis=1)
    np.testing.assert_array_equal(np.sign(value_steps), np.sign(count_steps))


@pytest.mark.parametrize("axis", [0, 1], ids=["cells", "genes"])
def test_permutation_equivariance(axis):
    """Reordering cells or genes reorders the output the same way."""
    perm = np.array([2, 0, 3, 1])
    np.testing.assert_allclose(
        _clr(np.take(X_clr, perm, axis=axis), csr_matrix),
        np.take(_clr(X_clr, csr_matrix), perm, axis=axis),
        atol=1e-6,
    )


def test_log_ratio():
    """With fixed `alpha`, a within-cell log-ratio ignores the other genes' counts."""
    changed = X_clr.copy()
    changed[0, 3] = 50
    result, result_changed = _clr(X_clr, alpha=0.5), _clr(changed, alpha=0.5)
    np.testing.assert_allclose(result_changed[0, 0] - result_changed[0, 2], result[0, 0] - result[0, 2], atol=1e-6)
    # other cells are normalized independently
    np.testing.assert_allclose(result_changed[1:], result[1:], atol=1e-6)


def test_alpha_scales_with_counts():
    """Scaling counts by k and `alpha` by 1/k leaves the result unchanged."""
    k = 3
    np.testing.assert_allclose(_clr(X_clr * k, alpha=0.5 / k), _clr(X_clr, alpha=0.5), atol=1e-6)


@pytest.mark.parametrize("alpha", [0.1, 0.3, 1.0])
def test_estimate_nb_overdispersion_simulated(alpha):
    """Recover a known `alpha` from negative-binomial counts."""
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.5, 20, size=300)
    n = 1 / alpha  # numpy's NB has var = mu + mu**2 / n
    x = rng.negative_binomial(n, n / (n + mu), size=(5000, mu.size))
    assert _estimate_nb_overdispersion(csr_matrix(cp.asarray(x, dtype=cp.float64))) == pytest.approx(alpha, rel=0.05)


@pytest.mark.parametrize("alpha", [0.0, -0.5], ids=["zero", "negative"])
def test_nonpositive_alpha_raises(alpha):
    adata = AnnData(csr_matrix(cp.asarray(X_clr)))
    with pytest.raises(ValueError, match=r"alpha.*positive"):
        rsc.pp.normalize_shifted_clr(adata, alpha=alpha)


def test_estimated_alpha_zero_mean_raises():
    adata = AnnData(cp.zeros((3, 4), dtype=cp.float32))
    with pytest.raises(ValueError, match="Cannot estimate overdispersion"):
        rsc.pp.normalize_shifted_clr(adata)


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
def test_zero_cell(array_type):
    """Keep an empty cell finite and all-zero."""
    x = X_clr.copy()
    x[1] = 0
    adata = AnnData(array_type(cp.asarray(x)))
    with pytest.warns(UserWarning, match="Some cells have zero counts"):
        rsc.pp.normalize_shifted_clr(adata)
    result = cp.asnumpy(adata.X)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[1], 0.0, atol=1e-6)


@pytest.mark.parametrize("array_type", [csr_matrix, csc_matrix])
@pytest.mark.parametrize("n_vars", [5, 3000])
def test_sparse_matches_dense(array_type, n_vars):
    rng = np.random.default_rng(1)
    x = rng.negative_binomial(1, 0.3, size=(200, n_vars)).astype(np.float64)
    x[rng.random(x.shape) < 0.6] = 0
    x[:, 0] += 1  # no empty cells
    np.testing.assert_allclose(_clr(x, array_type, alpha=0.7), _clr(x, cp.asarray, alpha=0.7), rtol=1e-6, atol=1e-6)


def test_inplace_false():
    adata = AnnData(csr_matrix(cp.asarray(X_clr_estimated)))
    x_before = _dense(adata.X).copy()
    with pytest.warns(FutureWarning, match="inplace"):
        out = rsc.pp.normalize_shifted_clr(adata, inplace=False)

    assert isinstance(out, dict)
    assert set(out) == {"X"}
    np.testing.assert_allclose(cp.asnumpy(out["X"]), X_clr_estimated_expected, rtol=1e-5, atol=1e-6)
    # input is left untouched
    assert isinstance(adata.X, csr_matrix)
    np.testing.assert_array_equal(_dense(adata.X), x_before)


def test_inplace_false_out_error():
    A = pytest.importorskip("anndata.acc").A
    adata = AnnData(cp.asarray(X_clr))
    with pytest.raises(TypeError, match="either `out` or `inplace`"), pytest.warns(FutureWarning, match="inplace"):
        rsc.pp.normalize_shifted_clr(adata, inplace=False, out=A.layers["clr"])
    assert "clr" not in adata.layers


def test_out_true():
    adata = AnnData(csr_matrix(cp.asarray(X_clr_estimated)))
    out = rsc.pp.normalize_shifted_clr(adata, out=True)
    np.testing.assert_allclose(cp.asnumpy(out), X_clr_estimated_expected, rtol=1e-5, atol=1e-6)
    assert isinstance(adata.X, csr_matrix)


def test_copy():
    adata = AnnData(csr_matrix(cp.asarray(X_clr_estimated)))
    with pytest.warns(FutureWarning, match="copy"):
        returned = rsc.pp.normalize_shifted_clr(adata, copy=True)

    assert isinstance(returned, AnnData)
    assert returned is not adata
    np.testing.assert_allclose(cp.asnumpy(returned.X), X_clr_estimated_expected, rtol=1e-5, atol=1e-6)
    assert isinstance(adata.X, csr_matrix)


def test_copy_inplace_error():
    adata = AnnData(csr_matrix(cp.asarray(X_clr)))
    with pytest.raises(ValueError, match="`copy=True` cannot be used"), pytest.warns(FutureWarning):
        rsc.pp.normalize_shifted_clr(adata, copy=True, inplace=False)


def test_use():
    """`use` selects the input layer and leaves `X` untouched."""
    A = pytest.importorskip("anndata.acc").A
    adata = AnnData(
        csr_matrix(cp.asarray(X_clr_estimated)),
        layers={"counts": csr_matrix(cp.asarray(X_clr_estimated))},
    )
    x_before = _dense(adata.X).copy()
    rsc.pp.normalize_shifted_clr(adata, use=A.layers["counts"])

    np.testing.assert_array_equal(_dense(adata.X), x_before)
    assert isinstance(adata.layers["counts"], cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(adata.layers["counts"]), X_clr_estimated_expected, rtol=1e-5, atol=1e-6)


def test_view():
    adata = AnnData(cp.asarray(X_clr))
    v = adata[:, :]
    with pytest.warns(UserWarning, match=r"Received a view"):
        rsc.pp.normalize_shifted_clr(v)
    assert not v.is_view
    cp.testing.assert_array_equal(adata.X, cp.asarray(X_clr))
