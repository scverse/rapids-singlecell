from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import rapids_singlecell.decoupler_gpu as dc


def test_funcs(rng):
    x = cp.array([[1, 2, 3, 4]], dtype=cp.float32)
    w = cp.array([rng.random(x.size)], dtype=cp.float32).T
    es = dc._method_waggr._wsum(x=x, w=w)
    assert isinstance(es, cp.ndarray)
    es = dc._method_waggr._wmean(x=x, w=w)
    assert isinstance(es, cp.ndarray)


def test_wsum_wmean(mat, adjmat):
    print("\n=== Testing with test data ===")
    X, obs, var = mat
    X = cp.array(X, dtype=cp.float32)
    adjmat = cp.array(adjmat, dtype=cp.float32)

    print(f"X shape: {X.shape}, adjmat shape: {adjmat.shape}")
    print(f"X dtype: {X.dtype}, adjmat dtype: {adjmat.dtype}")
    print(f"X min/max: {cp.min(X):.6f} / {cp.max(X):.6f}")
    print(f"adjmat min/max: {cp.min(adjmat):.6f} / {cp.max(adjmat):.6f}")

    # Test _wsum
    result_actual = dc._method_waggr._wsum(X, adjmat)
    expected_actual = X @ adjmat

    print(
        f"Expected shape: {expected_actual.shape}, Result shape: {result_actual.shape}"
    )
    print(
        f"Expected min/max: {cp.min(expected_actual):.6f} / {cp.max(expected_actual):.6f}"
    )
    print(f"Result min/max: {cp.min(result_actual):.6f} / {cp.max(result_actual):.6f}")

    is_close_actual = cp.allclose(expected_actual, result_actual, rtol=1e-4)
    print(f"wsum results match: {is_close_actual}")

    assert is_close_actual, "_wsum test failed"

    # Test _wmean
    result_wmean = dc._method_waggr._wmean(X, adjmat)
    div = cp.sum(cp.abs(adjmat), axis=0)
    expected_wmean = expected_actual / div
    is_close_actual = cp.allclose(expected_wmean, result_wmean, rtol=1e-4)
    print(f"wmean results match: {is_close_actual}")
    assert is_close_actual, "_wmean test failed"


@pytest.mark.parametrize(
    "fun,times,seed",
    [
        ["wmean", 0, 42],
        ["wsum", 0, 23],
    ],
)
def test_func_waggr(
    mat,
    adjmat,
    fun,
    times,
    seed,
):
    X, obs, var = mat
    X = cp.array(X)
    adjmat = cp.array(adjmat)
    es, pv = dc._method_waggr._func_waggr(
        mat=X, adj=adjmat, fun=fun, times=times, seed=seed
    )
    assert np.isfinite(es).all()
    assert ((0 <= pv) & (pv <= 1)).all()


@pytest.mark.parametrize("fun", ["wmean", "wsum"])
def test_func_waggr_permutation(mat, adjmat, fun):
    """Test waggr with permutation testing (times > 1)."""
    X, obs, var = mat
    X = cp.array(X)
    adjmat = cp.array(adjmat)
    times = 100
    seed = 42
    es, pv = dc._method_waggr._func_waggr(
        mat=X, adj=adjmat, fun=fun, times=times, seed=seed
    )
    assert np.isfinite(es).all()
    assert ((0 <= pv) & (pv <= 1)).all()
    # With permutation, es becomes NES (normalized enrichment score)
    # and pv are empirical p-values


def test_ridx():
    """Test _ridx random index generation."""
    times = 10
    nvar = 20
    seed = 42
    idx = dc._method_waggr._ridx(times=times, nvar=nvar, seed=seed)
    assert idx.shape == (times, nvar)
    # Check that each row contains all indices (is a permutation)
    for i in range(times):
        assert set(idx[i].get().tolist()) == set(range(nvar))


def test_ridx_reproducible():
    """Test that _ridx is reproducible with same seed."""
    idx1 = dc._method_waggr._ridx(times=5, nvar=10, seed=42)
    idx2 = dc._method_waggr._ridx(times=5, nvar=10, seed=42)
    assert cp.array_equal(idx1, idx2)


def test_custom_callable(mat, adjmat):
    """Test waggr with a custom callable function."""

    def custom_weighted_sum(x, w):
        return x @ w

    X, obs, var = mat
    X = cp.array(X, dtype=cp.float32)
    adjmat = cp.array(adjmat, dtype=cp.float32)
    es, pv = dc._method_waggr._func_waggr(
        mat=X, adj=adjmat, fun=custom_weighted_sum, times=0, seed=42
    )
    assert np.isfinite(es).all()
    assert ((0 <= pv) & (pv <= 1)).all()


def test_custom_callable_with_permutation(mat, adjmat):
    """Test waggr with a custom callable and permutation testing."""

    def custom_wmean(x, w):
        agg = x @ w
        div = cp.sum(cp.abs(w), axis=0)
        return agg / div

    X, obs, var = mat
    X = cp.array(X, dtype=cp.float32)
    adjmat = cp.array(adjmat, dtype=cp.float32)
    es, pv = dc._method_waggr._func_waggr(
        mat=X, adj=adjmat, fun=custom_wmean, times=50, seed=42
    )
    assert np.isfinite(es).all()
    assert ((0 <= pv) & (pv <= 1)).all()


def test_perm_function(mat, adjmat):
    """Test _perm function directly."""
    X, obs, var = mat
    X = cp.array(X, dtype=cp.float32)
    adjmat = cp.array(adjmat, dtype=cp.float32)

    # Compute initial ES
    es = dc._method_waggr._wmean(X, adjmat)

    # Generate permutation indices
    times = 50
    nvar = X.shape[1]
    idx = dc._method_waggr._ridx(times=times, nvar=nvar, seed=42)

    # Run permutation
    nes, pv = dc._method_waggr._perm(
        fun=dc._method_waggr._wmean, es=es, mat=X, adj=adjmat, idx=idx
    )
    assert np.isfinite(nes.get()).all()
    assert ((0 <= pv.get()) & (pv.get() <= 1)).all()
