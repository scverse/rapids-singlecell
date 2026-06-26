"""Unit tests for harmony CUDA kernels against CuPy references."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

from rapids_singlecell._cuda import (
    _harmony_clustering_cuda as _cl,
)
from rapids_singlecell._cuda import (
    _harmony_colsum_cuda as _colsum,
)
from rapids_singlecell._cuda import (
    _harmony_correction_cuda as _corr,
)
from rapids_singlecell._cuda import (
    _harmony_kmeans_cuda as _km,
)
from rapids_singlecell._cuda import (
    _harmony_normalize_cuda as _norm,
)
from rapids_singlecell._cuda import (
    _harmony_pen_cuda as _pen,
)
from rapids_singlecell._cuda import (
    _harmony_scatter_cuda as _scatter,
)

pytestmark = pytest.mark.skipif(
    _norm is None
    or _pen is None
    or _scatter is None
    or _colsum is None
    or _km is None
    or _cl is None
    or _corr is None,
    reason="Harmony CUDA modules not available",
)

DTYPES = [np.float32, np.float64]


def _random_idx(n_src, n_dst, seed=42):
    """Generate random unique indices using NumPy (CuPy Generator lacks choice)."""
    idx_np = np.random.default_rng(seed).choice(n_src, size=n_dst, replace=False)
    return cp.asarray(idx_np, dtype=cp.int32)


# ---------- l2_row_normalize ----------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_rows,n_cols", [(100, 50), (1, 20), (500, 3)])
def test_l2_row_normalize(dtype, n_rows, n_cols):
    rng = cp.random.default_rng(42)
    src = rng.standard_normal((n_rows, n_cols), dtype=dtype)
    dst = cp.empty_like(src)

    _norm.l2_row_normalize(src, dst=dst, n_rows=n_rows, n_cols=n_cols)
    cp.cuda.Device().synchronize()

    # Reference: L2 row normalize
    norms = cp.linalg.norm(src, axis=1, keepdims=True)
    norms = cp.maximum(norms, 1e-12)
    expected = src / norms

    atol = 1e-6 if dtype == np.float32 else 1e-12
    cp.testing.assert_allclose(dst, expected, atol=atol, rtol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_l2_row_normalize_zero_row(dtype):
    """Zero rows should not produce NaN (clamped to 1e-12)."""
    src = cp.zeros((3, 10), dtype=dtype)
    src[1, :] = 1.0  # only middle row is non-zero
    dst = cp.empty_like(src)

    _norm.l2_row_normalize(src, dst=dst, n_rows=3, n_cols=10)
    cp.cuda.Device().synchronize()

    assert not cp.any(cp.isnan(dst))
    # Zero rows should stay zero (0 / clamp(0, 1e-12) = 0)
    cp.testing.assert_array_equal(dst[0], cp.zeros(10, dtype=dtype))
    cp.testing.assert_array_equal(dst[2], cp.zeros(10, dtype=dtype))


# ---------- penalty_kernel ----------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_batches,n_clusters", [(3, 20), (10, 100), (1, 5)])
@pytest.mark.parametrize("stabilized", [True, False])
def test_penalty(dtype, n_batches, n_clusters, stabilized):
    rng = cp.random.default_rng(123)
    E = rng.random((n_batches, n_clusters), dtype=dtype) * 10
    O = rng.random((n_batches, n_clusters), dtype=dtype) * 10
    theta = rng.random(n_batches, dtype=dtype) * 2 + 0.5
    penalty = cp.empty_like(E)

    _pen.penalty(
        E,
        O=O,
        theta=theta,
        penalty=penalty,
        n_batches=n_batches,
        n_clusters=n_clusters,
        stabilized=stabilized,
    )
    cp.cuda.Device().synchronize()

    # Reference
    denom = (O + E + 1) if stabilized else (O + 1)
    expected = cp.power(
        (E + 1) / denom,
        theta[:, None],
    )

    atol = 1e-5 if dtype == np.float32 else 1e-10
    cp.testing.assert_allclose(penalty, expected, atol=atol, rtol=1e-4)


# ---------- fused_pen_norm_kernel_int ----------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_rows,n_cols", [(200, 50), (50, 100)])
def test_fused_pen_norm_int(dtype, n_rows, n_cols):
    rng = cp.random.default_rng(99)
    n_batches = 3

    similarities = rng.random((n_rows, n_cols), dtype=dtype)
    penalty = rng.random((n_batches, n_cols), dtype=dtype) + 0.1
    cats = rng.integers(0, n_batches, size=n_rows).astype(cp.int32)
    idx_in = cp.arange(n_rows, dtype=cp.int32)  # identity permutation
    R_out = cp.empty((n_rows, n_cols), dtype=dtype)
    term = float(dtype(-2) / dtype(0.1))  # native Python float for nanobind

    _pen.fused_pen_norm_int(
        similarities,
        penalty=penalty,
        cats=cats,
        idx_in=idx_in,
        R_out=R_out,
        term=term,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    cp.cuda.Device().synchronize()

    # Reference: exp(term * (1 - sim)) * penalty[cat], then row-normalize
    sim_gathered = similarities[idx_in]
    raw = cp.exp(dtype(term) * (1 - sim_gathered)) * penalty[cats]
    expected = raw / raw.sum(axis=1, keepdims=True)

    atol = 1e-5 if dtype == np.float32 else 1e-10
    cp.testing.assert_allclose(R_out, expected, atol=atol, rtol=1e-4)


@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_pen_norm_int_with_permutation(dtype):
    """Test with non-identity permutation to verify idx_in gather."""
    n_rows, n_cols, n_batches = 100, 30, 4
    rng = cp.random.default_rng(77)

    similarities = rng.random((200, n_cols), dtype=dtype)  # larger source
    penalty = rng.random((n_batches, n_cols), dtype=dtype) + 0.1
    cats = rng.integers(0, n_batches, size=n_rows).astype(cp.int32)
    idx_in = _random_idx(200, n_rows, seed=77)
    R_out = cp.empty((n_rows, n_cols), dtype=dtype)
    term = float(dtype(-20.0))

    _pen.fused_pen_norm_int(
        similarities,
        penalty=penalty,
        cats=cats,
        idx_in=idx_in,
        R_out=R_out,
        term=term,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    cp.cuda.Device().synchronize()

    sim_gathered = similarities[idx_in]
    raw = cp.exp(dtype(term) * (1 - sim_gathered)) * penalty[cats]
    expected = raw / raw.sum(axis=1, keepdims=True)

    atol = 1e-5 if dtype == np.float32 else 1e-10
    cp.testing.assert_allclose(R_out, expected, atol=atol, rtol=1e-4)


# ---------- gather_rows ----------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_rows,n_cols", [(100, 50), (500, 20)])
def test_gather_rows(dtype, n_rows, n_cols):
    rng = cp.random.default_rng(42)
    src = rng.standard_normal((n_rows * 2, n_cols), dtype=dtype)
    idx = _random_idx(n_rows * 2, n_rows)
    dst = cp.empty((n_rows, n_cols), dtype=dtype)

    _scatter.gather_rows(src, idx=idx, dst=dst, n_rows=n_rows, n_cols=n_cols)
    cp.cuda.Device().synchronize()

    expected = src[idx]
    cp.testing.assert_array_equal(dst, expected)


# ---------- scatter_rows ----------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_rows,n_cols", [(100, 50), (500, 20)])
def test_scatter_rows(dtype, n_rows, n_cols):
    rng = cp.random.default_rng(42)
    src = rng.standard_normal((n_rows, n_cols), dtype=dtype)
    idx_np = np.arange(n_rows, dtype=np.int32)
    np.random.default_rng(42).shuffle(idx_np)
    idx = cp.asarray(idx_np)
    dst = cp.zeros((n_rows, n_cols), dtype=dtype)

    _scatter.scatter_rows(src, idx=idx, dst=dst, n_rows=n_rows, n_cols=n_cols)
    cp.cuda.Device().synchronize()

    expected = cp.zeros_like(dst)
    expected[idx] = src
    cp.testing.assert_array_equal(dst, expected)


# ---------- gather_int ----------


@pytest.mark.parametrize("n", [100, 10000])
def test_gather_int(n):
    rng = cp.random.default_rng(42)
    src = rng.integers(0, 1000, size=n * 2).astype(cp.int32)
    idx = _random_idx(n * 2, n)
    dst = cp.empty(n, dtype=cp.int32)

    _scatter.gather_int(src, idx=idx, dst=dst, n=n)
    cp.cuda.Device().synchronize()

    expected = src[idx]
    cp.testing.assert_array_equal(dst, expected)


# ---------- colsum ----------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows", [0, 1, 257, 4096])
def test_colsum_columns_multiple_cols_per_block(dtype, rows):
    rng = cp.random.default_rng(123)
    n_sm = cp.cuda.Device().attributes["MultiProcessorCount"]
    cols = n_sm * 8 + 17
    x = rng.standard_normal((rows, cols), dtype=dtype)
    out = cp.zeros(cols, dtype=dtype)

    _colsum.colsum(x, out=out, rows=rows, cols=cols)
    cp.cuda.Device().synchronize()

    atol = 5e-4 if dtype == np.float32 else 1e-12
    rtol = 1e-4 if dtype == np.float32 else 1e-5
    expected_np = cp.asnumpy(x).sum(axis=0)
    cp.testing.assert_allclose(out, cp.asarray(expected_np), atol=atol, rtol=rtol)


# ---------- kmeans_err ----------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [0, 1, 257, 8192])
def test_kmeans_err_offset_contiguous(dtype, n):
    r_base = cp.arange(n + 1, dtype=dtype)
    dot_base = cp.linspace(dtype(0.1), dtype(0.9), n + 1, dtype=dtype)
    r = r_base[1:]
    dot = dot_base[1:]
    out = cp.zeros(1, dtype=dtype)

    _km.kmeans_err(r, dot=dot, n=n, out=out)
    cp.cuda.Device().synchronize()

    r_np = cp.asnumpy(r)
    dot_np = cp.asnumpy(dot)
    expected_np = np.sum(r_np * dtype(2) * (dtype(1) - dot_np), dtype=dtype)
    atol = 1e-4 if dtype == np.float32 else 1e-10
    cp.testing.assert_allclose(out[0], dtype(expected_np), atol=atol, rtol=1e-5)


# ---------- compute_objective ----------


def _compute_objective_reference(R, similarities, *, O, E, theta, sigma, stabilized):
    """Pure CuPy reference for the three-term harmony objective."""
    # K-means error: sum(R[i] * 2 * (1 - sim[i]))
    kmeans_err = float(cp.sum(R * 2 * (1 - similarities)))

    # Entropy: sigma * sum(x_norm * log(x_norm + eps))
    R_norm = R / R.sum(axis=1, keepdims=True)
    entropy = float(sigma * cp.sum(R_norm * cp.log(R_norm + 1e-12)))

    # Diversity
    numer = (O + E + 1) if stabilized else (O + 1)
    diversity = float(sigma * cp.sum(theta[:, None] * O * cp.log(numer / (E + 1))))

    return kmeans_err + entropy + diversity


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "n_cells,n_clusters,n_batches", [(500, 20, 3), (1000, 100, 10), (200, 5, 2)]
)
@pytest.mark.parametrize("stabilized", [True, False])
def test_compute_objective(dtype, n_cells, n_clusters, n_batches, stabilized):
    rng = cp.random.default_rng(42)

    # R must be non-negative (cluster assignments)
    R = rng.random((n_cells, n_clusters), dtype=dtype) + 0.01
    similarities = rng.random((n_cells, n_clusters), dtype=dtype)
    O = rng.random((n_batches, n_clusters), dtype=dtype) * 10 + 0.1
    E = rng.random((n_batches, n_clusters), dtype=dtype) * 10 + 0.1
    theta = rng.random(n_batches, dtype=dtype) * 2
    sigma = 0.1

    obj_scalar = cp.zeros(1, dtype=dtype)
    result = _cl.compute_objective(
        R,
        similarities=similarities,
        O=O,
        E=E,
        theta=theta,
        sigma=float(sigma),
        obj_scalar=obj_scalar,
        n_cells=n_cells,
        n_clusters=n_clusters,
        n_batches=n_batches,
        stabilized=stabilized,
    )

    expected = _compute_objective_reference(
        R, similarities, O=O, E=E, theta=theta, sigma=sigma, stabilized=stabilized
    )

    # atomicAdd reduction ordering causes slight non-determinism
    rtol = 1e-5 if dtype == np.float32 else 1e-10
    np.testing.assert_allclose(result, expected, rtol=rtol)


# ---- compute_inv_mat: correctness against CuPy reference ----


def _inv_mat_reference(O_col, lambda_col, dtype):
    """Pure CuPy reference for the algebraic fast-inverse."""
    n_batches = len(O_col)
    nb1 = n_batches + 1
    O_col = O_col.astype(dtype)
    lambda_col = lambda_col.astype(dtype)

    factor = dtype(1) / (O_col + lambda_col)
    P_row0 = -factor * O_col
    N_k = O_col.sum()
    c = N_k - (factor * O_col * O_col).sum()
    c_inv = dtype(1) / c

    inv = cp.empty((nb1, nb1), dtype=dtype)
    inv[0, 0] = c_inv
    inv[0, 1:] = c_inv * P_row0
    inv[1:, 0] = P_row0 * c_inv
    inv[1:, 1:] = cp.outer(P_row0, P_row0) * c_inv + cp.diag(factor)
    return inv


@pytest.mark.parametrize("n_batches", [5, 50, 200])
@pytest.mark.parametrize("n_clusters", [10, 50])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_compute_inv_mat(n_batches, n_clusters, dtype):
    """Test that compute_inv_mat matches CuPy reference."""
    rng = np.random.default_rng(42)
    O = cp.array(rng.random((n_batches, n_clusters)) * 100, dtype=dtype)
    # Non-uniform lambda_kb to catch stride/transpose bugs
    lambda_kb = cp.array(rng.random((n_batches, n_clusters)) * 5 + 0.1, dtype=dtype)
    lambda_kb[0, 0] = dtype(1e30)  # sentinel: exercises the pruning path
    nb1 = n_batches + 1
    stream = cp.cuda.get_current_stream().ptr

    g_factor = cp.empty(n_batches, dtype=dtype)
    g_P_row0 = cp.empty(n_batches, dtype=dtype)

    for k in range(min(n_clusters, 3)):  # test a few clusters
        inv_mat = cp.empty((nb1, nb1), dtype=dtype)

        _corr.compute_inv_mat(
            O,
            lambda_kb=lambda_kb,
            n_batches=n_batches,
            n_clusters=n_clusters,
            cluster_k=k,
            inv_mat=inv_mat,
            g_factor=g_factor,
            g_P_row0=g_P_row0,
            stream=stream,
        )
        cp.cuda.Device().synchronize()

        expected = _inv_mat_reference(O[:, k], lambda_kb[:, k], dtype)
        atol = 1e-6 if dtype == cp.float32 else 1e-12
        cp.testing.assert_allclose(inv_mat, expected, atol=atol, rtol=1e-5)


# ---------- edge cases: absent batches (O=0) ----------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("stabilized", [True, False])
def test_penalty_absent_batch(dtype, stabilized):
    """Penalty kernel stays finite when some O entries are zero."""
    n_batches, n_clusters = 4, 10
    rng = cp.random.default_rng(42)
    E = rng.random((n_batches, n_clusters), dtype=dtype) * 10
    O = rng.random((n_batches, n_clusters), dtype=dtype) * 10
    O[0, 0] = 0
    O[2, 5] = 0
    theta = rng.random(n_batches, dtype=dtype) * 2 + 0.5
    penalty = cp.empty_like(E)

    _pen.penalty(
        E,
        O=O,
        theta=theta,
        penalty=penalty,
        n_batches=n_batches,
        n_clusters=n_clusters,
        stabilized=stabilized,
    )
    cp.cuda.Device().synchronize()

    assert cp.all(cp.isfinite(penalty)), "Penalty has non-finite values with O=0"

    denom = (O + E + 1) if stabilized else (O + 1)
    expected = cp.power((E + 1) / denom, theta[:, None])
    atol = 1e-5 if dtype == np.float32 else 1e-10
    cp.testing.assert_allclose(penalty, expected, atol=atol, rtol=1e-4)


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_compute_inv_mat_absent_batch(dtype):
    """compute_inv_mat stays finite when O=0 and lambda_kb is large (pruned)."""
    n_batches, n_clusters = 5, 10
    rng = np.random.default_rng(99)
    O = cp.array(rng.random((n_batches, n_clusters)) * 100, dtype=dtype)
    lambda_kb = cp.array(rng.random((n_batches, n_clusters)) * 2 + 0.1, dtype=dtype)

    O[0, 0] = 0
    lambda_kb[0, 0] = dtype(1e30)
    O[3, 2] = 0
    lambda_kb[3, 2] = dtype(1e30)

    nb1 = n_batches + 1
    stream = cp.cuda.get_current_stream().ptr
    g_factor = cp.empty(n_batches, dtype=dtype)
    g_P_row0 = cp.empty(n_batches, dtype=dtype)

    for k in [0, 2]:
        inv_mat = cp.empty((nb1, nb1), dtype=dtype)
        _corr.compute_inv_mat(
            O,
            lambda_kb=lambda_kb,
            n_batches=n_batches,
            n_clusters=n_clusters,
            cluster_k=k,
            inv_mat=inv_mat,
            g_factor=g_factor,
            g_P_row0=g_P_row0,
            stream=stream,
        )
        cp.cuda.Device().synchronize()

        assert cp.all(cp.isfinite(inv_mat)), f"inv_mat non-finite for cluster {k}"
        expected = _inv_mat_reference(O[:, k], lambda_kb[:, k], dtype)
        atol = 1e-6 if dtype == cp.float32 else 1e-12
        cp.testing.assert_allclose(inv_mat, expected, atol=atol, rtol=1e-5)
