"""Stream and allocation regressions for native Harmony/GMM orchestration."""

from __future__ import annotations

from contextlib import nullcontext

import cupy as cp
import numpy as np
import pytest

from rapids_singlecell._cuda import _gmm_cuda

pytestmark = pytest.mark.skipif(
    getattr(_gmm_cuda, "__backend__", None) != "rust", reason="requires Rust backend"
)


def _pinned_em(values, control, guide, *, max_iter, tol, reg):
    """Independent CPU reference for the projected two-component mixture."""
    m0, m1 = values[control].mean(), values[guide].mean()
    v0 = max(values[control].var(ddof=1), 1e-12, reg)
    v1 = max(values[guide].var(ddof=1), 1e-12)
    w1, previous = 0.5, -1e30
    for _ in range(max_iter):
        lp0 = np.log(max(1 - w1, 1e-10)) - 0.5 * np.log(v0)
        lp1 = np.log(max(w1, 1e-10)) - 0.5 * np.log(v1)
        lp0 -= 0.5 * (values - m0) ** 2 / v0
        lp1 -= 0.5 * (values - m1) ** 2 / v1
        logsum = np.logaddexp(lp0, lp1)
        mean = logsum.mean() - 0.5 * np.log(2 * np.pi)
        if abs(mean - previous) < tol:
            break
        previous = mean
        r1 = np.exp(lp1 - logsum)
        count = r1.sum()
        m1 = np.dot(r1, values) / max(count, 1e-12)
        v1 = max(np.dot(r1, values**2) / max(count, 1e-12) - m1**2 + reg, reg)
        w1 = count / len(values)
    lp0 = np.log(max(1 - w1, 1e-10)) - 0.5 * np.log(v0)
    lp1 = np.log(max(w1, 1e-10)) - 0.5 * np.log(v1)
    lp0 -= 0.5 * (values - m0) ** 2 / v0
    lp1 -= 0.5 * (values - m1) ** 2 / v1
    return 1 / (1 + np.exp(lp0 - lp1))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("managed", [False, True])
def test_projection_em_preserves_stream_and_temporary_lifetime(dtype, managed):
    rng = np.random.default_rng(12)
    counts, features = [100, 80], [3, 4]
    matrices, guides, controls, nt_means, projections, probabilities = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for n, k in zip(counts, features, strict=True):
        guide = np.arange(n) >= n // 2
        control = ~guide
        data = rng.normal(size=(n, k)).astype(dtype)
        data[guide, 0] += 3
        nt_mean = data[control].mean(axis=0)
        direction = data[guide].mean(axis=0) - nt_mean
        projection = data @ direction / np.dot(direction, direction)
        matrices.append(data)
        guides.append(guide)
        controls.append(control)
        nt_means.append(nt_mean)
        projections.append(projection)
        probabilities.append(
            _pinned_em(projection, control, guide, max_iter=100, tol=1e-6, reg=1e-4)
        )

    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed) if managed else None
    allocator = (
        cp.cuda.using_allocator(pool.malloc) if pool is not None else nullcontext()
    )
    caller = cp.cuda.get_current_stream()
    work = cp.cuda.Stream(non_blocking=True)
    with allocator:
        with work:
            arrays = [
                cp.asarray(np.concatenate([a.ravel() for a in matrices])),
                cp.asarray([0, counts[0] * features[0]], dtype=cp.int64),
                cp.asarray(counts, dtype=cp.int32),
                cp.asarray(features, dtype=cp.int32),
                cp.asarray([0, counts[0]], dtype=cp.int32),
                cp.asarray([0, features[0]], dtype=cp.int32),
                cp.asarray(np.concatenate(nt_means)),
                cp.asarray(np.concatenate(guides)),
                cp.asarray(np.concatenate(controls)),
                cp.asarray([0, 1], dtype=cp.int32),
            ]
            projection_out = cp.empty(sum(counts), dtype=dtype)
            probability_out = cp.empty_like(projection_out)
        _gmm_cuda.mixscape_project_em(
            *arrays,
            projection_out,
            probability_out,
            n_active=2,
            max_k=max(features),
            max_iter=100,
            tol=1e-6,
            reg_covar=1e-4,
            stream=work.ptr,
        )
        assert cp.cuda.get_current_stream().ptr == caller.ptr
        # Stress recycling on the caller's other stream while the native kernel
        # may still consume its internal projection-vector allocation.
        allocations = [cp.full((2, 4), 123, dtype=dtype) for _ in range(16)]
        work.synchronize()
        tolerance = 3e-5 if dtype == np.float32 else 1e-10
        np.testing.assert_allclose(
            cp.asnumpy(projection_out),
            np.concatenate(projections),
            atol=tolerance,
            rtol=tolerance,
        )
        np.testing.assert_allclose(
            cp.asnumpy(probability_out),
            np.concatenate(probabilities),
            atol=tolerance,
            rtol=tolerance,
        )
        assert len(allocations) == 16


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gmm_rejects_overlapping_output_before_mutation(dtype):
    n, d, k = 4, 3, 3
    data = cp.arange(n * d, dtype=dtype).reshape(n, d)
    before = data.copy()
    weights = cp.full(k, 1 / k, dtype=dtype)
    means = cp.zeros((k, d), dtype=dtype)
    precision = cp.broadcast_to(cp.eye(d, dtype=dtype), (k, d, d)).copy()
    logdet = cp.zeros(k, dtype=dtype)
    resp = cp.empty((n, k), dtype=dtype)
    likelihood = cp.empty(n, dtype=dtype)
    with pytest.raises(ValueError, match="overlap"):
        _gmm_cuda.e_step(
            data,
            weights,
            means,
            precision,
            logdet,
            data,
            resp,
            likelihood,
            n=n,
            d=d,
            K=k,
        )
    cp.testing.assert_array_equal(data, before)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_large_harmony_intercept_matches_ridge_reference(dtype):
    from rapids_singlecell._cuda import _harmony_correction_cuda

    rng = np.random.default_rng(48)
    n, d, k, b = 300_003, 3, 2, 3
    categories = rng.integers(0, b, n, dtype=np.int32)
    data = (rng.normal(size=(n, d)) + categories[:, None]).astype(dtype)
    responsibilities = rng.uniform(0.1, 1, (n, k)).astype(dtype)
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    counts = np.array(
        [
            responsibilities[categories == batch].sum(axis=0, dtype=np.float64)
            for batch in range(b)
        ],
        dtype=dtype,
    )
    penalties = np.full((b, k), 50_000, dtype=dtype)
    expected = data.astype(np.float64)
    for cluster in range(k):
        weights = responsibilities[:, cluster].astype(np.float64)
        weighted = data * weights[:, None]
        rhs = np.vstack(
            [weighted.sum(axis=0)]
            + [weighted[categories == batch].sum(axis=0) for batch in range(b)]
        )
        count = counts[:, cluster].astype(np.float64)
        gram = np.diag(np.r_[count.sum(), count + penalties[:, cluster]])
        gram[0, 1:] = count
        gram[1:, 0] = count
        coefficients = np.linalg.solve(gram, rhs)
        expected -= weights[:, None] * coefficients[1:][categories]
    order = np.argsort(categories, kind="stable").astype(np.int32)
    offsets = np.r_[0, np.cumsum(np.bincount(categories, minlength=b))].astype(np.int32)
    work = cp.cuda.Stream(non_blocking=True)
    with work:
        x = cp.asarray(data)
        arguments = {
            "R": cp.asarray(responsibilities),
            "O": cp.asarray(counts),
            "cats": cp.asarray(categories),
            "cat_offsets": cp.asarray(offsets),
            "cell_indices": cp.asarray(order),
            "lambda_kb": cp.asarray(penalties),
            "n_cells": n,
            "n_pcs": d,
            "n_clusters": k,
            "n_batches": b,
            "Z": cp.empty_like(x),
            "inv_mat": cp.empty((b + 1, b + 1), dtype=dtype),
            "R_col": cp.empty(n, dtype=dtype),
            "Phi_t_diag_R_X": cp.empty((b + 1, d), dtype=dtype),
            "W": cp.empty((b + 1, d), dtype=dtype),
            "g_factor": cp.empty(b, dtype=dtype),
            "g_P_row0": cp.empty(b, dtype=dtype),
            "stream": work.ptr,
            "handle": cp.cuda.device.get_cublas_handle(),
        }
    _harmony_correction_cuda.correction_fast(x, **arguments)
    work.synchronize()
    tolerance = 1e-5 if dtype == np.float32 else 1e-11
    np.testing.assert_allclose(
        cp.asnumpy(arguments["Z"]), expected, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gmm_blas_and_solver_use_the_explicit_stream(dtype):
    rng = np.random.default_rng(38)
    n, d, k = 513, 3, 2
    data = rng.normal(size=(n, d)).astype(dtype)
    initial_means = np.array([[0, 0, 0], [1, 0.5, -1]], dtype=dtype)
    initial_weights = np.array([0.4, 0.6], dtype=dtype)
    variances = np.array([[1, 2, 3], [2, 3, 4]], dtype=dtype)
    delta = data[:, None, :].astype(np.float64) - initial_means
    log_prob = np.log(initial_weights.astype(np.float64)) - 0.5 * (
        d * np.log(2 * np.pi)
        + np.log(variances.astype(np.float64)).sum(axis=1)
        + (delta**2 / variances).sum(axis=2)
    )
    likelihood = np.logaddexp.reduce(log_prob, axis=1)
    expected_resp = np.exp(log_prob - likelihood[:, None])
    count = expected_resp.sum(axis=0) + 10 * np.finfo(dtype).eps
    expected_means = expected_resp.T @ data / count[:, None]
    expected_cov = np.array(
        [
            ((data - mean).T * expected_resp[:, cluster])
            @ (data - mean)
            / count[cluster]
            + np.eye(d) * 1e-4
            for cluster, mean in enumerate(expected_means)
        ]
    )
    caller = cp.cuda.get_current_stream()
    work = cp.cuda.Stream(non_blocking=True)
    with work:
        x = cp.asarray(data)
        initial_w = cp.asarray(initial_weights)
        initial_m = cp.asarray(initial_means)
        cov_work = cp.asarray(np.array([np.diag(v) for v in variances]))
        precision = cp.empty_like(cov_work)
        logdet = cp.empty(k, dtype=dtype)
        info = cp.empty(k, dtype=cp.int32)
        a_table = cp.asarray(
            [
                cov_work.data.ptr + i * d * d * np.dtype(dtype).itemsize
                for i in range(k)
            ],
            dtype=cp.uint64,
        )
        b_table = cp.asarray(
            [
                precision.data.ptr + i * d * d * np.dtype(dtype).itemsize
                for i in range(k)
            ],
            dtype=cp.uint64,
        )
        centered = cp.empty_like(x)
        transformed = cp.empty_like(x)
        logp = cp.empty((n, k), dtype=dtype)
        resp = cp.empty_like(logp)
        ll = cp.empty(n, dtype=dtype)
        ones = cp.ones(n, dtype=dtype)
        weights = cp.empty(k, dtype=dtype)
        means = cp.empty((k, d), dtype=dtype)
        cov = cp.empty((k, d, d), dtype=dtype)
        counts = cp.empty(k, dtype=dtype)
        numerators = cp.empty_like(means)
        handle = cp.cuda.device.get_cublas_handle()
        solver = cp.cuda.device.get_cusolver_handle()
    _gmm_cuda.precision_cholesky_full(
        cov_work,
        precision,
        logdet,
        info,
        dA=a_table.data.ptr,
        dB=b_table.data.ptr,
        d=d,
        K=k,
        stream=work.ptr,
        cublas_handle=handle,
        cusolver_handle=solver,
    )
    _gmm_cuda.e_step_cublas(
        x,
        initial_w,
        initial_m,
        precision,
        logdet,
        centered,
        transformed,
        logp,
        resp,
        ll,
        n=n,
        d=d,
        K=k,
        stream=work.ptr,
        handle=handle,
    )
    _gmm_cuda.m_step(
        resp,
        x,
        ones,
        weights,
        means,
        cov,
        counts,
        numerators,
        centered,
        n=n,
        d=d,
        K=k,
        reg_covar=1e-4,
        stream=work.ptr,
        handle=handle,
    )
    assert cp.cuda.get_current_stream().ptr == caller.ptr
    work.synchronize()
    tolerance = 5e-5 if dtype == np.float32 else 1e-10
    for actual, expected in [
        (resp, expected_resp),
        (ll, likelihood),
        (weights, count / n),
        (means, expected_means),
        (cov, expected_cov),
    ]:
        np.testing.assert_allclose(
            cp.asnumpy(actual), expected, atol=tolerance, rtol=tolerance
        )
    cp.testing.assert_array_equal(info, 0)
