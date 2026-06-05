from __future__ import annotations

import inspect

import cupy as cp
import numpy as np
import pytest
from cupyx.scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.mixture import GaussianMixture

from rapids_singlecell.squidpy_gpu._gmm import (
    _choose_e_step,
    _e_step,
    _e_step_cublas,
    _e_step_fused,
    _m_step,
    _precision_cholesky,
    gmm_fit_predict,
)


def _well_separated(n_per: int, K: int, d: int, sep: float, seed: int):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=sep, size=(K, d))
    X = np.vstack(
        [rng.normal(loc=c, scale=1.0, size=(n_per, d)) for c in centers]
    ).astype(np.float32)
    y = np.repeat(np.arange(K), n_per)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def _pca_like_mixture(n_per: int, K: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=5.0, size=(K, d))
    rows = []
    for k in range(K):
        # A compact low-rank perturbation of the identity gives each synthetic
        # cell state its own correlated PCA-space geometry.
        factors = rng.normal(scale=0.3 + 0.05 * k, size=(d, 3))
        cov = np.eye(d) * (0.35 + 0.03 * k) + factors @ factors.T
        rows.append(rng.multivariate_normal(centers[k], cov, size=n_per))
    X = np.vstack(rows).astype(np.float32)
    return np.ascontiguousarray(X[rng.permutation(len(X))])


_LOG_2PI = float(np.log(2.0 * np.pi))


def _reference_e_step(
    X: cp.ndarray,
    weights: cp.ndarray,
    means: cp.ndarray,
    prec_chol: cp.ndarray,
    log_det_half: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    n, d = X.shape
    K = means.shape[0]
    log_prob = cp.empty((n, K), dtype=X.dtype)
    half_d_log2pi = X.dtype.type(0.5 * d * _LOG_2PI)
    for k in range(K):
        y = (X - means[k]) @ prec_chol[k]
        mahal = cp.einsum("ij,ij->i", y, y)
        log_prob[:, k] = (
            -X.dtype.type(0.5) * mahal
            + log_det_half[k]
            - half_d_log2pi
            + cp.log(weights[k])
        )

    log_total = logsumexp(log_prob, axis=1, keepdims=True)
    return cp.exp(log_prob - log_total), log_total.mean()


def _reference_m_step(
    X: cp.ndarray,
    resp: cp.ndarray,
    reg_covar: float,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    n, d = X.shape
    K = resp.shape[1]
    N_k = resp.sum(axis=0) + 10.0 * cp.finfo(X.dtype).eps
    weights = N_k / n
    means = (resp.T @ X) / N_k[:, None]
    covariances = cp.empty((K, d, d), dtype=X.dtype)
    eye_reg = reg_covar * cp.eye(d, dtype=X.dtype)
    for k in range(K):
        diff = X - means[k]
        covariances[k] = ((resp[:, k : k + 1] * diff).T @ diff) / N_k[k] + eye_reg
    return weights, means, covariances


def _e_step_buffers(X: cp.ndarray, K: int, route: str):
    n = X.shape[0]
    return (
        cp.empty((n, K), dtype=X.dtype),
        cp.empty((n, K), dtype=X.dtype),
        cp.empty(n, dtype=X.dtype),
        cp.empty_like(X),
        cp.empty_like(X) if route == "cublas" else None,
    )


def _cuda_e_step(
    X: cp.ndarray,
    weights: cp.ndarray,
    means: cp.ndarray,
    prec_chol: cp.ndarray,
    log_det_half: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    n, d = X.shape
    K = int(means.shape[0])
    e_step_route = _choose_e_step(d, X.dtype)
    log_prob, responsibilities, ll_per_cell, centered, e_step_y = _e_step_buffers(
        X, K, e_step_route
    )
    return _e_step(
        X,
        weights,
        means,
        prec_chol,
        log_det_half,
        log_prob=log_prob,
        responsibilities=responsibilities,
        ll_per_cell=ll_per_cell,
        centered=centered,
        e_step_y=e_step_y,
        e_step_route=e_step_route,
        stream=cp.cuda.get_current_stream().ptr,
        handle=cp.cuda.device.get_cublas_handle(),
    )


def _cuda_m_step(
    X: cp.ndarray,
    resp: cp.ndarray,
    reg_covar: float,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    K = resp.shape[1]
    weights = cp.empty(K, dtype=X.dtype)
    means = cp.empty((K, X.shape[1]), dtype=X.dtype)
    covariances = cp.empty((K, X.shape[1], X.shape[1]), dtype=X.dtype)
    _m_step(
        X,
        resp,
        weights,
        means,
        covariances,
        reg_covar=reg_covar,
        ones=cp.ones(X.shape[0], dtype=X.dtype),
        effective_counts=cp.empty(K, dtype=X.dtype),
        weighted_sums=cp.empty((K, X.shape[1]), dtype=X.dtype),
        centered=cp.empty_like(X),
        stream=cp.cuda.get_current_stream().ptr,
        handle=cp.cuda.device.get_cublas_handle(),
    )
    return weights, means, covariances


def test_kmeans_init_recovers_well_separated_clusters():
    """kmeans init should land at near-truth on well-separated data."""
    X_np, y = _well_separated(n_per=300, K=5, d=20, sep=6.0, seed=0)
    labels = cp.asnumpy(
        gmm_fit_predict(cp.asarray(X_np), n_components=5, random_state=0, init="kmeans")
    )
    assert ARI(y, labels) >= 0.95


def test_full_cov_gmm_matches_sklearn_on_singlecell_embedding():
    X = _pca_like_mixture(n_per=250, K=5, d=16, seed=7)
    sk_labels = GaussianMixture(
        n_components=5,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        random_state=0,
    ).fit_predict(X)
    rsc_labels = cp.asnumpy(
        gmm_fit_predict(
            cp.asarray(X),
            n_components=5,
            random_state=0,
            init="kmeans",
            kmeans_n_init=1,
        )
    )

    assert ARI(sk_labels, rsc_labels) >= 0.99


def test_random_from_data_init_runs():
    """random_from_data may land at a worse local optimum than kmeans, but should
    still produce a non-trivial partition on well-separated data."""
    X_np, y = _well_separated(n_per=300, K=5, d=20, sep=6.0, seed=0)
    labels = cp.asnumpy(
        gmm_fit_predict(
            cp.asarray(X_np), n_components=5, random_state=0, init="random_from_data"
        )
    )
    assert ARI(y, labels) >= 0.35
    assert len(set(labels.tolist())) >= 2


def test_output_shape_and_dtype():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 8)).astype(np.float32)
    labels = gmm_fit_predict(cp.asarray(X), n_components=4, random_state=0)
    assert labels.shape == (500,)
    assert labels.dtype == cp.int32
    assert int(labels.max()) < 4
    assert int(labels.min()) >= 0


def test_non_contiguous_input_is_normalized_at_public_boundary():
    rng = np.random.default_rng(2)
    X = cp.asarray(rng.standard_normal((6, 240)).astype(np.float32)).T

    assert not X.flags.c_contiguous

    labels = gmm_fit_predict(
        X,
        n_components=3,
        random_state=0,
        max_iter=2,
        init="random_from_data",
    )

    assert labels.shape == (240,)
    assert labels.dtype == cp.int32


@pytest.mark.parametrize("init", ["kmeans", "random_from_data", "sklearn_kmeans"])
def test_determinism_same_seed(init):
    rng = np.random.default_rng(1)
    X = cp.asarray(rng.standard_normal((800, 10)).astype(np.float32))
    a = cp.asnumpy(gmm_fit_predict(X, n_components=5, random_state=42, init=init))
    b = cp.asnumpy(gmm_fit_predict(X, n_components=5, random_state=42, init=init))
    np.testing.assert_array_equal(a, b)


def test_invalid_init_raises():
    X = cp.asarray(np.zeros((100, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="init"):
        gmm_fit_predict(X, n_components=3, init="bogus")


def test_backend_parameter_is_not_exposed():
    assert "backend" not in inspect.signature(gmm_fit_predict).parameters


def test_invalid_kmeans_n_init_raises():
    X = cp.asarray(np.zeros((100, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="kmeans_n_init"):
        gmm_fit_predict(X, n_components=3, kmeans_n_init=0)


def test_invalid_n_components_raises():
    X = cp.asarray(np.zeros((100, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="n_components"):
        gmm_fit_predict(X, n_components=0)


@pytest.mark.parametrize(
    ("d", "dtype", "route"),
    [
        (16, cp.float32, "fused"),
        (32, cp.float32, "fused"),
        (50, cp.float32, "fused"),
        (64, cp.float32, "fused"),
        (80, cp.float32, "fused"),
        (96, cp.float32, "fused"),
        (128, cp.float32, "fused"),
        (256, cp.float32, "fused"),
        (384, cp.float32, "cublas"),
        (512, cp.float32, "cublas"),
        (768, cp.float32, "cublas"),
        (2000, cp.float32, "cublas"),
        (64, cp.float64, "fused"),
        (128, cp.float64, "cublas"),
        (512, cp.float64, "cublas"),
        (2000, cp.float64, "cublas"),
    ],
)
def test_cuda_e_step_routing_uses_cublas_for_high_d_and_wide_float64(d, dtype, route):
    assert _choose_e_step(d, dtype) == route


def test_n_components_one_returns_single_label():
    rng = np.random.default_rng(0)
    X = cp.asarray(rng.standard_normal((200, 4)).astype(np.float32))
    labels = cp.asnumpy(gmm_fit_predict(X, n_components=1, random_state=0))
    assert set(labels.tolist()) == {0}


def test_float64_input_accepted():
    rng = np.random.default_rng(0)
    X = cp.asarray(rng.standard_normal((300, 6)).astype(np.float64))
    labels = gmm_fit_predict(X, n_components=3, random_state=0)
    assert labels.shape == (300,)


def test_cuda_matches_reference_steps():
    rng = cp.random.RandomState(0)
    n, d, K = 40_000, 6, 3  # large enough to exercise the cuBLAS M-step path
    X = rng.standard_normal((n, d), dtype=cp.float32)
    logits = rng.standard_normal((n, K), dtype=cp.float32)
    resp = cp.exp(logits - cp.log(cp.exp(logits).sum(axis=1, keepdims=True)))

    w_c, m_c, cov_c = _reference_m_step(X, resp, 1e-6)
    w_g, m_g, cov_g = _cuda_m_step(X, resp, 1e-6)

    assert cp.max(cp.abs(w_c - w_g)).item() < 1e-5
    assert cp.max(cp.abs(m_c - m_g)).item() < 1e-5
    assert cp.max(cp.abs(cov_c - cov_g)).item() < 1e-4

    weights = cp.full(K, 1 / K, dtype=cp.float32)
    means = rng.standard_normal((K, d), dtype=cp.float32)
    A = rng.standard_normal((K, d, d), dtype=cp.float32)
    cov = A @ A.transpose(0, 2, 1) + cp.eye(d, dtype=cp.float32)[None] * 0.1
    prec_chol, log_det_half = _precision_cholesky(cov)

    r_c, ll_c = _reference_e_step(X, weights, means, prec_chol, log_det_half)
    r_g, ll_g = _cuda_e_step(X, weights, means, prec_chol, log_det_half)
    log_prob, resp, ll_per_cell, _, _ = _e_step_buffers(X, K, "fused")
    r_f, ll_f = _e_step_fused(
        X,
        weights,
        means,
        prec_chol,
        log_det_half,
        log_prob=log_prob,
        responsibilities=resp,
        ll_per_cell=ll_per_cell,
        stream=cp.cuda.get_current_stream().ptr,
    )

    assert cp.max(cp.abs(r_c - r_g)).item() < 1e-4
    assert cp.abs(ll_c - ll_g).item() < 1e-4
    assert cp.max(cp.abs(r_c - r_f)).item() < 1e-4
    assert cp.abs(ll_c - ll_f).item() < 1e-4


def test_cuda_large_e_step_matches_reference_for_large_feature_count():
    rng = cp.random.RandomState(2)
    n, d, K = 2048, 96, 4
    X = rng.standard_normal((n, d), dtype=cp.float32)
    weights = cp.asarray([0.15, 0.2, 0.3, 0.35], dtype=cp.float32)
    means = rng.standard_normal((K, d), dtype=cp.float32)
    A = rng.standard_normal((K, d, d), dtype=cp.float32)
    cov = (A @ A.transpose(0, 2, 1)) / d + cp.eye(d, dtype=cp.float32)[None] * 0.5
    prec_chol, log_det_half = _precision_cholesky(cov)

    r_c, ll_c = _reference_e_step(X, weights, means, prec_chol, log_det_half)
    r_g, ll_g = _cuda_e_step(X, weights, means, prec_chol, log_det_half)

    assert cp.max(cp.abs(r_c - r_g)).item() < 5e-4
    assert cp.abs(ll_c - ll_g).item() < 5e-4


def test_cuda_512_e_step_matches_reference_for_cublas_route():
    rng = cp.random.RandomState(5)
    n, d, K = 384, 512, 3
    X = rng.standard_normal((n, d), dtype=cp.float32)
    weights = cp.asarray([0.2, 0.3, 0.5], dtype=cp.float32)
    means = rng.standard_normal((K, d), dtype=cp.float32)
    A = rng.standard_normal((K, d, d), dtype=cp.float32)
    cov = (A @ A.transpose(0, 2, 1)) / d + cp.eye(d, dtype=cp.float32)[None] * 0.5
    prec_chol, log_det_half = _precision_cholesky(cov)

    r_c, ll_c = _reference_e_step(X, weights, means, prec_chol, log_det_half)
    r_g, ll_g = _cuda_e_step(X, weights, means, prec_chol, log_det_half)
    log_prob, resp, ll_per_cell, centered, e_step_y = _e_step_buffers(X, K, "cublas")
    r_b, ll_b = _e_step_cublas(
        X,
        weights,
        means,
        prec_chol,
        log_det_half,
        centered=centered,
        e_step_y=e_step_y,
        log_prob=log_prob,
        responsibilities=resp,
        ll_per_cell=ll_per_cell,
        stream=cp.cuda.get_current_stream().ptr,
        handle=cp.cuda.device.get_cublas_handle(),
    )

    assert cp.max(cp.abs(r_c - r_g)).item() < 1e-3
    assert cp.abs(ll_c - ll_g).item() < 1e-3
    assert cp.max(cp.abs(r_c - r_b)).item() < 1e-3
    assert cp.abs(ll_c - ll_b).item() < 1e-3


def test_cuda_768_e_step_uses_cublas_route():
    rng = cp.random.RandomState(8)
    n, d, K = 64, 768, 2
    X = rng.standard_normal((n, d), dtype=cp.float32)
    weights = cp.asarray([0.45, 0.55], dtype=cp.float32)
    means = rng.standard_normal((K, d), dtype=cp.float32)
    eye = cp.eye(d, dtype=cp.float32)
    cov = cp.stack((eye * 1.5, eye * 2.0))
    prec_chol, log_det_half = _precision_cholesky(cov)

    log_prob, resp, ll_per_cell, centered, e_step_y = _e_step_buffers(X, K, "cublas")
    stream = cp.cuda.get_current_stream().ptr
    handle = cp.cuda.device.get_cublas_handle()
    r_c, ll_c = _reference_e_step(X, weights, means, prec_chol, log_det_half)
    r_g, ll_g = _e_step(
        X,
        weights,
        means,
        prec_chol,
        log_det_half,
        log_prob=log_prob,
        responsibilities=resp,
        ll_per_cell=ll_per_cell,
        centered=centered,
        e_step_y=e_step_y,
        e_step_route="cublas",
        stream=stream,
        handle=handle,
    )
    log_prob_b, resp_b, ll_per_cell_b, centered_b, e_step_y_b = _e_step_buffers(
        X, K, "cublas"
    )
    r_b, ll_b = _e_step_cublas(
        X,
        weights,
        means,
        prec_chol,
        log_det_half,
        centered=centered_b,
        e_step_y=e_step_y_b,
        log_prob=log_prob_b,
        responsibilities=resp_b,
        ll_per_cell=ll_per_cell_b,
        stream=stream,
        handle=handle,
    )

    assert _choose_e_step(d, X.dtype) == "cublas"
    assert cp.max(cp.abs(r_c - r_g)).item() < 1e-3
    assert cp.abs(ll_c - ll_g).item() < 1e-3
    assert cp.max(cp.abs(r_c - r_b)).item() < 1e-3
    assert cp.abs(ll_c - ll_b).item() < 1e-3


def test_cuda_float64_wide_e_step_uses_cublas_route():
    rng = cp.random.RandomState(7)
    n, d, K = 256, 128, 3
    X = rng.standard_normal((n, d), dtype=cp.float64)
    weights = cp.asarray([0.2, 0.3, 0.5], dtype=cp.float64)
    means = rng.standard_normal((K, d), dtype=cp.float64)
    A = rng.standard_normal((K, d, d), dtype=cp.float64)
    cov = (A @ A.transpose(0, 2, 1)) / d + cp.eye(d, dtype=cp.float64)[None] * 0.5
    prec_chol, log_det_half = _precision_cholesky(cov)

    route = _choose_e_step(d, X.dtype)
    log_prob, resp, ll_per_cell, centered, e_step_y = _e_step_buffers(X, K, route)
    r_c, ll_c = _reference_e_step(X, weights, means, prec_chol, log_det_half)
    r_g, ll_g = _e_step(
        X,
        weights,
        means,
        prec_chol,
        log_det_half,
        log_prob=log_prob,
        responsibilities=resp,
        ll_per_cell=ll_per_cell,
        centered=centered,
        e_step_y=e_step_y,
        e_step_route=route,
        stream=cp.cuda.get_current_stream().ptr,
        handle=cp.cuda.device.get_cublas_handle(),
    )

    assert cp.max(cp.abs(r_c - r_g)).item() < 1e-12
    assert cp.abs(ll_c - ll_g).item() < 1e-12


def test_cuda_fixed_e_step_matches_reference_for_medium_regime():
    rng = cp.random.RandomState(4)
    n, d, K = 1024, 16, 8
    X = rng.standard_normal((n, d), dtype=cp.float32)
    weights = cp.full(K, 1 / K, dtype=cp.float32)
    means = rng.standard_normal((K, d), dtype=cp.float32)
    A = rng.standard_normal((K, d, d), dtype=cp.float32)
    cov = (A @ A.transpose(0, 2, 1)) / d + cp.eye(d, dtype=cp.float32)[None] * 0.5
    prec_chol, log_det_half = _precision_cholesky(cov)

    r_c, ll_c = _reference_e_step(X, weights, means, prec_chol, log_det_half)
    r_g, ll_g = _cuda_e_step(X, weights, means, prec_chol, log_det_half)

    assert cp.max(cp.abs(r_c - r_g)).item() < 5e-4
    assert cp.abs(ll_c - ll_g).item() < 5e-4


def test_cuda_fused_e_step_matches_reference_for_50_pc_regime():
    rng = cp.random.RandomState(6)
    n, d, K = 1024, 50, 12
    X = rng.standard_normal((n, d), dtype=cp.float32)
    weights = cp.full(K, 1 / K, dtype=cp.float32)
    means = rng.standard_normal((K, d), dtype=cp.float32)
    A = rng.standard_normal((K, d, d), dtype=cp.float32)
    cov = (A @ A.transpose(0, 2, 1)) / d + cp.eye(d, dtype=cp.float32)[None] * 0.5
    prec_chol, log_det_half = _precision_cholesky(cov)

    log_prob, resp, ll_per_cell, centered, e_step_y = _e_step_buffers(X, K, "fused")
    r_c, ll_c = _reference_e_step(X, weights, means, prec_chol, log_det_half)
    r_d, ll_d = _e_step(
        X,
        weights,
        means,
        prec_chol,
        log_det_half,
        log_prob=log_prob,
        responsibilities=resp,
        ll_per_cell=ll_per_cell,
        centered=centered,
        e_step_y=e_step_y,
        e_step_route="fused",
        stream=cp.cuda.get_current_stream().ptr,
        handle=cp.cuda.device.get_cublas_handle(),
    )

    assert cp.max(cp.abs(r_c - r_d)).item() < 5e-4
    assert cp.abs(ll_c - ll_d).item() < 5e-4


def test_cuda_runs_large_feature_count():
    rng = np.random.default_rng(3)
    X = cp.asarray(rng.standard_normal((360, 80)).astype(np.float32))
    labels = gmm_fit_predict(
        X,
        n_components=3,
        random_state=0,
        max_iter=2,
        reg_covar=1e-2,
        init="random_from_data",
    )

    assert labels.shape == (360,)
    assert labels.dtype == cp.int32
