"""Full-covariance GMM for the CellCharter niche flavor.

The public behavior mirrors :class:`sklearn.mixture.GaussianMixture` with
``covariance_type="full"``. EM is CUDA-only: CuPy is used for array handling and
the precision-Cholesky factorization, while E-step and M-step work is delegated
to the nanobind/CUDA extension.
"""

from __future__ import annotations

from typing import Literal

import cupy as cp
import numpy as np
from cupyx.scipy.linalg import solve_triangular

from rapids_singlecell._cuda import _gmm_cuda as _gc

_GMMInit = Literal["kmeans", "random_from_data", "sklearn_kmeans"]
_EStepRoute = Literal["fused", "cublas"]

_KMEANS_MAX_ITER = 100
_SKLEARN_SEEDED_KMEANS_MAX_ITER = 300

# Fused kernels cover the CellCharter regime. Wider float32 embeddings and
# float64 embeddings above 64 dimensions use the cuBLAS E-step.
_CUDA_FUSED_E_STEP_MAX_D = 512
_CUDA_FUSED_FLOAT64_MAX_D = 64
_CUDA_CUBLAS_E_STEP_MIN_D = 257


def _allocate_m_step_workspace(X: cp.ndarray, K: int) -> dict[str, cp.ndarray]:
    n, d = X.shape
    return {
        "ones": cp.ones(n, dtype=X.dtype),
        "effective_counts": cp.empty(K, dtype=X.dtype),
        "weighted_sums": cp.empty((K, d), dtype=X.dtype),
        "centered": cp.empty_like(X),
    }


def _allocate_em_workspace(
    X: cp.ndarray, K: int, e_step_route: _EStepRoute
) -> dict[str, cp.ndarray]:
    n = X.shape[0]
    workspace = {
        "log_prob": cp.empty((n, K), dtype=X.dtype),
        "responsibilities": cp.empty((n, K), dtype=X.dtype),
        "ll_per_cell": cp.empty(n, dtype=X.dtype),
        **_allocate_m_step_workspace(X, K),
    }
    if e_step_route == "cublas":
        workspace["e_step_y"] = cp.empty_like(X)
    return workspace


def _e_step(
    X: cp.ndarray,
    weights: cp.ndarray,
    means: cp.ndarray,
    prec_chol: cp.ndarray,
    log_det_half: cp.ndarray,
    *,
    log_prob: cp.ndarray,
    responsibilities: cp.ndarray,
    ll_per_cell: cp.ndarray,
    centered: cp.ndarray,
    e_step_y: cp.ndarray | None,
    e_step_route: _EStepRoute,
    stream: int,
    handle: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    if e_step_route == "cublas":
        return _e_step_cublas(
            X,
            weights,
            means,
            prec_chol,
            log_det_half,
            centered=centered,
            e_step_y=e_step_y,
            log_prob=log_prob,
            responsibilities=responsibilities,
            ll_per_cell=ll_per_cell,
            stream=stream,
            handle=handle,
        )
    return _e_step_fused(
        X,
        weights,
        means,
        prec_chol,
        log_det_half,
        log_prob=log_prob,
        responsibilities=responsibilities,
        ll_per_cell=ll_per_cell,
        stream=stream,
    )


def _e_step_fused(
    X: cp.ndarray,
    weights: cp.ndarray,
    means: cp.ndarray,
    prec_chol: cp.ndarray,
    log_det_half: cp.ndarray,
    *,
    log_prob: cp.ndarray,
    responsibilities: cp.ndarray,
    ll_per_cell: cp.ndarray,
    stream: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    n, d = X.shape
    K = int(weights.shape[0])
    _gc.e_step(
        X,
        weights,
        means,
        prec_chol,
        log_det_half,
        log_prob,
        responsibilities,
        ll_per_cell,
        n=int(n),
        d=int(d),
        K=K,
        stream=stream,
    )
    return responsibilities, ll_per_cell.mean()


def _e_step_cublas(
    X: cp.ndarray,
    weights: cp.ndarray,
    means: cp.ndarray,
    prec_chol: cp.ndarray,
    log_det_half: cp.ndarray,
    *,
    centered: cp.ndarray,
    e_step_y: cp.ndarray | None,
    log_prob: cp.ndarray,
    responsibilities: cp.ndarray,
    ll_per_cell: cp.ndarray,
    stream: int,
    handle: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    n, d = X.shape
    K = int(weights.shape[0])
    _gc.e_step_cublas(
        X,
        weights,
        means,
        prec_chol,
        log_det_half,
        centered,
        e_step_y,
        log_prob,
        responsibilities,
        ll_per_cell,
        n=int(n),
        d=int(d),
        K=K,
        stream=stream,
        handle=handle,
    )
    return responsibilities, ll_per_cell.mean()


def gmm_fit_predict(
    X: cp.ndarray,
    n_components: int,
    *,
    random_state: int = 0,
    max_iter: int = 100,
    tol: float = 1e-3,
    reg_covar: float = 1e-6,
    init: _GMMInit = "kmeans",
    kmeans_n_init: int = 1,
) -> cp.ndarray:
    """Fit a full-covariance GMM and return cluster labels.

    Parameters
    ----------
    X
        GPU matrix with observations in rows and features in columns.
    n_components
        Number of mixture components.
    random_state
        Seed used by the selected initialization strategy.
    max_iter
        Maximum number of EM iterations.
    tol
        Convergence threshold on the mean log-likelihood change.
    reg_covar
        Non-negative regularization added to each covariance diagonal.
    init
        Initialization strategy. ``"kmeans"`` uses native cuML KMeans,
        ``"random_from_data"`` matches sklearn/Squidpy random-from-data, and
        ``"sklearn_kmeans"`` uses sklearn k-means++ seeding followed by cuML
        KMeans.
    kmeans_n_init
        Number of cuML KMeans restarts for ``init="kmeans"``.
    """
    K = int(n_components)
    if K < 1:
        raise ValueError("n_components must be >= 1.")
    if int(kmeans_n_init) < 1:
        raise ValueError("kmeans_n_init must be >= 1.")

    X = cp.ascontiguousarray(X)
    weights, means, covariances = _initialize_parameters(
        X,
        K,
        init=init,
        random_state=random_state,
        reg_covar=reg_covar,
        kmeans_n_init=int(kmeans_n_init),
    )
    responsibilities = _run_em(
        X,
        weights,
        means,
        covariances,
        max_iter=int(max_iter),
        tol=float(tol),
        reg_covar=float(reg_covar),
    )
    return responsibilities.argmax(axis=1).astype(cp.int32)


def _run_em(
    X: cp.ndarray,
    weights: cp.ndarray,
    means: cp.ndarray,
    covariances: cp.ndarray,
    *,
    max_iter: int,
    tol: float,
    reg_covar: float,
) -> cp.ndarray:
    n, d = X.shape
    K = int(weights.shape[0])
    stream = cp.cuda.get_current_stream().ptr
    handle = cp.cuda.device.get_cublas_handle()
    e_step_route = _choose_e_step(int(d), X.dtype)
    workspace = _allocate_em_workspace(X, K, e_step_route)

    prec_chol, log_det_half = _precision_cholesky(covariances)
    previous_ll = -np.inf

    for _ in range(max_iter):
        responsibilities, mean_ll = _e_step(
            X,
            weights,
            means,
            prec_chol,
            log_det_half,
            log_prob=workspace["log_prob"],
            responsibilities=workspace["responsibilities"],
            ll_per_cell=workspace["ll_per_cell"],
            centered=workspace["centered"],
            e_step_y=workspace.get("e_step_y"),
            e_step_route=e_step_route,
            stream=stream,
            handle=handle,
        )
        mean_ll = float(mean_ll)
        if abs(mean_ll - previous_ll) < tol:
            return responsibilities

        previous_ll = mean_ll
        _m_step(
            X,
            responsibilities,
            weights,
            means,
            covariances,
            reg_covar=reg_covar,
            ones=workspace["ones"],
            effective_counts=workspace["effective_counts"],
            weighted_sums=workspace["weighted_sums"],
            centered=workspace["centered"],
            stream=stream,
            handle=handle,
        )
        prec_chol, log_det_half = _precision_cholesky(covariances)

    responsibilities, _ = _e_step(
        X,
        weights,
        means,
        prec_chol,
        log_det_half,
        log_prob=workspace["log_prob"],
        responsibilities=workspace["responsibilities"],
        ll_per_cell=workspace["ll_per_cell"],
        centered=workspace["centered"],
        e_step_y=workspace.get("e_step_y"),
        e_step_route=e_step_route,
        stream=stream,
        handle=handle,
    )
    return responsibilities


def _initialize_parameters(
    X: cp.ndarray,
    K: int,
    *,
    init: str,
    random_state: int,
    reg_covar: float,
    kmeans_n_init: int,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    if init == "random_from_data":
        return _random_from_data_init(X, K, random_state, reg_covar)

    if init not in ("kmeans", "sklearn_kmeans"):
        raise ValueError(
            "init must be 'kmeans', 'random_from_data', or "
            f"'sklearn_kmeans', got {init!r}"
        )

    labels, centers = _fit_kmeans(
        X,
        K,
        init=init,
        random_state=random_state,
        kmeans_n_init=kmeans_n_init,
    )
    return _parameters_from_labels(X, labels, centers, reg_covar)


def _random_from_data_init(
    X: cp.ndarray,
    K: int,
    random_state: int,
    reg_covar: float,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    n, d = X.shape
    rng = np.random.RandomState(random_state)
    idx = cp.asarray(rng.choice(n, size=K, replace=False))
    eye_reg = reg_covar * cp.eye(d, dtype=X.dtype)
    return (
        cp.full(K, 1.0 / K, dtype=X.dtype),
        X[idx].copy(),
        cp.broadcast_to(eye_reg, (K, d, d)).copy(),
    )


def _fit_kmeans(
    X: cp.ndarray,
    K: int,
    *,
    init: str,
    random_state: int,
    kmeans_n_init: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    from cuml.cluster import KMeans

    kwargs = {}
    if init == "sklearn_kmeans":
        from sklearn.cluster import kmeans_plusplus

        centers, _ = kmeans_plusplus(cp.asnumpy(X), K, random_state=random_state)
        kwargs["init"] = cp.asarray(centers, dtype=X.dtype)
        kmeans_n_init = 1
        max_iter = _SKLEARN_SEEDED_KMEANS_MAX_ITER
    else:
        max_iter = _KMEANS_MAX_ITER

    km = KMeans(
        n_clusters=K,
        random_state=random_state,
        n_init=int(kmeans_n_init),
        max_iter=max_iter,
        **kwargs,
    )
    km.fit(X)
    return (
        cp.asarray(km.labels_).astype(cp.int64, copy=False),
        cp.asarray(km.cluster_centers_, dtype=X.dtype),
    )


def _parameters_from_labels(
    X: cp.ndarray,
    labels: cp.ndarray,
    means_init: cp.ndarray,
    reg_covar: float,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    n, d = X.shape
    K = int(means_init.shape[0])
    weights = cp.empty(K, dtype=X.dtype)
    means = cp.empty((K, d), dtype=X.dtype)
    covariances = cp.empty((K, d, d), dtype=X.dtype)
    responsibilities = cp.zeros((n, K), dtype=X.dtype)
    workspace = _allocate_m_step_workspace(X, K)
    responsibilities[cp.arange(n), labels] = X.dtype.type(1.0)
    _m_step(
        X,
        responsibilities,
        weights,
        means,
        covariances,
        reg_covar=reg_covar,
        ones=workspace["ones"],
        effective_counts=workspace["effective_counts"],
        weighted_sums=workspace["weighted_sums"],
        centered=workspace["centered"],
        stream=cp.cuda.get_current_stream().ptr,
        handle=cp.cuda.device.get_cublas_handle(),
    )
    _restore_empty_components(
        weights,
        means,
        covariances,
        labels,
        means_init,
        n=n,
        reg_covar=reg_covar,
    )
    return weights, means, covariances


def _restore_empty_components(
    weights: cp.ndarray,
    means: cp.ndarray,
    covariances: cp.ndarray,
    labels: cp.ndarray,
    means_init: cp.ndarray,
    *,
    n: int,
    reg_covar: float,
) -> None:
    """Repair empty cuML KMeans components before EM starts.

    sklearn's GMM init estimates parameters from hard KMeans responsibilities
    and adds ``10 * eps`` to component counts, relying on sklearn KMeans to
    avoid empty final labels in normal cases. cuML can still hand back an empty
    component, so keep its center, give it a tiny finite weight, and use the
    regularized identity covariance instead of letting the M-step create a
    zero-mean component from an empty responsibility column.
    """
    K, d = means_init.shape
    counts = cp.bincount(labels, minlength=int(K)).astype(means_init.dtype, copy=False)
    empty = counts == 0
    eye_reg = reg_covar * cp.eye(d, dtype=means_init.dtype)

    weights[...] = cp.where(empty, means_init.dtype.type(1.0 / n), counts / n)
    means[...] = cp.where(empty[:, None], means_init, means)
    covariances[...] = cp.where(
        empty[:, None, None],
        cp.broadcast_to(eye_reg, covariances.shape),
        covariances,
    )


def _precision_cholesky(covariances: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """Return sklearn-oriented precision Cholesky without forming an inverse."""
    cov_chol = cp.linalg.cholesky(covariances)
    eye = cp.broadcast_to(
        cp.eye(covariances.shape[-1], dtype=covariances.dtype),
        covariances.shape,
    )
    cov_chol_inv = solve_triangular(cov_chol, eye, lower=True)
    return (
        cp.ascontiguousarray(cov_chol_inv.transpose(0, 2, 1)),
        -cp.sum(
            cp.log(cp.diagonal(cov_chol, axis1=1, axis2=2)),
            axis=1,
        ),
    )


def _m_step(
    X: cp.ndarray,
    responsibilities: cp.ndarray,
    weights: cp.ndarray,
    means: cp.ndarray,
    covariances: cp.ndarray,
    *,
    reg_covar: float,
    ones: cp.ndarray,
    effective_counts: cp.ndarray,
    weighted_sums: cp.ndarray,
    centered: cp.ndarray,
    stream: int,
    handle: int,
) -> None:
    n, d = X.shape
    K = int(weights.shape[0])

    _gc.m_step(
        responsibilities,
        X,
        ones,
        weights,
        means,
        covariances,
        effective_counts,
        weighted_sums,
        centered,
        n=int(n),
        d=int(d),
        K=K,
        reg_covar=float(reg_covar),
        stream=stream,
        handle=handle,
    )


def _choose_e_step(d: int, dtype) -> _EStepRoute:
    """Select the CUDA E-step implementation for a feature width and dtype."""
    dtype = np.dtype(dtype)
    if dtype == np.dtype("float32"):
        return "cublas" if d >= _CUDA_CUBLAS_E_STEP_MIN_D else "fused"
    if dtype == np.dtype("float64"):
        return "cublas" if d > _CUDA_FUSED_FLOAT64_MAX_D else "fused"
    return "cublas" if d >= _CUDA_CUBLAS_E_STEP_MIN_D else "fused"
