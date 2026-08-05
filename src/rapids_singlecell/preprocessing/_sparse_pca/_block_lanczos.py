"""
GPU-accelerated Randomized SVD for sparse matrices.

This implements randomized SVD using the GPU-optimized approach from:
    Tomás, Quintana-Ortí, Anzt (2024)
    "Fast Truncated SVD of Sparse and Dense Matrices on Graphics Processors"
    https://arxiv.org/abs/2403.06218

Key optimizations:
- CholeskyQR2 for orthogonalization (3x faster than standard QR on GPU)
- Block power iteration for efficient GPU execution
"""

from __future__ import annotations

import cupy as cp
import cupyx.scipy.linalg as cpla
import cupyx.scipy.sparse as cpsparse


def _matvec(A, V):
    """Matrix-vector/matrix product handling sparse and operators."""
    if hasattr(A, "dot") and not isinstance(A, cp.ndarray) and not cpsparse.issparse(A):
        return A.dot(V)
    return A @ V


def _matvec_T(A, V):
    """Transposed matrix product."""
    if hasattr(A, "T"):
        AT = A.T
        if hasattr(AT, "dot") and not isinstance(AT, cp.ndarray):
            return AT.dot(V)
        return AT @ V
    return A.T @ V


def _cholesky_qr2(Q: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """
    CholeskyQR2 orthogonalization (Algorithm 4 from Tomás et al. 2024).

    More numerically stable than single CholeskyQR, and GPU-efficient
    as it uses GEMM operations (3x faster than standard QR).

    Parameters
    ----------
    Q
        Matrix to orthogonalize of shape (m, k).

    Returns
    -------
    Q
        Orthonormalized matrix of shape (m, k).
    R
        Upper triangular factor of shape (k, k).
    """
    # Tolerance for detecting near-singular matrices
    eps = cp.finfo(Q.dtype).eps
    tol = eps * max(Q.shape) * 100

    # First pass
    W = Q.T @ Q  # k x k
    try:
        L = cp.linalg.cholesky(W)  # W = L @ L.T
    except cp.linalg.LinAlgError:
        # Fallback to QR if Cholesky fails
        Q, R = cp.linalg.qr(Q, mode="reduced")
        return Q, R

    # Check for NaN or near-singularity (small diagonal in L)
    L_diag = cp.diag(L)
    if cp.any(cp.isnan(L_diag)):
        Q, R = cp.linalg.qr(Q, mode="reduced")
        return Q, R
    min_diag = cp.min(cp.abs(L_diag))
    if min_diag < tol:
        Q, R = cp.linalg.qr(Q, mode="reduced")
        return Q, R

    Q = cpla.solve_triangular(L, Q.T, lower=True).T  # Q = Q @ L^{-T}

    # Second pass for numerical stability
    W_bar = Q.T @ Q
    try:
        L_bar = cp.linalg.cholesky(W_bar)
    except cp.linalg.LinAlgError:
        Q, R = cp.linalg.qr(Q, mode="reduced")
        return Q, R

    # Check for near-singularity in second pass
    min_diag_bar = cp.min(cp.abs(cp.diag(L_bar)))
    if min_diag_bar < tol:
        Q, R = cp.linalg.qr(Q, mode="reduced")
        return Q, R

    Q = cpla.solve_triangular(L_bar, Q.T, lower=True).T

    # R = L.T @ L_bar.T
    R = L.T @ L_bar.T
    return Q, R


def randomized_svd(
    A,
    k: int,
    *,
    n_oversamples: int = 10,
    n_iter: int = 2,
    random_state: int | None = 0,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Compute truncated SVD using randomized algorithm with GPU optimizations.

    Implements randomized SVD (Halko et al. 2009) with CholeskyQR2
    orthogonalization from Tomás et al. (2024) for efficient GPU execution.

    Parameters
    ----------
    A
        Matrix to decompose of shape (m, n). Can be sparse matrix,
        dense array, or linear operator.
    k
        Number of singular values/vectors to compute.
    n_oversamples
        Number of extra random vectors for better approximation.
        Total subspace dimension is k + n_oversamples.
    n_iter
        Number of power iterations. More iterations improve accuracy
        for matrices with slowly decaying singular values.
    random_state
        Random seed for reproducibility.

    Returns
    -------
    U
        Left singular vectors of shape (m, k).
    s
        Singular values of shape (k,) in descending order.
    Vt
        Right singular vectors of shape (k, n).

    References
    ----------
    .. [1] Halko, Martinsson, Tropp (2009) "Finding structure with randomness"
           https://arxiv.org/abs/0909.4061
    .. [2] Tomás, Quintana-Ortí, Anzt (2024) "Fast Truncated SVD of Sparse
           and Dense Matrices on Graphics Processors"
           https://arxiv.org/abs/2403.06218
    """
    m, n = A.shape
    dtype = A.dtype
    rng = cp.random.RandomState(random_state if random_state is not None else 0)

    # Block size = k + oversamples, but cannot exceed matrix dimensions
    block_size = min(k + n_oversamples, m, n)

    # Generate random starting matrix
    Omega = rng.standard_normal(size=(n, block_size)).astype(dtype)

    # Build subspace via block power iteration
    # Y = A @ Omega
    Y = _matvec(A, Omega)
    Y, _ = _cholesky_qr2(Y)

    # Power iterations: Y = (A @ A.T)^n_iter @ Y
    for _ in range(n_iter):
        # Z = A.T @ Y
        Z = _matvec_T(A, Y)
        Z, _ = _cholesky_qr2(Z)
        # Y = A @ Z
        Y = _matvec(A, Z)
        Y, _ = _cholesky_qr2(Y)

    # Y now spans approximate range of A
    # Compute B = Y.T @ A
    Q = Y
    B = _matvec_T(A, Q).T  # B = Q.T @ A = (A.T @ Q).T

    # SVD of the small matrix B
    Uhat, s, Vt = cp.linalg.svd(B, full_matrices=False)

    # Recover U
    U = Q @ Uhat

    # Truncate to k components
    U = U[:, :k]
    s = s[:k]
    Vt = Vt[:k, :]

    # Sign correction for deterministic output
    max_abs_cols = cp.argmax(cp.abs(U), axis=0)
    signs = cp.sign(U[max_abs_cols, cp.arange(k)])
    signs[signs == 0] = 1
    U *= signs
    Vt *= signs[:, cp.newaxis]

    return U, s, Vt
