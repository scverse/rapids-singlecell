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

from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.linalg as cpla
from cupyx.scipy.sparse import issparse
from cupyx.scipy.sparse.linalg import LinearOperator

from rapids_singlecell._utils._random import _seed_from_rng

from ._operators import _svd_flip, mean_centered_operator

if TYPE_CHECKING:
    from rapids_singlecell._utils._random import RNGLike, SeedLike

_valid_pivots = cp.ReductionKernel(
    "T pivot, float64 tol",
    "bool valid",
    "isfinite(pivot) && pivot >= tol",
    "a && b",
    "valid = a",
    "true",
    "svd_valid_pivots",
)


def _cholesky_qr2(Q: cp.ndarray) -> cp.ndarray:
    """Orthonormalize a block with two Cholesky passes, falling back to Householder QR."""
    tol = cp.finfo(Q.dtype).eps * max(Q.shape) * 100
    for _ in range(2):
        try:
            L = cp.linalg.cholesky(Q.T.conj() @ Q)
        except cp.linalg.LinAlgError:
            return cp.linalg.qr(Q, mode="reduced")[0]
        # The Gram matrix may overflow even when the input block is finite.
        if not _valid_pivots(L.diagonal().real, tol):
            return cp.linalg.qr(Q, mode="reduced")[0]
        Q = cpla.solve_triangular(L.conj(), Q.T, lower=True, overwrite_b=True).T
    return Q


def randomized_svd(
    A,
    k: int,
    *,
    n_oversamples: int = 10,
    n_iter: int = 2,
    rng: SeedLike | RNGLike | None = None,
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
    rng
        Random seed or :class:`~numpy.random.Generator` for reproducibility.

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
    if issparse(A):
        A = mean_centered_operator(A)
    AH = A.H if isinstance(A, LinearOperator) else A.T.conj()
    dot = A.dot if hasattr(A, "dot") else A.__matmul__
    dot_h = AH.dot if hasattr(AH, "dot") else AH.__matmul__
    m, n = A.shape
    dtype = A.dtype
    # CuPy generators are seeded, so this is where the host generator
    # collapses into an integer.
    rng = cp.random.default_rng(_seed_from_rng(rng, allow_none=False))

    # Block size = k + oversamples, but cannot exceed matrix dimensions
    block_size = min(k + n_oversamples, m, n)

    # Generate random starting matrix
    Omega = rng.standard_normal(size=(n, block_size)).astype(dtype)

    # Build subspace via block power iteration
    # Y = A @ Omega
    Y = dot(Omega)
    Y = _cholesky_qr2(Y)

    # Power iterations: Y = (A @ A.H)^n_iter @ Y
    for _ in range(n_iter):
        Z = dot_h(Y)
        Z = _cholesky_qr2(Z)
        # Y = A @ Z
        Y = dot(Z)
        Y = _cholesky_qr2(Y)

    # Y now spans the approximate range of A.
    Q = Y
    B = dot_h(Q).T.conj()  # Q.H @ A

    # SVD of the small matrix B
    Uhat, s, Vt = cp.linalg.svd(B, full_matrices=False)

    # Recover U
    U = Q @ Uhat[:, :k]

    # Truncate to k components
    s = s[:k]
    Vt = Vt[:k, :]

    _svd_flip(U, Vt)

    return U, s, Vt
