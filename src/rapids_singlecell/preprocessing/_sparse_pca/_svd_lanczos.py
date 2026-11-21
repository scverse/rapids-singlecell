"""
GPU-accelerated Sparse SVD using Lanczos Bidiagonalization.

This module provides a custom SVD solver optimized for sparse matrices on GPU,
using Lanczos bidiagonalization with Classical Gram-Schmidt (CGS2) orthogonalization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import issparse
from cupyx.scipy.sparse.linalg import LinearOperator

from rapids_singlecell._utils._random import _seed_from_rng

from ._operators import _svd_flip, mean_centered_operator

if TYPE_CHECKING:
    from rapids_singlecell._utils._random import RNGLike, SeedLike

# Simple fused AXPY kernel - faster than function call overhead
_kernel_axpy = cp.ElementwiseKernel("T alpha, T y", "T x", "x -= alpha * y", "lanczos_axpy")


def _cgs2_orth(
    target: cp.ndarray,
    basis_matrix: cp.ndarray,
    n_valid: int,
    coeffs_buf: cp.ndarray,
) -> cp.ndarray:
    """
    Orthogonalize 'target' against the first 'n_valid' columns of 'basis_matrix' using CGS2.

    Parameters
    ----------
    target
        Vector to orthogonalize (modified in-place).
    basis_matrix
        Matrix containing orthonormal basis vectors as columns.
    n_valid
        Number of valid basis vectors (columns) to orthogonalize against.
    coeffs_buf
        Pre-allocated buffer for storing projection coefficients.

    Returns
    -------
    The orthogonalized target vector.
    """
    if n_valid == 0:
        return target

    basis_view = basis_matrix[:, :n_valid]
    coeffs = coeffs_buf[:n_valid]

    target = cp.ascontiguousarray(target)
    trans = "H" if basis_view.dtype.kind == "c" else "T"
    for _ in range(2):
        cp.cublas.gemv(trans, 1, basis_view, target, 0, coeffs)
        cp.cublas.gemv("N", -1, basis_view, coeffs, 1, target)

    return target


def _breakdown_threshold(scale: float, dtype) -> float:
    """
    Scale- and precision-aware breakdown criterion.

    A candidate Lanczos vector counts as numerically zero when its norm is negligible
    relative to ``scale``, the largest bidiagonal entry produced so far (a running lower
    bound on ``||A||_2``). Until scale information exists, only norms below the smallest
    normal number are treated as zero, so the solver behaves identically for ``A`` and
    ``c * A`` for any ``c > 0``.

    This replaces a hardcoded ``1e-9``, which was effectively dead in float32: the
    smallest genuine Lanczos vector norms sit around ``sqrt(eps_f32) ~= 3.4e-4``, five
    orders of magnitude above the old threshold, so breakdown was never detected and the
    random-restart path that recovers repeated singular values never engaged. Matches
    ``breakdown_threshold`` in RAFT's ``lanczos_svds.cuh``.
    """
    finfo = cp.finfo(dtype)
    return max(float(finfo.eps) * float(scale), float(finfo.tiny))


def _projected_svd(alphas, betas):
    steps = alphas.size
    B = cp.zeros((steps, steps), dtype=alphas.dtype)
    cp.fill_diagonal(B, alphas)
    cp.fill_diagonal(B[:-1, 1:], betas[1:steps])
    return cp.linalg.svd(B)


def _restart_converged(alphas, betas, remaining):
    # Inspect the projected problem in double precision; its trailing Ritz
    # entries can be much smaller than the single-precision SVD can resolve.
    left, values, _ = _projected_svd(alphas.astype(cp.float64, copy=False), betas.astype(cp.float64, copy=False))
    residual = abs(betas[-1] * left[-1, : remaining + 2]).max()
    gap = values[remaining - 1] - values[remaining]
    eps = cp.finfo(alphas.dtype).eps
    # Require two guard pairs, a resolved cutoff gap, and an unbroken recurrence.
    return (
        (gap > 8 * eps * values[0])
        & (residual < cp.minimum(eps * values[0], eps**0.5 * gap))
        & (alphas != 0).all()
        & (betas[1:] != 0).all()
    )


def _lanczos_bidiag(
    A,
    *,
    ncv: int,
    v_start: cp.ndarray,
    U_full: cp.ndarray,
    V_full: cp.ndarray,
    n_locked: int,
    rng: cp.random.Generator,
    ortho_buf: cp.ndarray,
    remaining: int = 0,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Lanczos bidiagonalization step.

    Parameters
    ----------
    A
        Sparse matrix (m x n) to decompose.
    ncv
        Number of Lanczos vectors to compute.
    v_start
        Starting vector for the iteration.
    U_full
        Pre-allocated buffer for left singular vectors.
    V_full
        Pre-allocated buffer for right singular vectors.
    n_locked
        Number of already converged (locked) vectors.
    rng
        Random number generator for restarting.
    ortho_buf
        Pre-allocated buffer for orthogonalization coefficients.
    remaining
        Number of leading Ritz pairs to check during a restart. Zero uses the full sweep.

    Returns
    -------
    Tuple of (alphas, betas) - the bidiagonal matrix entries.
    """
    m, n = A.shape
    dtype = A.dtype
    adjoint_dot = A.rmatvec if isinstance(A, LinearOperator) else A.T.conj().dot

    # Breakdown decisions already need the norms on the host. Keep the small
    # bidiagonal there until the projected SVD and normalize straight into the basis.
    norm_out = np.empty((), dtype=dtype).real
    # AXPY takes scalars of the vector dtype. The projected bidiagonal is real.
    alphas = np.empty(ncv, dtype=dtype)
    betas = np.empty(ncv + 1, dtype=dtype)
    betas[0] = 0

    def norm(x):
        return cp.cublas.nrm2(cp.ascontiguousarray(x), out=norm_out)[()]

    # Running estimate of ||A||_2 used to scale the breakdown checks below.
    scale = 0.0

    # Initialize v_start
    v = v_start

    # Ortho against locked V vectors
    v = _cgs2_orth(v, V_full, n_locked, ortho_buf)

    # v_start is unit-scale by construction (a normalized restart vector or a standard
    # normal vector), so its breakdown check is relative to unit scale.
    beta_val = norm(v)
    if beta_val < _breakdown_threshold(1.0, dtype):
        v = rng.standard_normal(n).astype(dtype)
        v = _cgs2_orth(v, V_full, n_locked, ortho_buf)
        beta_val = norm(v)

    cp.divide(v, beta_val, out=V_full[:, n_locked])

    # Small remaining subspaces can converge early in a restart.
    checkpoints = (max(40, 2 * remaining + 16), max(48, 2 * remaining + 16)) if remaining else ()

    # --- LANCZOS LOOP ---
    for i in range(ncv):
        idx_u = n_locked + i
        idx_v = n_locked + i

        # u = A @ v
        u = A.dot(V_full[:, idx_v])

        # Deflation: u = u - beta * u_prev
        if i > 0:
            _kernel_axpy(betas[i], U_full[:, idx_u - 1], u)

        # Ortho U against all valid U
        u = _cgs2_orth(u, U_full, idx_u, ortho_buf)

        alpha_val = norm(u)
        if alpha_val < _breakdown_threshold(scale, dtype):
            u = rng.standard_normal(m).astype(dtype)
            u = _cgs2_orth(u, U_full, idx_u, ortho_buf)
            alpha_val = norm(u)
            alphas[i] = 0.0
        else:
            alphas[i] = alpha_val
            scale = max(scale, alpha_val)
        u = cp.divide(u, alpha_val, out=U_full[:, idx_u])

        # v = A.H @ u
        if i < ncv:
            v = adjoint_dot(u)
            _kernel_axpy(alphas[i], V_full[:, idx_v], v)

            # Ortho V against all valid V
            v = _cgs2_orth(v, V_full, idx_v + 1, ortho_buf)

            beta_val = norm(v)
            if beta_val < _breakdown_threshold(scale, dtype):
                v = rng.standard_normal(n).astype(dtype)
                v = _cgs2_orth(v, V_full, idx_v + 1, ortho_buf)
                beta_val = norm(v)
                betas[i + 1] = 0.0
            else:
                betas[i + 1] = beta_val
                scale = max(scale, beta_val)
            cp.divide(v, beta_val, out=V_full[:, idx_v + 1])

        steps = i + 1
        if steps in checkpoints and steps < ncv:
            a, b = cp.asarray(alphas[:steps].real), cp.asarray(betas[: steps + 1].real)
            if _restart_converged(a, b, remaining):
                return a, b

    return cp.asarray(alphas.real), cp.asarray(betas.real)


def lanczos_svd(
    A,
    k: int = 6,
    *,
    ncv: int | None = None,
    tol: float = 1e-4,
    max_iter: int = 100,
    rng: SeedLike | RNGLike | None = None,
    refine_results: bool = True,
    _return_scores: bool = False,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Compute the largest k singular values and vectors for a sparse matrix.

    This function uses Lanczos bidiagonalization with implicit restarts
    optimized for GPU execution with CuPy.

    Parameters
    ----------
    A
        Sparse matrix (m x n) to decompose. Should be a CuPy sparse matrix
        (CSR or CSC format recommended).
    k
        Number of singular values and vectors to compute.
    ncv
        Number of Lanczos vectors generated. Must be greater than k.
        Default is min(max(3*k, 50), min(m,n)).
    tol
        Tolerance for convergence.
    max_iter
        Maximum number of restart iterations.
    rng
        Random seed or :class:`~numpy.random.Generator` for reproducibility.
    refine_results
        If True, refine the singular vectors after convergence.
    _return_scores
        Return the projected scores in place of normalized left singular vectors for PCA.

    Returns
    -------
    U : cp.ndarray
        Left singular vectors of shape (m, k).
    S : cp.ndarray
        Singular values of shape (k,).
    Vt : cp.ndarray
        Right singular vectors of shape (k, n).
    """
    if issparse(A):
        A = mean_centered_operator(A)
    m, n = A.shape
    dtype = A.dtype

    automatic_sweep = ncv is None and m > 100000 and dtype.kind == "f" and tol >= cp.finfo(dtype).eps
    if ncv is None:
        # Balance between restart cost and per-iteration cost
        # Larger ncv = fewer restarts but more orthogonalization work per iteration
        if m > 100000:
            # For large matrices with mean-centering (PCA use case):
            # - Small k (k<75): ~2.5*k works well
            # - Larger k: need ~3.9*k to minimize restarts (empirically tuned)
            if k < 75:
                ncv = min(int(2.5 * k), min(m, n))
            else:
                ncv = min(int(3.9 * k), min(m, n))
        else:
            ncv = min(max(3 * k, 50), min(m, n))

    ncv = max(ncv, k + 10)

    # CuPy generators are seeded, so this is where the host generator
    # collapses into an integer.
    rng = cp.random.default_rng(_seed_from_rng(rng, allow_none=False))

    # Pre-allocate buffers
    total_capacity = k + ncv + 2
    U_full = cp.zeros((m, total_capacity), dtype=dtype, order="F")
    V_full = cp.zeros((n, total_capacity), dtype=dtype, order="F")
    locked_S = cp.zeros(k, dtype=U_full.real.dtype)
    ortho_buf = cp.zeros(total_capacity, dtype=dtype)

    n_locked = 0
    total_iter = 0
    v_start = rng.standard_normal(n).astype(dtype)

    while n_locked < k and total_iter < max_iter:
        alphas, betas = _lanczos_bidiag(
            A,
            ncv=ncv,
            v_start=v_start,
            U_full=U_full,
            V_full=V_full,
            n_locked=n_locked,
            rng=rng,
            ortho_buf=ortho_buf,
            remaining=k - n_locked if automatic_sweep and n_locked else 0,
        )

        # SVD of Bidiagonal Matrix
        steps = alphas.size
        p, s, qt = _projected_svd(alphas, betas)

        # Sort descending
        idx = cp.argsort(s)[::-1]
        s = s[idx]
        p = p[:, idx]
        qt = qt[idx, :]

        # Error estimates
        resid = betas[-1] * cp.abs(p[-1, :])
        max_s = s[0] if s.shape[0] > 0 else 1.0

        # Transfer all convergence decisions in one synchronization per sweep.
        converged = (resid[: k - n_locked + 2] < tol * max_s).get()
        converged_indices = converged.nonzero()[0].tolist()

        if converged_indices:
            num_found = len(converged_indices)
            if n_locked + num_found > k:
                num_found = k - n_locked
                converged_indices = converged_indices[:num_found]

            good_idx = cp.array(converged_indices)

            start = n_locked
            end = n_locked + num_found
            if end < k:
                # The restart coefficients refer to the basis before locking.
                best_nc = next(i for i in range(steps) if i not in converged_indices)
                v_start = V_full[:, start : start + steps] @ qt[best_nc, :]
            # Refinement reconstructs U from the final right singular vectors.
            if end < k or not refine_results:
                U_full[:, start:end] = cp.dot(U_full[:, start : start + steps], p[:, good_idx])
            V_full[:, start:end] = cp.dot(qt[good_idx, :], V_full[:, start : start + steps].T).T
            locked_S[start:end] = s[good_idx]

            n_locked += num_found

            if n_locked >= k:
                break

        else:
            # Thick restart
            k_mix = min(k - n_locked, steps)
            weights = cp.ones(k_mix, dtype=dtype)
            combo_small = cp.dot(qt[:k_mix, :].T, weights)
            v_start = cp.dot(V_full[:, n_locked : n_locked + steps], combo_small)

        v_start /= cp.linalg.norm(v_start)
        total_iter += 1

    # Final extraction
    final_V = V_full[:, :k].copy(order="F")
    if refine_results:
        del U_full, V_full
        final_U = A.dot(final_V)
        final_S = cp.linalg.norm(final_U, axis=0)
        if not _return_scores:
            final_U /= final_S[None, :]
    else:
        final_U = U_full[:, :k].copy()
        final_S = locked_S
        if _return_scores:
            final_U *= final_S[None, :]

    # Sort by singular value
    sort_idx = cp.argsort(final_S)[::-1]
    final_S = final_S[sort_idx]
    final_U = final_U[:, sort_idx]
    final_Vt = final_V.T[sort_idx, :].conj()
    _svd_flip(final_U, final_Vt)

    return final_U, final_S, final_Vt
