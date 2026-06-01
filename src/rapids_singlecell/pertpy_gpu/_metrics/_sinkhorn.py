"""Batched log-domain Sinkhorn solver (nanobind/CUDA kernels).

Solves entropic-regularized optimal transport between many pairs of point
clouds at once. Pairs in a batch are padded to a common shape; the per-pair
return value matches OTT-JAX's ``reg_ot_cost``:

    reg_ot_cost = f @ a + g @ b + eps * (log n_a + log n_b)   [uniform a, b]

The solver is built from three CUDA kernels (the ``_sinkhorn_cuda`` nanobind
extension): ``auto_eps`` (per-pair ``0.05 * std(C)``), ``update_g`` (parallel
over the M axis), and ``update_f`` (one block per (pair, row), cooperative over
the M axis). One launch handles a whole batch ("multiple Sinkhorns at once").

The iteration loop lives in Python (:func:`run_async`) so it can dispatch work
to several devices' streams without ever waiting mid-flight — an async
multi-stream work queue. Convergence is decided per pair on the device (a
``conv`` flag the update kernels short-circuit on); the host only reads it at
the end of each ``check_every``-iteration super-step.
"""

from __future__ import annotations

import warnings

import cupy as cp

from rapids_singlecell._cuda import _sinkhorn_cuda as _sk

# Solver defaults. epsilon_scale and the max_iter cap match OTT-JAX / pertpy
# (PointCloud relative-epsilon 0.05, Sinkhorn max_iterations 2000). tol is NOT
# OTT's threshold (1e-3): OTT's threshold is a marginal-error criterion, ours is
# the relative change in the potentials between iterations. At 1e-3 our criterion
# yields only ~3e-4 relative error in reg_ot_cost; 1e-4 gives ~5e-6 (tighter than
# OTT's effective default), so the result matches OTT at least as well.
DEFAULT_EPSILON_SCALE = 0.05
DEFAULT_MAX_ITER = 2000
DEFAULT_TOL = 1e-4
DEFAULT_CHECK_EVERY = 20
DEFAULT_RELAXATION = 1.0
EPSILON_FLOOR = 1e-12


def _validate(cost, mask_a, mask_b):
    if cost.ndim != 3:
        raise ValueError(f"cost must be 3-D (B, n, m), got shape {cost.shape}")
    B, n, m = cost.shape
    if mask_a.shape != (B, n):
        raise ValueError(f"mask_a shape {mask_a.shape} != (B, n) = {(B, n)}")
    if mask_b.shape != (B, m):
        raise ValueError(f"mask_b shape {mask_b.shape} != (B, m) = {(B, m)}")


def make_state(cost, mask_a, mask_b, *, epsilon, epsilon_scale):
    """Build per-pair solver state on the *current* device.

    ``cost`` is ``(B, N, M)``; orient the larger group as M (the caller's job)
    for best parallelism. Returns a dict consumed by :func:`run_async`.
    """
    dtype = cost.dtype
    if dtype not in (cp.float32, cp.float64):
        raise TypeError(f"Sinkhorn supports float32/float64; got {dtype}")
    B, N, M = cost.shape
    cost = cp.ascontiguousarray(cost)
    mask_a = cp.ascontiguousarray(mask_a.astype(cp.bool_))
    mask_b = cp.ascontiguousarray(mask_b.astype(cp.bool_))

    na_s = cp.maximum(mask_a.sum(axis=1), 1).astype(dtype)
    nb_s = cp.maximum(mask_b.sum(axis=1), 1).astype(dtype)
    log_a = (-cp.log(na_s)).astype(dtype)
    log_b = (-cp.log(nb_s)).astype(dtype)

    eps = cp.empty(B, dtype=dtype)
    if epsilon is None:
        total = na_s * nb_s
        _sk.auto_eps(
            cost,
            mask_a,
            mask_b,
            total,
            float(epsilon_scale),
            float(EPSILON_FLOOR),
            eps,
            cp.cuda.get_current_stream().ptr,
        )
    elif cp.isscalar(epsilon) or (
        isinstance(epsilon, cp.ndarray) and epsilon.ndim == 0
    ):
        eps[:] = float(epsilon)
        eps = cp.maximum(eps, dtype.type(EPSILON_FLOOR))
    else:
        eps = cp.ascontiguousarray(cp.asarray(epsilon, dtype=dtype))
        if eps.shape != (B,):
            raise ValueError(f"epsilon shape {eps.shape} != (B,) = ({B},)")
        eps = cp.maximum(eps, dtype.type(EPSILON_FLOOR))

    return {
        "cost": cost,
        "mask_a": mask_a,
        "mask_b": mask_b,
        "na_s": na_s,
        "nb_s": nb_s,
        "log_a": log_a,
        "log_b": log_b,
        "eps": eps,
        "B": B,
        "N": N,
        "M": M,
        "dtype": dtype,
        "f": cp.zeros((B, N), dtype=dtype),
        "g": cp.zeros((B, M), dtype=dtype),
        "f_prev": cp.empty((B, N), dtype=dtype),
        "g_prev": cp.empty((B, M), dtype=dtype),
        "conv": cp.zeros(B, dtype=cp.int32),
    }


def _step(st, *, will_check, tol, omega=DEFAULT_RELAXATION):
    """Queue one Sinkhorn iteration on the current stream (no host sync).

    ``omega`` is the over-relaxation factor passed to the update kernels;
    ``omega == 1`` is plain Sinkhorn (Gauss-Seidel), ``omega in (1, 2)``
    accelerates convergence but can diverge if too large.
    """
    f, g = st["f"], st["g"]
    cost, ma, mb, conv, eps = (
        st["cost"],
        st["mask_a"],
        st["mask_b"],
        st["conv"],
        st["eps"],
    )
    sp = cp.cuda.get_current_stream().ptr
    if will_check:
        cp.copyto(st["f_prev"], f)
        cp.copyto(st["g_prev"], g)
    _sk.update_g(cost, ma, mb, f, eps, st["log_b"], conv, g, omega, sp)
    _sk.update_f(cost, ma, mb, g, eps, st["log_a"], conv, f, omega, sp)
    if will_check:
        ma, mb = st["mask_a"], st["mask_b"]
        df = cp.where(ma, cp.abs(f - st["f_prev"]), 0).max(axis=1)
        dg = cp.where(mb, cp.abs(g - st["g_prev"]), 0).max(axis=1)
        change = cp.maximum(df, dg)
        scale = cp.maximum(
            cp.where(ma, cp.abs(f), 0).max(axis=1),
            cp.where(mb, cp.abs(g), 0).max(axis=1),
        )
        crit = (change / (scale + 1) < tol).astype(cp.int32)
        cp.maximum(st["conv"], crit, out=st["conv"])


def finalize(st) -> cp.ndarray:
    """Per-pair reg_ot_cost from the converged potentials (current stream)."""
    f, g, ma, mb = st["f"], st["g"], st["mask_a"], st["mask_b"]
    fa = cp.where(ma, f, 0).sum(axis=1) / st["na_s"]
    gb = cp.where(mb, g, 0).sum(axis=1) / st["nb_s"]
    return fa + gb + st["eps"] * (cp.log(st["na_s"]) + cp.log(st["nb_s"]))


def run_async(units, *, max_iter, tol, check_every, omega=DEFAULT_RELAXATION):
    """Run the Sinkhorn loop across ``units`` as an async multi-stream queue.

    ``units`` is a list of ``{"dev": int, "stream": Stream, "state": state}``.
    Within a ``check_every``-iteration super-step the per-iteration launches are
    dispatched to every unit's stream in turn (so the devices' work queues stay
    fed and overlap), then all streams are synced once and per-pair convergence
    is read back. Stops once every pair on every unit has converged.

    ``omega`` is the over-relaxation factor (see :func:`_step`).
    """
    cev = max(int(check_every), 1)
    it = 0
    while it < max_iter:
        n = min(cev, max_iter - it)
        for k in range(n):
            will_check = k == n - 1
            for u in units:
                with cp.cuda.Device(u["dev"]), u["stream"]:
                    _step(u["state"], will_check=will_check, tol=tol, omega=omega)
        it += n
        for u in units:
            with cp.cuda.Device(u["dev"]):
                u["stream"].synchronize()
        done = True
        for u in units:
            with cp.cuda.Device(u["dev"]):
                done = done and bool(u["state"]["conv"].all().get())
        if done:
            break


def batched_sinkhorn(
    cost: cp.ndarray,
    mask_a: cp.ndarray,
    mask_b: cp.ndarray,
    *,
    epsilon: cp.ndarray | float | None = None,
    epsilon_scale: float = DEFAULT_EPSILON_SCALE,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    check_every: int = DEFAULT_CHECK_EVERY,
    relaxation: float = DEFAULT_RELAXATION,
) -> cp.ndarray:
    """Solve ``B`` padded Sinkhorn problems on the current device.

    Parameters
    ----------
    cost
        ``(B, N, M)`` non-negative cost tensor (squared-Euclidean for
        2-Wasserstein). Padded entries are ignored via the masks.
    mask_a, mask_b
        ``(B, N)`` / ``(B, M)`` boolean masks; ``True`` for real points.
    epsilon
        Per-pair regularization: ``(B,)`` array, scalar, or ``None`` (auto:
        ``epsilon_scale * std(C)`` per pair, matching OTT-JAX).
    epsilon_scale, max_iter, tol, check_every
        Auto-eps multiplier, iteration cap, relative tolerance, and convergence
        check frequency.
    relaxation
        Over-relaxation factor ``omega``. ``1.0`` is plain Sinkhorn; values in
        ``(1, 2)`` cut the iteration count (roughly 2x near ``1.5``) but can
        diverge if too large for the problem.

    Returns
    -------
    cp.ndarray
        ``reg_ot_cost`` per pair, shape ``(B,)``.
    """
    _validate(cost, mask_a, mask_b)
    st = make_state(cost, mask_a, mask_b, epsilon=epsilon, epsilon_scale=epsilon_scale)
    units = [
        {
            "dev": cp.cuda.runtime.getDevice(),
            "stream": cp.cuda.get_current_stream(),
            "state": st,
        }
    ]
    run_async(
        units,
        max_iter=max_iter,
        tol=tol,
        check_every=check_every,
        omega=relaxation,
    )
    reg = finalize(st)
    if not bool(st["conv"].all().get()):
        warnings.warn(
            f"Sinkhorn did not converge in {max_iter} iterations (tol={tol}).",
            RuntimeWarning,
            stacklevel=2,
        )
    return reg
