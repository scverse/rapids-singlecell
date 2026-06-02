"""Batched log-domain Sinkhorn solver over a ragged (flat, no-padding) layout.

Pairs of point clouds are stored flat with per-pair int64 offsets; the CUDA
kernels (``_sinkhorn_cuda``: build_cost, auto_eps, update_g, update_f,
check_convergence) solve a whole batch at once. The Python iteration loop
(:func:`run_async`) dispatches per-iteration launches to several devices'
streams and reads per-pair convergence at the end of each ``check_every``-step.
The per-pair return matches OTT-JAX's ``reg_ot_cost = f@a + g@b + eps*(log n_a +
log n_b)`` for uniform marginals.
"""

from __future__ import annotations

import cupy as cp

from rapids_singlecell._cuda import _sinkhorn_cuda as _sk

# Defaults match OTT-JAX/pertpy (eps scale 0.05, max_iter 2000). tol is our
# potential-change criterion, not OTT's marginal threshold; 1e-4 gives ~5e-6
# reg_ot_cost accuracy (tighter than OTT's default), so don't loosen it.
DEFAULT_EPSILON_SCALE = 0.05
DEFAULT_MAX_ITER = 2000
DEFAULT_TOL = 1e-4
DEFAULT_CHECK_EVERY = 20
DEFAULT_RELAXATION = 1.0
EPSILON_FLOOR = 1e-12


def make_state(layout: dict, cost: cp.ndarray, *, epsilon_scale: float) -> dict:
    """Build per-pair solver state (auto-eps + potentials) on the current device."""
    dtype = cost.dtype
    if dtype not in (cp.float32, cp.float64):
        raise TypeError(f"Sinkhorn supports float32/float64; got {dtype}")
    n = layout["n"]
    m = layout["m"]
    B = int(layout["B"])
    total_rows = int(layout["total_rows"])
    total_cols = int(layout["total_cols"])

    na_s = cp.maximum(n, 1).astype(dtype)
    nb_s = cp.maximum(m, 1).astype(dtype)
    log_a = (-cp.log(na_s)).astype(dtype)
    log_b = (-cp.log(nb_s)).astype(dtype)

    eps = cp.empty(B, dtype=dtype)
    _sk.auto_eps(
        cost,
        layout["cost_off"],
        n,
        m,
        float(epsilon_scale),
        float(EPSILON_FLOOR),
        eps,
        cp.cuda.get_current_stream().ptr,
    )

    return {
        "cost": cost,
        "cost_off": layout["cost_off"],
        "f_off": layout["f_off"],
        "g_off": layout["g_off"],
        "n": n,
        "m": m,
        "row2pair": layout["row2pair"],
        "col2pair": layout["col2pair"],
        "na_s": na_s,
        "nb_s": nb_s,
        "log_a": log_a,
        "log_b": log_b,
        "eps": eps,
        "B": B,
        "dtype": dtype,
        "f": cp.zeros(total_rows, dtype=dtype),
        "g": cp.zeros(total_cols, dtype=dtype),
        "f_prev": cp.zeros(total_rows, dtype=dtype),
        "g_prev": cp.zeros(total_cols, dtype=dtype),
        "conv": cp.zeros(B, dtype=cp.int32),
    }


def _step(st, *, will_check, tol, omega=DEFAULT_RELAXATION):
    """Queue one Sinkhorn iteration (no host sync). omega in [1, 2) over-relaxes."""
    f, g = st["f"], st["g"]
    cost, conv, eps = st["cost"], st["conv"], st["eps"]
    co, fo, go = st["cost_off"], st["f_off"], st["g_off"]
    n, m, r2p, c2p = st["n"], st["m"], st["row2pair"], st["col2pair"]
    sp = cp.cuda.get_current_stream().ptr
    if will_check:
        # Snapshot for the single-iteration change measured by check_convergence.
        cp.copyto(st["f_prev"], f)
        cp.copyto(st["g_prev"], g)
    _sk.update_g(cost, co, n, m, f, fo, g, go, c2p, eps, st["log_b"], conv, omega, sp)
    _sk.update_f(cost, co, m, g, go, f, fo, r2p, eps, st["log_a"], conv, omega, sp)
    if will_check:
        _sk.check_convergence(
            f, st["f_prev"], fo, n, g, st["g_prev"], go, m, float(tol), conv, sp
        )


def _segment_sum(values: cp.ndarray, starts: cp.ndarray, total: int) -> cp.ndarray:
    """Per-pair sum over contiguous segments via a float64 prefix-sum difference.

    Deterministic (unlike atomic scatter-add, which would make the bootstrap
    variance vary run to run) and float64-accumulated for float32 potentials.
    """
    pref = cp.concatenate(
        [cp.zeros(1, dtype=cp.float64), cp.cumsum(values.astype(cp.float64))]
    )
    ends = cp.concatenate([starts[1:], cp.asarray([total], dtype=starts.dtype)])
    return pref[ends] - pref[starts]


def finalize(st) -> cp.ndarray:
    """Per-pair reg_ot_cost from the converged potentials (current stream)."""
    na, nb = st["na_s"], st["nb_s"]
    fsum = _segment_sum(st["f"], st["f_off"], int(st["f"].size))
    gsum = _segment_sum(st["g"], st["g_off"], int(st["g"].size))
    reg = fsum / na + gsum / nb + st["eps"] * (cp.log(na) + cp.log(nb))
    return reg.astype(st["dtype"])


def run_async(units, *, max_iter, tol, check_every, omega=DEFAULT_RELAXATION):
    """Run the Sinkhorn loop across ``units`` (one per device) as a multi-stream
    queue: dispatch ``check_every`` iters to each stream, sync, read per-pair
    convergence, and stop once every pair on every unit has converged.
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
