from __future__ import annotations

import cupy as cp
import numba as nb
import numpy as np

from rapids_singlecell._cuda import _pv_cuda as _pv


def _rev_cummin64(x, n_rows, m):
    y = cp.empty_like(x)

    _pv.rev_cummin64(
        x, out=y, n_rows=n_rows, m=m, stream=cp.cuda.get_current_stream().ptr
    )
    return y


def fdr_bh_axis1_cupy_optimized(ps, *, mem_gb: float = 4.0) -> cp.ndarray:
    """
    Benjamini–Hochberg adjusted p-values along axis=1 (rows), GPU-optimized.
    - Keeps values in float64 for CPU-equivalence.
    - Uses int32 index buffers where safe.
    - Processes rows in chunks to cap peak memory.
    - Avoids a second argsort by building inverse permutation via scatter.
    - Uses a custom kernel for reverse cumulative min.

    Parameters
    ----------
    ps : cupy.ndarray, shape (n_rows, n_tests), dtype float64 (or castable)
         P-values in [0, 1].
    mem_gb : float
         Approx memory budget for temporaries per batch (default 4 GB).

    Returns
    -------
    cupy.ndarray, shape (n_rows, n_tests), dtype float64
         BH-adjusted p-values on device.
    """
    ps = cp.asarray(ps, dtype=cp.float64)
    if ps.ndim != 2:
        raise ValueError("ps must be 2D (n_rows, n_tests).")

    # Bounds check without host transfer
    if not bool((ps == cp.clip(ps, 0.0, 1.0)).all()):
        raise ValueError("`ps` must be within [0, 1].")

    n_rows, m = map(int, ps.shape)
    if m <= 1:
        return ps.copy()

    # Precompute BH scale in float64: scale[j] = m / (j+1)
    scale = m / cp.arange(1, m + 1, dtype=cp.float64)

    # Batch size: very conservative estimate of bytes/row for temps:
    # ps_chunk(8) + order(4) + ps_sorted(8) + ps_bh(8) + ps_mon(8) + inv_order(4) + flat_idx(8) ~= 48 bytes * m
    bytes_per_row = 48 * m
    mem_bytes = int(mem_gb * (1024**3))
    B = max(1, mem_bytes // max(bytes_per_row, 1))

    out = cp.empty_like(ps, dtype=cp.float64)

    for s in range(0, n_rows, B):
        e = min(n_rows, s + B)
        R = e - s

        ps_chunk = ps[s:e]  # (R, m) float64

        # 1) per-row argsort (ascending); keep indices as int32
        order = cp.argsort(ps_chunk, axis=1).astype(cp.int32, copy=False)  # (R, m)

        # 2) gather sorted values with flattened indexing (faster at large sizes)
        row_base = (cp.arange(R, dtype=cp.int32) * m)[:, None]  # (R,1)
        flat_idx = (order.astype(cp.int32) + row_base).ravel()  # (R*m,)
        ps_sorted = ps_chunk.ravel()[flat_idx].reshape(R, m)  # (R, m) float64

        # 3) BH scaling
        ps_bh = ps_sorted * scale  # (R, m) float64

        # 4) reverse cumulative min via custom kernel
        ps_mon = _rev_cummin64(ps_bh, R, m)

        # 5) build inverse permutation without argsort (scatter)
        inv_order = cp.empty_like(order, dtype=cp.int32)  # (R, m) int32
        # Broadcast-safe RHS; no R*m materialization for RHS
        inv_order[cp.arange(R)[:, None], order] = cp.arange(m, dtype=cp.int32)

        # 6) unsort back via flattened gather
        flat_idx2 = (inv_order.astype(cp.int32) + row_base).ravel()
        out[s:e] = ps_mon.ravel()[flat_idx2].reshape(R, m)

    # 7) clamp to [0,1]
    return cp.clip(out, 0.0, 1.0).get().astype(np.float32)


@nb.jit(nopython=True, cache=True)
def _fdr_bh_single_row(ps_row, m):
    """
    Apply Benjamini-Hochberg correction to a single row.
    """
    # Sort the row and get indices
    order = np.argsort(ps_row)
    ps_sorted = ps_row[order]

    # BH scale: p_(i) * m / i
    ps_bh = np.empty_like(ps_sorted, dtype=np.float64)
    for i in range(m):
        ps_bh[i] = ps_sorted[i] * (m / (i + 1))

    # Reverse cumulative min
    ps_rev = np.empty_like(ps_bh, dtype=np.float64)
    for i in range(m):
        ps_rev[i] = ps_bh[m - 1 - i]

    for j in range(1, m):
        ps_rev[j] = min(ps_rev[j], ps_rev[j - 1])

    # Reverse back
    ps_monotone = np.empty_like(ps_rev, dtype=np.float64)
    for i in range(m):
        ps_monotone[i] = ps_rev[m - 1 - i]

    # Unsort back to original order
    ps_adj = np.empty_like(ps_monotone, dtype=np.float64)
    for i in range(m):
        ps_adj[order[i]] = ps_monotone[i]

    # Clip to [0, 1]
    for i in range(m):
        ps_adj[i] = max(0.0, min(1.0, ps_adj[i]))

    return ps_adj


@nb.jit(nopython=True, parallel=True, cache=True)
def _fdr_bh_parallel(ps, m):
    """
    Apply Benjamini-Hochberg correction to all rows in parallel.
    """
    n_rows = ps.shape[0]
    result = np.empty_like(ps, dtype=np.float64)

    for i in nb.prange(n_rows):
        result[i] = _fdr_bh_single_row(ps[i], m)

    return result


def fdr_bh_axis1_numba(ps):
    """
    Benjamini–Hochberg adjusted p-values along axis=1 (rows).
    ps: numpy.ndarray (n_rows, n_tests), values in [0, 1].
    Returns: numpy.ndarray of same shape.
    """
    ps = np.asarray(ps, dtype=np.float64)
    if ps.ndim != 2:
        raise ValueError("ps must be 2D (n_rows, n_tests) for axis=1.")
    if not np.issubdtype(ps.dtype, np.number):
        raise ValueError("`ps` must be numeric.")
    if not np.all((ps >= 0) & (ps <= 1)):
        raise ValueError("`ps` must be within [0, 1].")

    n_rows, m = ps.shape
    if m <= 1:
        return ps.copy().astype(np.float32)

    # Process each row in parallel
    result = _fdr_bh_parallel(ps, m)
    return result.astype(np.float32)


def fdr_bh_axis1(ps: np.ndarray, *, if_gpu: bool = False) -> np.ndarray:
    """
    Main function with CPU/GPU selection and optimization options.

    Parameters:
    -----------
    ps : array-like
        Input p-values array
    if_gpu : bool, default=False
        Whether to use GPU (CuPy) or CPU (Numba)
    """

    if if_gpu:
        out = fdr_bh_axis1_cupy_optimized(ps)
    else:
        out = fdr_bh_axis1_numba(ps)
    return out
