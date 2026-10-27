"""Chunked copies between the host and the GPU: the cells uploaded in sorted
order and the result downloaded, through pinned chunks that several host
threads fill or drain."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import numpy as np

_UPLOAD_BYTES = 1 << 28  # largest pinned staging chunk
_UPLOAD_THREADS = 8


def _sorted_device_copy(Z, order: cp.ndarray) -> cp.ndarray:
    """``Z[order]`` on the GPU. Host input is staged in pinned chunks by
    several threads while the previous chunk is copied, and placed directly,
    so no unsorted device copy is kept."""
    if isinstance(Z, cp.ndarray):
        return Z[order]
    n, d = Z.shape
    rows = _staging_rows(n, d * Z.itemsize)
    out = cp.empty(Z.shape, dtype=Z.dtype)
    position = cp.empty_like(order)
    position[order] = cp.arange(order.size)
    stream = cp.cuda.get_current_stream()
    slots = [
        [_pinned(rows, d, Z.dtype), cp.empty((rows, d), dtype=Z.dtype), None]
        for _ in range(2)
    ]
    with ThreadPoolExecutor(_UPLOAD_THREADS) as pool:
        for j, a in enumerate(range(0, n, rows)):
            b = min(n, a + rows)
            host, staged, done = slots[j % 2]
            if done is not None:
                done.synchronize()  # the slot's previous chunk is placed
            _parallel_copy(pool, host[: b - a], Z[a:b])
            staged[: b - a].set(host[: b - a], stream=stream)
            out[position[a:b]] = staged[: b - a]
            slots[j % 2][2] = stream.record()
    stream.synchronize()  # the staging slots are released on return
    return out


def _staging_rows(n: int, row_bytes: int) -> int:
    """Rows per pinned staging chunk: about 1/32 of the data, 64-256 MiB.
    Pinning costs time on first use; every chunk costs a host sync."""
    chunk = min(_UPLOAD_BYTES, max(_UPLOAD_BYTES >> 2, n * row_bytes >> 5))
    return max(1, min(n, chunk // row_bytes))


def _pinned(rows: int, d: int, dtype) -> np.ndarray:
    """A (rows, d) host array in pinned memory (CuPy caches these blocks)."""
    dtype = np.dtype(dtype)
    memory = cp.cuda.alloc_pinned_memory(rows * d * dtype.itemsize)
    return np.frombuffer(memory, dtype, rows * d).reshape(rows, d)


def _parallel_copy(pool: ThreadPoolExecutor, dst: np.ndarray, src: np.ndarray) -> None:
    """``dst[...] = src`` by several threads (host memory bound)."""
    cuts = np.linspace(0, len(src), _UPLOAD_THREADS + 1, dtype=np.int64)
    list(pool.map(lambda a, b: np.copyto(dst[a:b], src[a:b]), cuts[:-1], cuts[1:]))


def _download(src: cp.ndarray, out: np.ndarray) -> None:
    """``out[...] = src`` through pinned chunks: the next chunk is copied from
    the GPU while host threads move the previous one into ``out``."""
    n, d = src.shape
    rows = _staging_rows(n, d * src.itemsize)
    stream = cp.cuda.get_current_stream()
    hosts = [_pinned(rows, d, src.dtype) for _ in range(2)]
    with ThreadPoolExecutor(_UPLOAD_THREADS) as pool:

        def drain(chunk, a, b, done):
            done.synchronize()
            _parallel_copy(pool, out[a:b], chunk)

        pending = None
        for j, a in enumerate(range(0, n, rows)):
            b = min(n, a + rows)
            chunk = hosts[j % 2][: b - a]
            src[a:b].get(stream=stream, out=chunk, blocking=False)
            ready = (chunk, a, b, stream.record())
            if pending is not None:
                drain(*pending)
            pending = ready
        if pending is not None:
            drain(*pending)
