"""Chunked copies between the host and the GPUs: the cells uploaded in sorted
order (each GPU its shard) and the results downloaded in input order, through
pinned chunks that several host threads fill or drain."""

from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from ._multi_gpu import _copy

if TYPE_CHECKING:
    import threading

_UPLOAD_BYTES = 1 << 28  # largest pinned staging chunk
_UPLOAD_THREADS = 8


def _upload_sorted(
    Z,
    order: cp.ndarray,
    bounds: list[int],
    devices: list[int],
    shard_rows: tuple[list, list] | None = None,
) -> list[cp.ndarray]:
    """Rows ``Z[order[lo:hi]]`` of every shard on its GPU. Device input is
    gathered on its own GPU, the other GPUs' shards first and in chunks of
    _UPLOAD_BYTES, so only a shard's rows move and that GPU holds one extra
    chunk at a time. Host input is staged in pinned chunks by several host
    threads while the previous chunk is copied, then placed on the GPU (all
    GPUs at once), so no unsorted device copy is kept: one GPU reads the input
    in order, several GPUs each only their own rows (``shard_rows``, see
    _shard_rows; gathered in input order, from C-ordered input)."""
    shards = list(itertools.pairwise(bounds))
    n, d = Z.shape
    if isinstance(Z, cp.ndarray):
        source, parts = Z.device.id, [None] * len(devices)
        step = max(1, _UPLOAD_BYTES // (d * Z.itemsize))
        with cp.cuda.Device(source):
            for r in sorted(range(len(devices)), key=lambda r: devices[r] == source):
                lo, hi = shards[r]  # the temporaries die with the statement
                if devices[r] == source:
                    parts[r] = Z[_copy(order[lo:hi], source)]
                    continue
                with cp.cuda.Device(devices[r]):
                    parts[r] = cp.empty((hi - lo, d), dtype=Z.dtype)
                for a in range(lo, hi, step):
                    b = min(hi, a + step)
                    chunk = Z[_copy(order[a:b], source)]
                    _copy(chunk, devices[r], out=parts[r][a - lo : b - lo])
                    del chunk  # before the next one
            cp.cuda.get_current_stream().synchronize()  # read on other streams
        return parts
    rows = _staging_rows(n, d * Z.itemsize)
    sources, targets = shard_rows or _shard_rows(order, bounds, devices)
    if len(shards) > 1:
        Z = np.ascontiguousarray(Z)  # np.take copies other layouts whole
        # The upload threads read them on their own streams.
        cp.cuda.get_current_stream().synchronize()

    def upload(rank, fill):
        (lo, hi), source = shards[rank], sources[rank]
        with cp.cuda.Device(devices[rank]):
            target = targets[rank]
            stream = cp.cuda.get_current_stream()
            X = cp.empty((hi - lo, d), dtype=Z.dtype)
            step = max(1, min(rows, hi - lo))  # no larger than the shard
            slots = [
                [_pinned(step, d, Z.dtype), cp.empty((step, d), dtype=Z.dtype), None]
                for _ in range(2)
            ]
            for i, a in enumerate(range(0, hi - lo, step)):
                host, staged, done = slots[i % 2]
                if done is not None:
                    done.synchronize()  # the slot's previous chunk is placed
                b = min(hi - lo, a + step)
                if source is None:
                    _parallel_copy(fill, host[: b - a], Z[a:b])
                else:
                    _parallel_copy(fill, host[: b - a], Z, source[a:b])
                staged[: b - a].set(host[: b - a], stream=stream)
                X[target[a:b]] = staged[: b - a]
                slots[i % 2][2] = stream.record()
            stream.synchronize()  # the caller reads it on its own stream
            return X

    with ThreadPoolExecutor(_UPLOAD_THREADS) as fill:
        if len(devices) == 1:
            return [upload(0, fill)]
        with ThreadPoolExecutor(len(devices)) as pool:
            return list(pool.map(lambda rank: upload(rank, fill), range(len(devices))))


def _shard_rows(
    order: cp.ndarray, bounds: list[int], devices: list[int]
) -> tuple[list, list]:
    """Each shard's cells in input order: their input rows (increasing, on the
    host; ``None`` for a single shard, which has them all) and their rows in
    the shard (on its GPU)."""
    n = order.size
    position = cp.empty(n, dtype=cp.int32)
    position[order] = cp.arange(n, dtype=cp.int32)
    if len(bounds) == 2:
        return [None], [_copy(position, devices[0])]
    sources, targets = [], []
    for (lo, hi), device in zip(itertools.pairwise(bounds), devices, strict=True):
        mine = cp.flatnonzero((position >= lo) & (position < hi))
        targets.append(_copy(position[mine] - lo, device))
        sources.append(cp.asnumpy(mine.astype(cp.int32)))
    return sources, targets


def _write_shards(
    results: list[cp.ndarray],
    shard_rows: tuple[list, list],
    devices: list[int],
    out: np.ndarray,
    prefault: threading.Thread | None,
) -> None:
    """``out[order] = concatenated results`` (the shards' rows in sorted
    order): every GPU puts its rows in input order and copies them into
    theirs of ``out`` (see _shard_rows), so no row moves between the GPUs.
    ``out`` is written once ``prefault`` (see _prefaulted) has finished."""
    sources, targets = shard_rows

    def write(rank):
        with cp.cuda.Device(devices[rank]):
            values = results[rank][targets[rank]]
            results[rank] = None
            if prefault is not None:
                prefault.join()
            _download(values, out, sources[rank])

    if len(devices) == 1:
        return write(0)
    with ThreadPoolExecutor(len(devices)) as pool:
        list(pool.map(write, range(len(devices))))


def _staging_rows(n: int, row_bytes: int) -> int:
    """Rows per pinned staging chunk: about 1/32 of the data, 64-256 MiB.

    Pinning costs time on first use; every chunk costs a host sync.
    """
    chunk = min(_UPLOAD_BYTES, max(_UPLOAD_BYTES >> 2, n * row_bytes >> 5))
    return max(1, min(n, chunk // row_bytes))


def _pinned(rows: int, d: int, dtype) -> np.ndarray:
    """A (rows, d) host array in pinned memory (CuPy caches these blocks)."""
    dtype = np.dtype(dtype)
    memory = cp.cuda.alloc_pinned_memory(rows * d * dtype.itemsize)
    return np.frombuffer(memory, dtype, rows * d).reshape(rows, d)


def _parallel_copy(
    pool: ThreadPoolExecutor,
    dst: np.ndarray,
    src: np.ndarray,
    index: np.ndarray | None = None,
    *,
    scatter: bool = False,
) -> None:
    """``dst[...] = src``, ``src[index]`` or (``scatter``) ``dst[index] = src``
    for increasing ``index`` into C-ordered arrays, by several threads (host
    memory bound)."""
    n = len(src if scatter else dst)
    cuts = np.linspace(0, n, _UPLOAD_THREADS + 1, dtype=np.int64)

    def copy(a, b):
        if index is None:
            np.copyto(dst[a:b], src[a:b])
        elif b > a and index[b - 1] - index[a] == b - a - 1:  # consecutive rows
            run = slice(index[a], index[a] + b - a)
            np.copyto(*((dst[run], src[a:b]) if scatter else (dst[a:b], src[run])))
        elif scatter:
            dst[index[a:b]] = src[a:b]
        elif b > a:  # mode="raise" would buffer the output
            np.take(src, index[a:b], axis=0, out=dst[a:b], mode="clip")

    list(pool.map(copy, cuts[:-1], cuts[1:]))


def _download(
    src: cp.ndarray, out: np.ndarray, index: np.ndarray | None = None
) -> None:
    """``out[...] = src``, or ``out[index] = src`` for increasing ``index``,
    through pinned chunks: the next chunk is copied from the GPU while host
    threads move the previous one into ``out``."""
    n, d = src.shape
    rows = _staging_rows(n, d * src.itemsize)
    stream = cp.cuda.get_current_stream()
    hosts = [_pinned(rows, d, src.dtype) for _ in range(2)]
    with ThreadPoolExecutor(_UPLOAD_THREADS) as pool:

        def drain(chunk, a, b, done):
            done.synchronize()
            if index is None:
                _parallel_copy(pool, out[a:b], chunk)
            else:
                _parallel_copy(pool, out, chunk, index[a:b], scatter=True)

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
