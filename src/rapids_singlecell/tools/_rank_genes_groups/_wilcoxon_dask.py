"""Read CSR rows once and route losslessly encoded records to the GPU owning each gene."""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from contextvars import copy_context
from tempfile import TemporaryFile
from threading import Lock
from types import SimpleNamespace

import cupy as cp
import numpy as np

from ._wilcoxon_dictionary import assemble_records, pack_dictionary
from ._wilcoxon_panel import rank_panel


def _results(futures):
    """Drain all submitted operations before their backing store can be closed."""
    try:
        return [future.result() for future in futures]
    except Exception:
        for future in futures:
            with suppress(Exception):
                future.result()
        raise


class _ColumnStore:
    """A Dask actor owns one contiguous gene range and its local spill files."""

    def __init__(self, available):
        from distributed import get_worker  # type: ignore[import-not-found]

        worker = get_worker()
        self.scratch = worker.local_directory
        self.limits = (int(0.40 * available), (worker.memory_manager.memory_limit or 0) // 4)
        self.used = [0, 0]
        self.blocks = {}
        self.threads = min(3, worker.state.nthreads)
        self.lock = Lock()
        self.device = cp.cuda.Device().id

    def put(self, row, block):
        offsets, words, aux, bits = block
        size = words.nbytes + aux.nbytes

        def spill(arr, tier):
            if tier == 1 or not arr.size:
                return arr.get()
            try:
                with TemporaryFile(dir=self.scratch) as file:
                    os.posix_fallocate(file.fileno(), 0, arr.nbytes)
                    mapped = np.memmap(file, mode="r+", shape=arr.shape, dtype=arr.dtype)
            except OSError as exc:
                raise MemoryError("Cannot allocate worker scratch space for Wilcoxon") from exc
            arr.get(out=mapped)
            return mapped

        with self.lock:
            tier = next((i for i, limit in enumerate(self.limits) if self.used[i] + size <= limit), 2)
            if tier < 2:
                self.used[tier] += size
        if tier:
            with cp.cuda.Device(self.device):
                block = offsets, spill(words, tier), spill(aux, tier), bits
        with self.lock:
            self.blocks[row] = block

    async def ingest(self, readers, codes, owner, stores, bounds):
        # Leave the actor executor free to receive records from the other readers.
        def read(j):
            with cp.cuda.Device(self.device):
                try:
                    pieces = pack_dictionary(readers[j](), codes[j], bounds)
                    cp.cuda.get_current_stream().synchronize()
                except MemoryError:
                    # No records have been sent yet. Retry this row alone after
                    # the other reads on this worker release their decode buffers.
                    return None
                if pieces is None:
                    return False
                row = owner + j * len(stores)
                sends = [
                    peer.put(row, part) for i, (peer, part) in enumerate(zip(stores, pieces, strict=True)) if i != owner
                ]
                try:
                    self.put(row, pieces[owner])
                finally:
                    _results(sends)
                return True

        def process():
            with ThreadPoolExecutor(self.threads) as pool:
                jobs = [pool.submit(copy_context().run, read, j) for j in range(len(readers))]
                done = [job.result() for job in jobs]
            return all(read(j) if result is None else result for j, result in enumerate(done))

        return await asyncio.to_thread(process)

    def rank(self, n_rows, rg, options, max_entries):
        if len(self.blocks) != n_rows:
            raise RuntimeError("Wilcoxon did not receive every row block")
        blocks = [self.blocks[i] for i in range(n_rows)]
        counts = sum(np.diff(b[0]).astype(np.int64) for b in blocks)
        ptr = np.r_[0, np.cumsum(counts, dtype=np.int64)]
        groups = len(rg.group_sizes) - (rg.ireference is not None)
        outputs = [np.empty((groups, len(counts)), np.float64, order="F") for _ in range(3)]
        lo = 0
        with cp.cuda.Device(self.device):
            max_columns = max(1, min(4096, self.limits[0] // (128 * len(rg.group_sizes))))
            while lo < len(counts):
                hi = np.searchsorted(ptr, ptr[lo] + max_entries, side="right") - 1
                hi = int(min(max(lo + 1, hi), lo + max_columns))
                indptr, values, codes = assemble_records(blocks, lo, hi)
                result = rank_panel(values, codes, indptr, rg, options)
                if result is None:
                    return None
                for output, stat in zip(outputs, result[1:], strict=True):
                    stat.get(out=output[:, lo:hi])
                del indptr, values, codes, result, stat
                lo = hi
        return outputs

    def close(self):
        self.blocks.clear()


def try_streamed(rg, client, workers, options, readers):
    from ._wilcoxon import _available_device_bytes

    X = rg.X
    if len(rg.group_sizes) > 65535 or rg.comp_pts or options["multi_gpu"] not in (False, None):
        return None
    hosts = list(workers)[: min(len(readers), X.shape[1])]
    available = client.run(_available_device_bytes, workers=hosts)
    # Reserve space for radix-sort keys, group summaries, and the incoming CSR blocks.
    max_entries = min(2**30, min(available.values()) // 360)
    if not 0 < X.shape[0] <= max_entries:
        return None
    bounds = np.linspace(0, X.shape[1], len(hosts) + 1, dtype=int)
    labels = np.split(rg.group_codes.astype(np.uint16), np.cumsum(X.chunks[0])[:-1])
    fields = ("group_sizes", "ireference", "mean_in_log_space", "_log1p_base")
    context = SimpleNamespace(n_obs=X.shape[0], **{field: getattr(rg, field) for field in fields})
    stores = []
    handles = [client.submit(_ColumnStore, available[h], workers=[h], actor=True, pure=False) for h in hosts]
    try:
        stores = client.gather(handles)
        reads = [
            store.ingest(readers[i :: len(hosts)], labels[i :: len(hosts)], i, stores, bounds)
            for i, store in enumerate(stores)
        ]
        if not all(_results(reads)):
            return None
        pieces = _results([s.rank(len(readers), context, options, max_entries) for s in stores])
        if any(p is None for p in pieces):
            return None
        outputs = [np.concatenate(arrays, axis=1) for arrays in zip(*pieces, strict=True)]
        with suppress(MemoryError, cp.cuda.runtime.CUDARuntimeError):
            if _available_device_bytes() >= 100 * outputs[0].size:
                return [cp.asarray(a) for a in outputs]
        return outputs
    except MemoryError:
        return None
    finally:
        try:
            _results([s.close() for s in stores])
        finally:
            for handle in handles:
                handle.release()
