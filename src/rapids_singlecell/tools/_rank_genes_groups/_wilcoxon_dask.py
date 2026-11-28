"""Distributed exact ranking of bounded windows from row-chunked GPU arrays."""

from __future__ import annotations

from contextlib import suppress
from functools import partial
from operator import getitem
from types import SimpleNamespace

import cupy as cp
import numpy as np

from ._wilcoxon_records import _on_device, csc_records, pack_records, range_base, rank_records


def _pack_row(x, codes, *, edges, on_host):
    def to_host(a):
        return a.get(out=np.empty(a.shape, dtype=a.dtype))

    parts = pack_records(x, codes[:, 0], edges)
    if on_host and parts is not None:
        # The gene shards of a row share its lookup table.
        table = to_host(parts[0][2]) if parts[0][3] else None
        parts = [(ptr, to_host(words), table if bits else to_host(aux), bits) for ptr, words, aux, bits in parts]
    block = np.empty((1, 1), dtype=object)
    block[0, 0] = parts
    return block


def _record_sizes(parts):
    return None if parts is None else [words.nbytes + aux.nbytes + offsets.nbytes for offsets, words, aux, _ in parts]


def _gene_bytes(x, labels):
    if isinstance(x, cp.ndarray):
        return cp.count_nonzero(x, axis=0).get()[None, :] * 6
    bits = max(1, int(labels.max(initial=0)).bit_length())
    record_bytes = 4 if range_base(x.data, bits) is not None else 6
    return cp.bincount(x.indices, minlength=x.shape[1]).get()[None, :] * record_bytes


def _sample_gene_sizes(client, X, codes, concurrency):
    heights = np.asarray(X.chunks[0])
    nonempty = np.flatnonzero(heights)
    n = min(len(nonempty), concurrency)
    sample = nonempty[(2 * np.arange(n) + 1) * len(nonempty) // (2 * n)]
    sizes = X.map_blocks(_gene_bytes, codes, chunks=(1, X.chunks[1]), meta=np.empty((0, 0), np.int64))
    tasks = sizes.to_delayed()[sample, 0].tolist()
    sampled = sum(client.gather(client.compute(tasks)))
    return np.r_[0.0, np.cumsum(sampled * (X.shape[0] / heights[sample].sum()) + 4 * len(heights))]


def _rank_columns(blocks, context, options, budget):
    ptr = sum(block[0].astype(np.int64) for block in blocks)
    n_cols = len(ptr) - 1
    results = []
    start = 0
    max_entries = max(1, min(2**28, budget // 16))
    max_cols = max(1, min(4096, budget // (128 * len(context.group_sizes))))
    if isinstance(blocks[0][1], np.ndarray) and (ptr[-1] > max_entries or n_cols > max_cols):
        # Reuse row dictionaries across sort panels. Keep packed expression in RAM.
        blocks = list(blocks)
        remaining = budget
        for i, (offsets, words, aux, bits) in enumerate(blocks):
            if bits and aux.nbytes <= remaining:
                remaining -= aux.nbytes
                blocks[i] = offsets, words, _on_device(aux), bits
    while start < n_cols:
        end = int(np.searchsorted(ptr, ptr[start] + max_entries, side="right") - 1)
        end = min(start + max_cols, max(start + 1, end))
        ranked = rank_records(blocks, start, end, context, options)
        if ranked is None:
            return None
        if not results:
            results = [np.empty((a.shape[0], n_cols), dtype=a.dtype, order="F") for a in ranked]
        for result, values in zip(results, ranked, strict=True):
            values.get(out=result[:, start:end])
        start = end
    return results


def _set_pts(rg, counts, n_obs):
    sizes = rg.group_sizes[:, None]
    rg.pts = counts[: len(sizes)] / sizes
    if rg.ireference is None:
        rg.pts_rest = (counts.sum(axis=0) - counts[: len(sizes)]) / (n_obs - sizes)


def rank_large_groups(rows, rg, options):
    """Use integer rank counts when a float64 rank sum can lose a half-rank."""
    width = rows[0].shape[1]
    step = max(1, 2**26 // width)
    blocks = []
    start = 0
    for row in rows:
        for i in range(0, row.shape[0], step):
            chunk = row[i : i + step]
            if chunk.dtype.kind in "biu":
                chunk = chunk.astype(np.float64)
            labels = cp.asarray(rg.group_codes[start : start + chunk.shape[0]], cp.uint16)
            blocks.append(csc_records(chunk, labels))
            start += chunk.shape[0]
    rg.n_obs = start
    arrays = _rank_columns(blocks, rg, options, start * width * np.dtype(np.float32).itemsize)
    if arrays is None:
        return None
    if rg.comp_pts:
        _set_pts(rg, arrays.pop(), start)
    stats = dict(zip(("scores", "pvals", "logfoldchanges"), arrays, strict=True))
    return stats | {name: v for name in ("pts", "pts_rest") if (v := getattr(rg, name)) is not None}


def _project_window(X, codes, edges, *, client, workers, concurrency, budget, on_host=False, selected=None):
    import dask  # type: ignore[import-not-found]
    from dask.local import identity  # type: ignore[import-not-found]
    from distributed import as_completed, wait  # type: ignore[import-not-found]

    columns = [[] for _ in range(len(edges) - 1)]
    retained = np.zeros(len(columns), np.int64)
    sources = list(dict.fromkeys(workers)) if isinstance(X._meta, cp.ndarray) else []
    pending = {}
    packed = X.map_blocks(_pack_row, codes, edges=edges, on_host=on_host, chunks=(1, 1), meta=np.empty((0, 0), object))
    (packed,) = dask.optimize(packed)
    rows = range(X.numblocks[0]) if selected is None else selected
    todo = iter(rows)
    completed = as_completed(loop=client.loop, with_results=True)

    def submit():
        if (row := next(todo, None)) is not None:
            # Unbox the native arrays on the worker so Dask accounts for their bytes.
            source = [sources[row % len(sources)]] if sources else None
            task = client.compute(packed[row, 0].to_delayed().item(), workers=source)
            receipt = client.submit(_record_sizes, task)
            pending[receipt] = task
            completed.add(receipt)

    for _ in range(min((len(rows) + 1) // 2, concurrency)):
        submit()
    observed = 0
    for receipt, sizes in completed:
        part = pending.pop(receipt)
        if sizes is None:
            wait(list(pending))
            return None, 0
        retained += sizes
        observed += 1
        if retained.max() > budget:
            wait(list(pending))
            break
        for i, (column, worker) in enumerate(zip(columns, workers, strict=False)):
            piece = client.submit(getitem, part, i)
            # Place only the projected piece on the GPU that will rank its genes.
            column.append(client.submit(identity, piece, workers=[worker], pure=False))
        submit()
    # Predict the next width from the actual sizes of the completed rows.
    width = int(edges[-1] - edges[0])
    estimate = max(1, int(width * (0.9 * budget / max(1, retained.max())) * (observed / len(rows))))
    if retained.max() > budget:
        return [], min(max(1, width - 1), estimate)
    return columns, max(width, min(2 * width, estimate))


def rank_record_windows(rg, options, client, workers, budget, *, host_budget):
    import dask
    import dask.array as da  # type: ignore[import-not-found]

    window_budget = host_budget or budget
    X = rg.X
    lookahead = (
        not host_budget
        and budget >= 16 * 2**30
        and X.numblocks[0] >= 4
        and X.shape[1] >= 2 * len(workers)
        and dask.config.get("rapids-singlecell.wilcoxon.batch-size", default=None) is None
    )
    per_worker = 7 if lookahead else 3
    if isinstance(X._meta, cp.ndarray):
        row_bytes = max(X.chunks[0]) * X.shape[1] * X.itemsize
        per_worker = min(per_worker, max(1, budget // max(1, 2 * row_bytes)))
    concurrency = min(sum(min(n, per_worker) for n in workers.values()), (X.numblocks[0] + 1) // 2)
    workers = list(workers)
    codes = da.from_array(rg.group_codes.astype(np.uint16)[:, None], chunks=(X.chunks[0], 1))
    project = partial(_project_window, X, codes, client=client, concurrency=concurrency, on_host=bool(host_budget))
    fields = ("group_sizes", "ireference", "mean_in_log_space", "_log1p_base", "comp_pts")
    context = SimpleNamespace(n_obs=X.shape[0], **{name: getattr(rg, name) for name in fields})
    results = []
    start = 0
    max_width = (X.shape[1] + 1) // 2
    width = max_width
    prefix = None
    capacity = 0.93 * window_budget
    saved = None
    if lookahead:
        # Reuse two windows for up to two thirds of the row chunks, with a byte cap on each shard.
        rows = np.arange(X.numblocks[0])
        reserve = 2 * budget // 3
        left = np.linspace(0, max_width, len(workers) + 1, dtype=int)
        right = np.linspace(max_width, X.shape[1], len(workers) + 1, dtype=int)
        edges = np.r_[left, right[1:]]
        with suppress(MemoryError):
            saved, _ = project(edges, workers=workers * 2, budget=reserve, selected=rows[rows % 3 != 0])
    while start < X.shape[1]:
        stop = min(start + width, X.shape[1])
        if prefix is None:
            edges = np.linspace(start, stop, min(len(workers), stop - start) + 1, dtype=int)
        else:
            # Balance the workers by estimated bytes when genes vary widely in density.
            upper = prefix[start] + capacity * len(workers)
            stop = min(stop, max(start + 1, int(np.searchsorted(prefix, upper, side="right")) - 1))
            cuts = np.linspace(prefix[start], prefix[stop], min(len(workers), stop - start) + 1)
            edges = np.unique(np.searchsorted(prefix, cuts))
        columns = None
        if saved:
            with suppress(MemoryError):
                columns, next_width = project(edges, workers=workers, budget=budget - reserve, selected=rows[::3])
            if columns:
                columns = [old + new for old, new in zip(saved[: len(columns)], columns, strict=True)]
                saved = saved[len(columns) :]
            else:
                saved = None
        if not columns:
            columns, next_width = project(edges, workers=workers, budget=window_budget)
        if columns is None or (not columns and stop - start == 1):
            return None
        if not columns:
            if prefix is None:
                prefix = _sample_gene_sizes(client, X, codes, concurrency)
            else:
                capacity *= min(0.9, next_width / (stop - start))
                width = next_width
            continue
        width = max_width
        capacity = min(1.5 * window_budget, capacity * next_width / (stop - start))
        jobs = [
            client.submit(_rank_columns, column, context, options, budget, workers=[worker], pure=False)
            for column, worker in zip(columns, workers, strict=False)
        ]
        batches = client.gather(jobs)
        del columns, jobs
        if any(batch is None for batch in batches):
            return None
        for i, arrays in enumerate(batches):
            if not results:
                results = [np.empty((a.shape[0], X.shape[1]), dtype=a.dtype) for a in arrays]
            for out, part in zip(results, arrays, strict=True):
                out[:, edges[i] : edges[i + 1]] = part
        start = stop
    if rg.comp_pts:
        _set_pts(rg, results.pop(), X.shape[0])
    return results
