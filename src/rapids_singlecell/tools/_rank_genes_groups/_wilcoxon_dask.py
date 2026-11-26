"""Distributed exact ranking of bounded windows from row-chunked GPU CSR."""

from __future__ import annotations

from operator import getitem
from types import SimpleNamespace

import numpy as np

from ._wilcoxon_records import pack_records, rank_records


def _record_sizes(parts):
    return None if parts is None else [words.nbytes + aux.nbytes + offsets.nbytes for offsets, words, aux, _ in parts]


def _rank_columns(blocks, context, options, budget):
    ptr = sum(block[0].astype(np.int64) for block in blocks)
    n_cols = len(ptr) - 1
    results = []
    start = 0
    max_entries = max(1, min(2**28, budget // 16))
    max_cols = max(1, min(4096, budget // (128 * len(context.group_sizes))))
    while start < n_cols:
        end = int(np.searchsorted(ptr, ptr[start] + max_entries, side="right") - 1)
        end = min(start + max_cols, max(start + 1, end))
        ranked = rank_records(blocks, start, end, context, options)
        if ranked is None:
            return None
        if not results:
            results = [np.empty((a.shape[0], n_cols), dtype=a.dtype, order="F") for a in ranked[1:]]
        for result, values in zip(results, ranked[1:], strict=True):
            values.get(out=result[:, start:end])
        start = end
    return results


def _project_window(rows, codes, edges, *, client, workers, concurrency, budget):
    import dask  # type: ignore[import-not-found]
    from dask.local import identity  # type: ignore[import-not-found]
    from distributed import as_completed, wait  # type: ignore[import-not-found]

    columns = [[] for _ in range(len(edges) - 1)]
    retained = np.zeros(len(columns), np.int64)
    pending = {}
    todo = iter(zip(rows, codes, strict=True))
    completed = as_completed(loop=client.loop)

    def submit():
        if (item := next(todo, None)) is not None:
            row, labels = item
            task = client.compute(dask.delayed(pack_records)(row, labels, edges))
            receipt = client.submit(_record_sizes, task)
            pending[receipt] = task
            completed.add(receipt)

    for _ in range(min(len(rows), concurrency)):
        submit()
    for receipt in completed:
        sizes = receipt.result()
        part = pending.pop(receipt)
        if sizes is None or np.any((retained := retained + sizes) > budget):
            wait(list(pending))
            return None if sizes is None else []
        for i, (column, worker) in enumerate(zip(columns, workers, strict=False)):
            piece = client.submit(getitem, part, i)
            # Place only the projected piece on the GPU that will rank its genes.
            column.append(client.submit(identity, piece, workers=[worker], pure=False))
        submit()
    return columns


def rank_csr_windows(rg, options, client, workers, budget):
    X = rg.X
    concurrency = min(sum(min(n, 3) for n in workers.values()), (X.numblocks[0] + 1) // 2)
    workers = list(workers)
    rows = X.to_delayed()[:, 0]
    codes = np.split(rg.group_codes.astype(np.uint16), np.cumsum(X.chunks[0])[:-1])
    fields = ("group_sizes", "ireference", "mean_in_log_space", "_log1p_base", "comp_pts")
    context = SimpleNamespace(n_obs=X.shape[0], **{name: getattr(rg, name) for name in fields})
    results = []
    start = 0
    width = (X.shape[1] + 1) // 2
    while start < X.shape[1]:
        stop = min(start + width, X.shape[1])
        edges = np.linspace(start, stop, min(len(workers), stop - start) + 1, dtype=int)
        columns = _project_window(
            rows, codes, edges, client=client, workers=workers, concurrency=concurrency, budget=budget
        )
        if columns is None or (not columns and width == 1):
            return None
        if not columns:
            width = max(1, width // 2)
            continue
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
        counts = results.pop()
        sizes = rg.group_sizes[:, None]
        rg.pts = counts[: len(sizes)] / sizes
        if rg.ireference is None:
            rg.pts_rest = (counts.sum(axis=0) - counts[: len(sizes)]) / (X.shape[0] - sizes)
    return results
