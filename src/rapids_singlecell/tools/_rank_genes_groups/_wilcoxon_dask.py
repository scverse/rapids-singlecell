"""Distributed exact ranking of bounded windows from row-chunked GPU CSR."""

from __future__ import annotations

from operator import getitem
from types import SimpleNamespace

import numpy as np

from ._wilcoxon_records import pack_records, range_base, rank_records


def _record_sizes(parts):
    return None if parts is None else [words.nbytes + aux.nbytes + offsets.nbytes for offsets, words, aux, _ in parts]


def _gene_bytes(csr, labels):
    import cupy as cp

    bits = max(1, int(labels.max(initial=0)).bit_length())
    record_bytes = 4 if range_base(csr.data, bits) is not None else 6
    return cp.bincount(csr.indices, minlength=csr.shape[1]).get() * record_bytes


def _sample_gene_sizes(client, rows, codes, concurrency):
    import dask  # type: ignore[import-not-found]

    # One bounded wave of row chunks estimates how many records each gene contributes.
    nonempty = np.flatnonzero([len(c) for c in codes])
    n = min(len(nonempty), concurrency)
    sample = nonempty[(2 * np.arange(n) + 1) * len(nonempty) // (2 * n)]
    tasks = [dask.delayed(_gene_bytes)(rows[i], codes[i]) for i in sample]
    sampled = sum(client.gather(client.compute(tasks)))
    scale = sum(map(len, codes)) / sum(len(codes[i]) for i in sample)
    return np.r_[0.0, np.cumsum(sampled * scale + 4 * len(rows))]


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


def _set_pts(rg, counts, n_obs):
    sizes = rg.group_sizes[:, None]
    rg.pts = counts[: len(sizes)] / sizes
    if rg.ireference is None:
        rg.pts_rest = (counts.sum(axis=0) - counts[: len(sizes)]) / (n_obs - sizes)


def rank_large_groups(rows, rg, options):
    """Use integer rank counts when a float64 rank sum can lose a half-rank."""
    import cupy as cp
    import cupyx.scipy.sparse as cpsp

    width = rows[0].shape[1]
    step = max(1, 2**26 // width)
    blocks = []
    start = 0
    for row in rows:
        for i in range(0, row.shape[0], step):
            chunk = row[i : i + step]
            if chunk.dtype.kind in "biu":
                chunk = chunk.astype(np.float64)
            part = cpsp.csc_matrix(cp.asarray(chunk) if isinstance(chunk, np.ndarray) else chunk)
            if not part.has_canonical_format:
                part = part.copy()
                part.sum_duplicates()
            labels = cp.asarray(rg.group_codes[start : start + part.shape[0]], cp.uint16)
            offsets = part.indptr.get().astype(np.int32, copy=False)
            words = part.data.astype(cp.float32, copy=False).view(cp.uint32)
            blocks.append((offsets, words, labels[part.indices], 0))
            start += part.shape[0]
    rg.n_obs = start
    arrays = _rank_columns(blocks, rg, options, start * width * np.dtype(np.float32).itemsize)
    if arrays is None:
        return None
    if rg.comp_pts:
        _set_pts(rg, arrays.pop(), start)
    stats = dict(zip(("scores", "pvals", "logfoldchanges"), arrays, strict=True))
    return stats | {name: v for name in ("pts", "pts_rest") if (v := getattr(rg, name)) is not None}


def _project_window(rows, codes, edges, *, client, workers, concurrency, budget):
    import dask
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
    observed = 0
    for receipt in completed:
        sizes = receipt.result()
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
    # Estimate the next width from measured bytes, including dictionaries. The
    # budget check above still applies when later genes or row blocks are denser.
    width = int(edges[-1] - edges[0])
    estimate = max(1, int(width * (0.9 * budget / max(1, retained.max())) * (observed / len(rows))))
    if retained.max() > budget:
        return [], min(max(1, width - 1), estimate)
    return columns, max(width, min(2 * width, estimate))


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
    max_width = (X.shape[1] + 1) // 2
    width = max_width
    prefix = None
    capacity = 0.93 * budget
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
        columns, next_width = _project_window(
            rows, codes, edges, client=client, workers=workers, concurrency=concurrency, budget=budget
        )
        if columns is None or (not columns and stop - start == 1):
            return None
        if not columns:
            if prefix is None:
                prefix = _sample_gene_sizes(client, rows, codes, concurrency)
            else:
                capacity *= min(0.9, next_width / (stop - start))
                width = next_width
            continue
        width = max_width
        capacity = min(0.93 * budget, capacity * next_width / (stop - start))
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
