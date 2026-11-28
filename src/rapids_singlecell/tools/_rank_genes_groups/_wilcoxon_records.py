"""Lossless (expression, group) records for a feature-major Wilcoxon panel."""

from __future__ import annotations

from functools import partial

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csc_matrix, isspmatrix_csr

from rapids_singlecell._cuda import _wilcoxon_cuda as _wc

from ._wilcoxon import _log1p_scale, _logfoldchanges_from_means, _wilcoxon_z_p


def range_base(data, bits):
    """Return the base when the value bits and a group code fit losslessly in one word."""
    if data.dtype != cp.float32 or not data.size or not 1 <= bits <= 6:
        return None
    values = data.view(cp.uint32)
    extrema = cp.stack((values.min(), values.max()))
    low, high = map(int, extrema.get())
    return extrema[:1] if high - low < 1 << (32 - bits) else None


def csc_records(chunk, labels):
    """Gather raw float32 words and row group codes in column order."""
    part = csc_matrix(cp.asarray(chunk) if isinstance(chunk, np.ndarray) else chunk)
    if not isinstance(chunk, np.ndarray | cp.ndarray) and not part.has_canonical_format:
        part = part.copy()
        part.sum_duplicates()
    offsets = part.indptr.get().astype(np.int32, copy=False)
    words = part.data.astype(cp.float32, copy=False).view(cp.uint32)
    return offsets, words, labels[part.indices], 0


def pack_records(x, labels, bounds):
    """Pack a GPU row block into independent gene shards with bounded scratch."""
    rows, cols = x.shape
    if _wc is None or x.dtype != cp.float32 or labels.dtype != np.uint16 or len(labels) != rows or rows >= 2**31:
        return None
    shards = list(zip(bounds[:-1], bounds[1:], strict=True))
    if isinstance(x, cp.ndarray):
        if any(rows * int(hi - lo) >= 2**31 for lo, hi in shards):
            return None
        labels = cp.asarray(labels)
        pieces = [csc_records(x[:, lo:hi], labels) for lo, hi in shards]
        cp.cuda.get_current_stream().synchronize()  # Dask can transfer these arrays on another stream.
        return pieces
    csr = x
    if not isspmatrix_csr(csr) or csr.nnz >= 2**31 or cols > 128 * 65535:
        return None
    if not csr.has_canonical_format:
        csr = csr.copy()
        csr.sum_duplicates()
    empty = cp.empty(0, cp.uint32)
    no_codes = empty.view(cp.uint16)
    if not csr.nnz:
        return [(np.zeros(hi - lo + 1, np.int32), empty, no_codes, 0) for lo, hi in shards]
    tiles = (rows + 31) // 32
    budget = max(1, min(2**25, csr.nnz) // cols)  # Cap the stripe prefix at 128 MiB.
    tiles_per = (tiles + budget - 1) // budget
    stripes = (tiles + tiles_per - 1) // tiles_per
    prefix = cp.empty((cols, stripes), cp.int32)
    bits = max(1, int(labels.max(initial=0)).bit_length())
    stream = cp.cuda.get_current_stream()
    failed = cp.zeros(1, cp.int32)
    indptr = cp.ascontiguousarray(csr.indptr, dtype=cp.int32)
    indices = cp.ascontiguousarray(csr.indices, dtype=cp.int32)
    values = cp.ascontiguousarray(csr.data)
    labels = cp.asarray(labels, order="C")
    transpose = partial(_wc.transpose_csr, indptr, indices, values, labels, tiles_per, prefix, failed)
    begin, end = (int(bounds[0]), int(bounds[-1]))
    if begin:
        prefix[begin - 1, -1] = 0
    transpose(begin, end, 0, empty, no_codes, empty, -1, stream.ptr)
    active = prefix[begin:end]
    cp.cumsum(active.ravel(), dtype=cp.int32, out=active.ravel())
    offsets = np.r_[np.int32(0), active[:, -1].get()]
    table = range_base(csr.data, bits)
    if table is None:
        slots = 1 << min(32 - bits, 22, max(1, (int(offsets[-1]) // 64).bit_length()))
        shard_entries = np.diff(offsets[np.asarray(bounds) - begin])
        table = cp.zeros(slots, cp.uint32) if (shard_entries >= 2 * slots).all() else empty
    while True:
        mode = bits if table.size else 0
        pieces = []
        for lo, hi in shards:
            ptr = offsets[lo - begin : hi - begin + 1] - offsets[lo - begin]
            words = cp.empty(int(ptr[-1]), cp.uint32)
            codes = no_codes if mode else cp.empty(words.size, cp.uint16)
            if lo < hi:
                transpose(lo, hi, offsets[lo - begin], words, codes, table, mode, stream.ptr)
            pieces.append((ptr, words, table if mode else codes, mode))
        if table.size <= 1:
            stream.synchronize()  # Dask can transfer these arrays on another stream.
        elif failed.item():
            del pieces, words, codes  # Release the failed dictionary before allocating raw shards.
            table = empty
            continue
        return pieces


def _on_device(a):
    if isinstance(a, np.ndarray):
        out = cp.empty(a.shape, a.dtype)
        out.set(a)
        return out
    return a


def _gather_keys(blocks, lo, hi, group_count):
    source = np.stack([b[0][lo : hi + 1] for b in blocks])
    counts = np.diff(source)
    cumulative = counts.cumsum(axis=0, dtype=np.int64)
    ptr = np.r_[0, cumulative[-1].cumsum()]
    if ptr[-1] > np.iinfo(np.int32).max:
        return None, None
    dest = cp.asarray(ptr[:-1] + cumulative - counts, dtype=cp.int32)
    offsets = cp.asarray(source - source[:, :1])
    keys = cp.empty(int(ptr[-1]), cp.uint64)
    invalid = cp.zeros(1, cp.int32)
    stream = cp.cuda.get_current_stream()
    no_table = cp.empty(0, cp.uint32)
    no_codes = no_table.view(cp.uint16)
    for i, (_, words, aux, bits) in enumerate(blocks):
        span = slice(source[i, 0], source[i, -1])
        table = _on_device(aux) if bits else no_table
        labels = no_codes if bits else _on_device(aux[span])
        _wc.gather_rank_keys(
            _on_device(words[span]), table, labels, bits, offsets[i], dest[i], keys, invalid, group_count, stream.ptr
        )
    return ptr.astype(np.int32), None if invalid.item() else keys


def rank_records(blocks, lo, hi, rg, options):
    """Rank one full-height gene panel, including cells excluded from the target groups."""
    sizes = np.asarray(rg.group_sizes)
    outside = rg.n_obs - int(sizes.sum(dtype=np.int64))
    all_sizes = np.append(sizes, outside) if outside else sizes
    ref = rg.ireference
    shape = (hi - lo, len(all_sizes))
    if _wc is None or outside < 0 or (shape[0] - 1).bit_length() + max(1, (shape[1] - 1).bit_length()) > 32:
        return None
    indptr, keys = _gather_keys(blocks, lo, hi, len(all_sizes))
    if keys is None:
        return None
    sums = cp.zeros(shape, cp.float64)
    twice_u = cp.zeros(shape, cp.uint64)
    ties = cp.zeros(shape[0] if ref is None else shape, cp.float64)
    correct = options["tie_correct"]
    need_counts = rg.comp_pts or (ref is not None and correct)
    nonzero = cp.zeros(shape if need_counts else 0, cp.int32)
    scale = 0.0 if rg.mean_in_log_space else _log1p_scale(rg)
    population = rg.n_obs if ref is None else int(sizes[ref])
    ref_code = -1 if ref is None else int(ref)
    indptr = cp.asarray(indptr, cp.int32)
    group_sizes = cp.asarray(all_sizes, cp.int32)
    stream = cp.cuda.get_current_stream().ptr
    _wc.rank_panel(
        keys, indptr, group_sizes, sums, twice_u, ties, nonzero, population, ref_code, correct, scale, stream
    )

    targets = slice(len(sizes)) if ref is None else np.delete(np.arange(len(sizes)), ref)
    n = cp.asarray(sizes[targets], cp.int64)[:, None]
    other = population - n if ref is None else population
    selected = sums[:, targets].T
    reference = (sums.sum(axis=1) - selected) / other if ref is None else sums[:, ref] / other
    variance = (ties if ref is None else ties[:, targets].T) * n * other * ((n + other + 1) / 12)
    u2 = twice_u.view(cp.int64)[:, targets].T
    z, p = _wilcoxon_z_p((u2 - n * other) * 0.5, variance, use_continuity=options["use_continuity"])
    logfoldchanges = _logfoldchanges_from_means(rg, selected / n, reference).astype(cp.float32)
    result = (u2 * 0.5 if options["return_u_values"] else z, p, logfoldchanges)
    return (*result, nonzero.T) if rg.comp_pts else result
