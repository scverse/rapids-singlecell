"""Lossless (expression, group) records for a feature-major Wilcoxon panel."""

from __future__ import annotations

from functools import partial

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import isspmatrix_csr

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


def pack_records(csr, labels, bounds):
    """Pack canonical CSR directly into independent gene shards with bounded scratch."""
    if _wc is None or not isspmatrix_csr(csr) or csr.dtype != cp.float32:
        return None
    rows, cols = csr.shape
    if labels.dtype != np.uint16 or len(labels) != rows or max(csr.nnz, rows) >= 2**31 or cols > 128 * 65535:
        return None
    if not csr.has_canonical_format:
        csr = csr.copy()
        csr.sum_duplicates()
    shards = list(zip(bounds[:-1], bounds[1:], strict=True))
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
    stream = cp.cuda.get_current_stream().ptr
    failed = cp.zeros(1, cp.int32)
    transpose = partial(
        _wc.transpose_csr,
        cp.ascontiguousarray(csr.indptr, dtype=cp.int32),
        cp.ascontiguousarray(csr.indices, dtype=cp.int32),
        cp.ascontiguousarray(csr.data),
        cp.asarray(labels, dtype=cp.uint16, order="C"),
        tiles_per,
        prefix,
        failed,
    )
    begin, end = (int(bounds[0]), int(bounds[-1]))
    if begin:
        prefix[begin - 1, -1] = 0
    transpose(begin, end, 0, empty, no_codes, empty, -1, stream)
    active = prefix[begin:end]
    cp.cumsum(active.ravel(), dtype=cp.int32, out=active.ravel())
    offsets = np.r_[np.int32(0), active[:, -1].get()]
    slots = 1 << min(32 - bits, 22, max(1, (int(offsets[-1]) // 64).bit_length()))
    shard_entries = np.diff(offsets[np.asarray(bounds) - begin])
    modes = (bits, 0) if (slots * 4 <= 2 * shard_entries.astype(np.int64)).all() else (0,)
    base = range_base(csr.data, bits)
    if base is not None:
        modes = (bits,)
    for mode in modes:
        table = empty
        if mode:
            table = base if base is not None else cp.zeros(slots, cp.uint32)
        pieces = []
        for lo, hi in shards:
            ptr = offsets[lo - begin : hi - begin + 1] - offsets[lo - begin]
            words = cp.empty(int(ptr[-1]), cp.uint32)
            codes = no_codes if mode else cp.empty(words.size, cp.uint16)
            if lo < hi:
                transpose(lo, hi, offsets[lo - begin], words, codes, table, mode, stream)
            pieces.append((ptr, words, table if mode else codes, mode))
        if not mode or base is not None:
            cp.cuda.get_current_stream().synchronize()  # Dask can transfer these arrays on another stream.
            return pieces
        if not failed.item():
            return pieces
        del pieces, words, codes, table  # Release the failed dictionary before allocating raw shards.


def _on_device(a):
    if isinstance(a, np.ndarray):
        # The source stays live in blocks until the gather stream synchronizes.
        out = cp.empty(a.shape, dtype=a.dtype)
        out.set(a)
        return out
    return a


def _gather_keys(blocks, lo, hi, group_count):
    source = np.stack([b[0][lo : hi + 1] for b in blocks])
    cumulative = np.cumsum(np.diff(source), axis=0, dtype=np.int64)
    ptr = np.r_[0, cumulative[-1].cumsum()]
    if ptr[-1] > np.iinfo(np.int32).max:
        return None, None
    dest = cp.asarray(ptr[:-1] + cumulative - np.diff(source), dtype=cp.int32)
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
    twice_u = cp.zeros(shape, cp.int64)
    ties = cp.zeros(shape[0] if ref is None else shape, cp.float64)
    need_counts = rg.comp_pts or (ref is not None and options["tie_correct"])
    nonzero = cp.zeros(shape if need_counts else 0, cp.int32)
    scale = 0.0 if rg.mean_in_log_space else _log1p_scale(rg)
    _wc.rank_panel(
        keys,
        cp.asarray(indptr, cp.int32),
        cp.asarray(all_sizes, cp.int32),
        sums,
        twice_u.view(cp.uint64),
        ties,
        nonzero,
        rg.n_obs if ref is None else int(sizes[ref]),
        -1 if ref is None else int(ref),
        options["tie_correct"],
        scale,
        cp.cuda.get_current_stream().ptr,
    )

    groups = np.arange(len(sizes), dtype=np.intp)
    targets = slice(len(sizes)) if ref is None else groups[groups != ref]
    n = cp.asarray(sizes[targets], cp.int64)[:, None]
    other = rg.n_obs - n if ref is None else int(sizes[ref])
    selected = sums[:, targets].T
    reference = (sums.sum(axis=1) - selected) / other if ref is None else sums[:, ref] / other
    variance = (ties if ref is None else ties[:, targets].T) * n * other * ((n + other + 1) / 12)
    u2 = twice_u[:, targets].T
    z, p = _wilcoxon_z_p((u2 - n * other) * 0.5, variance, use_continuity=options["use_continuity"])
    result = (
        groups[targets],
        u2 * 0.5 if options["return_u_values"] else z,
        p,
        _logfoldchanges_from_means(rg, selected / n, reference),
    )
    return (*result, nonzero.T) if rg.comp_pts else result
