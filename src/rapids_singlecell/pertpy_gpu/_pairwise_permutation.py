"""Blockwise ground distances and exact permutation reductions."""

from __future__ import annotations

from math import isqrt

import cupy as cp
import numpy as np
from cuml.metrics import pairwise_distances

from ._metrics._base_metric import _as_gpu_data, _is_sparse
from ._reference_energy import offdiagonal_energy


def _wide_count_block(left, right, *, same):
    from ._metrics._edistance import _direct_row_distances

    # Bound the feature-wise temporaries used to subtract wide integers exactly.
    step = max(1, isqrt((1 << 20) // max(1, left.shape[1])))
    out = cp.empty((left.shape[0], right.shape[0]), dtype=cp.float64)

    def dense(rows):
        return cp.asarray(rows.toarray() if _is_sparse(rows) else rows)

    for i in range(0, left.shape[0], step):
        a = dense(left[i : i + step])
        for j in range(0, i + 1 if same else right.shape[0], step):
            b = a if same and i == j else dense(right[j : j + step])
            tile = _direct_row_distances(a[:, None, :], b[None, :, :])
            out[i : i + step, j : j + step] = tile
            if same and i != j:
                out[j : j + step, i : i + step] = tile.T
    return out


def _append_zero_feature(block):
    # cuML's sparse kernels require at least one stored entry on each side.
    # A shared all-zero feature leaves all supported sparse distances intact.
    data, indices, indptr = block.data, block.indices, block.indptr
    if not block.nnz:
        data = cp.zeros(1, dtype=block.dtype)
        indices = cp.asarray([block.shape[1]], dtype=indices.dtype)
        indptr = cp.zeros_like(indptr)
        indptr[-1] = 1
    return type(block)((data, indices, indptr), shape=(block.shape[0], block.shape[1] + 1))


def cell_distance_blocks(cells, metric):
    """Return a rectangular block reader with consistent sparse, integer and diagonal handling."""
    sparse = _is_sparse(cells)
    integer64 = cells.dtype.kind in "iu" and cells.dtype.itemsize == 8
    if sparse:
        cells = cells.tocsr()
        if not cells.has_canonical_format:
            cells = cells.astype(cells.dtype if integer64 else np.float64, copy=True)
            cells.sum_duplicates()
    values = cells.data if sparse else cells
    euclidean = metric in ("euclidean", "l2")
    wide = euclidean and integer64 and values.size and (int(values.max()) > 2**53 or int(values.min()) < -(2**53))
    if sparse and euclidean:
        metric = "minkowski"

    def read(i, stop_i, j, stop_j):
        same = i == j and stop_i == stop_j
        left = cells[i:stop_i]
        right = left if same else cells[j:stop_j]
        if wide:
            result = _wide_count_block(left, right, same=same)
        else:
            left = _as_gpu_data(left).astype(cp.float64, copy=False)
            right = left if same else _as_gpu_data(right).astype(cp.float64, copy=False)
            if sparse and (not left.nnz or not right.nnz):
                left = _append_zero_feature(left)
                right = left if same else _append_zero_feature(right)
            result = cp.asarray(pairwise_distances(left, right, metric=metric))
            if metric == "cosine":
                zeros = cp.asarray((left != 0).sum(axis=1)).ravel() == 0
                result[zeros, :] = 1
                result[:, zeros if same else cp.asarray((right != 0).sum(axis=1)).ravel() == 0] = 1
        if same:
            cp.fill_diagonal(result, 0)
        return result

    return read


def _mask_windows(batches, n):
    # Reuse each tile across as many independent shuffles as fit. Keeping the
    # original draws intact preserves CuPy's shape-dependent RNG sequence.
    pending = []
    first = count = 0
    for start, masks in batches:
        if pending and (count + len(masks)) * n > 64 << 20:
            yield first, cp.concatenate(pending) if len(pending) > 1 else pending[0]
            pending = []
            first, count = start, 0
        pending.append(masks)
        count += len(masks)
    if pending:
        yield first, cp.concatenate(pending) if len(pending) > 1 else pending[0]


def tiled_scores(read, membership, batches, *, n_perms, block_size, symmetric, statistic):
    """Evaluate all shuffles of each distance tile before releasing it."""
    n = len(membership)
    k = int(membership.sum())
    m = n - k
    step = min(n, block_size)
    scores = cp.empty(n_perms + 1, dtype=cp.float64)
    # An off-diagonal shift keeps equidistant cells exactly tied in the U statistic.
    offset = read(0, 1, 1, 2)[0, 0].copy() if statistic == "contrast" else 0.0
    rows = cp.zeros(n, dtype=cp.float64)
    cols = rows if symmetric else cp.zeros_like(rows)
    for start, masks in _mask_windows(batches, n):
        # wᵀDw gives the biased E statistic directly. For U and the directed
        # cross mean, reduce a binary indicator against D and its row/column sums.
        quadratic = cp.zeros(len(masks), dtype=cp.float64)
        if statistic != "edistance" and k > m:
            masks = ~masks
        for i in range(0, n, step):
            si = slice(i, min(n, i + step))
            for j in range(0, i + 1 if symmetric else n, step):
                sj = slice(j, min(n, j + step))
                tile = read(i, si.stop, j, sj.stop)
                if statistic == "contrast":
                    tile -= offset
                    if i == j:
                        cp.fill_diagonal(tile, 0)
                mirror = symmetric and i != j
                if start == 0 and statistic != "edistance":
                    rows[si] += tile.sum(axis=1)
                    if mirror or not symmetric:
                        cols[sj] += tile.sum(axis=0)
                # Limit FP64 label panels and GEMM output independently of the tile.
                chunk = max(1, (16 << 20) // (8 * step))
                for p in range(0, len(masks), chunk):
                    window = masks[p : p + chunk]
                    if statistic == "edistance":
                        left = cp.where(window[:, si], 1.0 / k, -1.0 / m)
                        right = left if i == j else cp.where(window[:, sj], 1.0 / k, -1.0 / m)
                    else:
                        left = window[:, si].astype(cp.float64)
                        right = left if i == j else window[:, sj].astype(cp.float64)
                    products = left @ tile
                    products *= right
                    quadratic[p : p + len(window)] += (2 if mirror else 1) * products.sum(axis=1)
                del tile
        if statistic == "edistance":
            scores[start : start + len(masks)] = -quadratic
            continue
        # Sum marginals in panels so the full float64 label matrix is never required.
        selected_rows = cp.zeros_like(quadratic)
        selected_cols = selected_rows if symmetric else cp.zeros_like(quadratic)
        width = max(1, (16 << 20) // (8 * len(masks)))
        for i in range(0, n, width):
            s = slice(i, i + width)
            weights = masks[:, s].astype(cp.float64)
            selected_rows += weights @ rows[s]
            if not symmetric:
                selected_cols += weights @ cols[s]
        cross = (selected_rows if k <= m else selected_cols) - quadratic
        if statistic == "contrast":
            other = rows.sum() - selected_rows - selected_cols + quadratic
            batch = offdiagonal_energy(cross, quadratic, other, min(k, m), max(k, m), offset=offset)
        else:
            batch = cross / (k * m)
        scores[start : start + len(masks)] = batch
    return scores.get()
