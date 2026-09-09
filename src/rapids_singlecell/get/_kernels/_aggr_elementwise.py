"""Native duplicate handling and scatter operations for sparse aggregation."""

from __future__ import annotations

import cupy as cp

from rapids_singlecell._cuda import _elementwise_cuda


def _sum_duplicates_diff(row, col, *, size):
    if row.size != size or col.size != size:
        raise ValueError("row and col must have the requested size")
    diff = cp.empty(size, dtype=row.dtype)
    _elementwise_cuda.duplicates_diff(
        row, col, diff, stream=cp.cuda.get_current_stream().ptr
    )
    return diff


def _sum_duplicates_assign(src_row, src_col, index, rows, indices):
    _elementwise_cuda.duplicates_assign(
        src_row,
        src_col,
        index,
        rows,
        indices,
        stream=cp.cuda.get_current_stream().ptr,
    )
    return rows, indices


def _scatter_sum(src, index, sums):
    _elementwise_cuda.scatter(src, index, sums, stream=cp.cuda.get_current_stream().ptr)
    return sums


def _scatter_mean_var(src, index, means, var):
    _elementwise_cuda.scatter(
        src, index, means, squares=var, stream=cp.cuda.get_current_stream().ptr
    )
    return means, var


def _scatter_count_nonzero(src, index, counts):
    _elementwise_cuda.scatter(
        src, index, counts, count=True, stream=cp.cuda.get_current_stream().ptr
    )
    return counts
