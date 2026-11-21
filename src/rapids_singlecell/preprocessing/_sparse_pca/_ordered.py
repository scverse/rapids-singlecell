from __future__ import annotations

import cupy as cp

from rapids_singlecell._cuda import _spca_cuda as _spca


def _needs_centered_gram(gram, mean):
    second_moment = gram.diagonal()
    # Recompute explicitly if subtracting the squared mean would lose over
    # half the working precision of any feature variance.
    margin = cp.finfo(gram.dtype).eps ** 0.5
    return bool(cp.any(~cp.isfinite(second_moment) | ((mean != 0) & (mean**2 >= (1 - margin) * second_moment))))


def _integer_scale(x, stream):
    limits = cp.zeros(2, dtype=cp.uint32)
    _spca.gram_bounds(x.data, limits, stream)
    maximum, inverse = map(int, limits.get())
    info = cp.finfo(x.dtype)
    bias = info.maxexp - 1
    low, high = bias + 1 - inverse, maximum - bias
    # Each rounded product is a multiple of 2**unit and smaller than 2**upper.
    # Reserve enough headroom for every row to contribute with the same sign.
    unit, upper = 2 * low - info.nmant, 2 * (high + 1)
    capacity = 61 - (x.shape[0] - 1).bit_length()
    bits = 0 if upper - unit <= capacity else min(52, capacity)
    # A full unsigned low word needs a second atomic only on a carry or borrow.
    if capacity < upper - unit <= 64 and info.minexp <= -unit < info.maxexp:
        bits = 64
    if (
        capacity > 0
        and 2 * low >= info.minexp
        and upper < info.maxexp
        and upper - unit - bits <= capacity
        and (bits == 64 or info.minexp <= -unit - bits < info.maxexp)
    ):
        return unit, bits
    return None


def _ordered_gram(x):
    """Compute a reproducible upper Gram triangle for canonical CSR input."""
    p = x.shape[1]
    if not x.nnz:
        return cp.zeros((p, p), dtype=x.dtype)
    stream = cp.cuda.get_current_stream().ptr
    scale = _integer_scale(x, stream)
    if scale is not None:
        unit, bits = scale
        bins = cp.zeros((1 + bool(bits), p * (p + 1) // 2), dtype=cp.int64)
        gram = cp.empty((p, p), dtype=x.dtype)
        _spca.integer_gram(x.indptr, x.indices, x.data, bins, gram, unit, bits, stream)
        return gram
    parts = min(8, max(1, (64 << 20) // max(1, p * p * x.dtype.itemsize)))
    partial = cp.zeros((parts, p, p), dtype=x.dtype)
    by_column = x.tocsc()
    by_column.sort_indices()
    index_dtype = x.indptr.dtype
    _spca.ordered_gram(
        x.indptr,
        x.indices,
        x.data,
        by_column.indptr.astype(index_dtype, copy=False),
        by_column.indices.astype(index_dtype, copy=False),
        by_column.data,
        partial,
        stream,
    )
    return partial[0] if parts == 1 else partial.sum(axis=0)
