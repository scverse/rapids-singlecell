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
    if 2 * low < info.minexp or upper >= info.maxexp:
        return None
    capacity = 61 - (x.shape[0] - 1).bit_length()
    product_bits = upper - unit
    if product_bits <= capacity:
        return unit, 0
    if product_bits <= 64:
        return unit, 64
    bits = min(52, capacity)
    if bits > 0 and product_bits - bits <= capacity:
        return unit, bits
    return None


def _ordered_gram(x):
    """Compute a reproducible upper Gram triangle for canonical CSR input."""
    p = x.shape[1]
    if not x.nnz:
        return cp.zeros((p, p), dtype=x.dtype)
    stream = cp.cuda.get_current_stream().ptr
    unit, bits = _integer_scale(x, stream) or (0, -1)
    groups = (p + 31) // 32
    rows = min(x.shape[0], 32768, max(1, (16 << 20) // (groups * 8)))
    index = cp.empty((groups, rows), dtype=cp.uint64)
    itemsize = x.dtype.itemsize if bits < 0 else (16 if bits else 8)
    entries = p * p if bits < 0 else p * (p + 1) // 2
    part_bytes = entries * itemsize
    parts = min(2, max(1, (64 << 20) // part_bytes))
    partial = cp.empty((parts, part_bytes), dtype=cp.uint8)
    # Reuse a single floating-point matrix. Release multi-part workspaces after reduction.
    gram = partial.view(x.dtype).reshape(p, p) if bits < 0 and parts == 1 else cp.empty((p, p), dtype=x.dtype)
    index_dtype = cp.promote_types(x.indptr.dtype, x.indices.dtype)
    _spca.ordered_gram(
        x.indptr.astype(index_dtype, copy=False),
        x.indices.astype(index_dtype, copy=False),
        x.data,
        index,
        partial,
        gram,
        unit,
        bits,
        stream,
    )
    return gram
