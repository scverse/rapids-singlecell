"""Float64 dense reductions using the native Rust device backend."""

from __future__ import annotations

import cupy as cp

from rapids_singlecell._cuda import _elementwise_cuda


def _reduce(data, axis, *, square):
    data = cp.asarray(data)
    if data.dtype not in (cp.float32, cp.float64):
        data = data.astype(cp.float64)
    scalar = axis is None
    if scalar:
        data = data.reshape(1, -1)
        axis = 1
    if data.ndim != 2:
        raise ValueError("dense moment reduction expects a two-dimensional array")
    axis = int(axis)
    if axis < 0:
        axis += data.ndim
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    if not (data.flags.c_contiguous or data.flags.f_contiguous):
        data = cp.ascontiguousarray(data)
    result = cp.empty(data.shape[1 - axis], dtype=cp.float64)
    _elementwise_cuda.dense_sum(
        data,
        result,
        axis=axis,
        square=square,
        stream=cp.cuda.get_current_stream().ptr,
    )
    return result.reshape(()) if scalar else result


def sq_sum(data, axis=None):
    return _reduce(data, axis, square=True)


def mean_sum(data, axis=None):
    return _reduce(data, axis, square=False)
