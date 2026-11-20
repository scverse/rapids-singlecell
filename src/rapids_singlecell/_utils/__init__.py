from __future__ import annotations

from typing import TYPE_CHECKING, Union

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csc_matrix, csr_matrix
from dask.array import Array as DaskArray

from ._multi_gpu import (
    _calculate_blocks_per_pair,
    _concat_on_device,
    _copies_on_device,
    _copy_to_device,
    _create_category_index_mapping,
    _get_device_attrs,
    _split_pairs,
    _sum_on_device,
    parse_device_ids,
)

__all__ = [
    "_array_to_device",
    "_calculate_blocks_per_pair",
    "_concat_on_device",
    "_copies_on_device",
    "_copy_to_device",
    "_create_category_index_mapping",
    "_get_device_attrs",
    "_split_pairs",
    "_sum_on_device",
    "parse_device_ids",
]


# Below this, a plain copy beats page-locking the host memory first.
_PIN_BYTES = 1 << 26


def _array_to_device(x: np.ndarray, dtype=None) -> cp.ndarray:
    """Page-lock large host arrays for the transfer to the current CUDA stream."""
    x = np.ascontiguousarray(x)
    if x.nbytes < _PIN_BYTES:
        return cp.asarray(x, dtype=dtype)
    rt = cp.cuda.runtime
    try:
        rt.hostRegister(x.ctypes.data, x.nbytes, 0)
    except cp.cuda.runtime.CUDARuntimeError:
        return cp.asarray(x, dtype=dtype)
    try:
        out = cp.empty(x.shape, x.dtype)
        stream = cp.cuda.get_current_stream()
        rt.memcpyAsync(out.data.ptr, x.ctypes.data, x.nbytes, rt.memcpyHostToDevice, stream.ptr)
        stream.synchronize()
    finally:
        rt.hostUnregister(x.ctypes.data)
    return out if dtype is None else out.astype(dtype, copy=False)


ArrayTypes = Union[cp.ndarray, csc_matrix, csr_matrix]  # noqa: UP007
ArrayTypesDask = Union[cp.ndarray, csc_matrix, csr_matrix, DaskArray]  # noqa: UP007


def _get_logger_level(logger):
    for i in range(15):
        out = logger.should_log_for(i)
        if out:
            return i
