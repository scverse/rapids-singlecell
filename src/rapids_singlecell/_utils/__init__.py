from __future__ import annotations

from typing import TYPE_CHECKING, Union

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csc_matrix, csr_matrix
from dask.array import Array as DaskArray

from ._multi_gpu import (
    MultiGPUFallbackWarning,
    _calculate_blocks_per_pair,
    _copy_to_device_via_host,
    _create_category_index_mapping,
    _get_device_attrs,
    _split_pairs,
    parse_device_ids,
    peer_copy_verified,
    peer_copy_works,
    validate_multi_gpu,
)

__all__ = [
    "MultiGPUFallbackWarning",
    "_calculate_blocks_per_pair",
    "_copy_to_device_via_host",
    "_create_category_index_mapping",
    "_get_device_attrs",
    "_split_pairs",
    "parse_device_ids",
    "peer_copy_verified",
    "peer_copy_works",
    "validate_multi_gpu",
]


ArrayTypes = Union[cp.ndarray, csc_matrix, csr_matrix]  # noqa: UP007
ArrayTypesDask = Union[cp.ndarray, csc_matrix, csr_matrix, DaskArray]  # noqa: UP007


def _get_logger_level(logger):
    for i in range(15):
        out = logger.should_log_for(i)
        if out:
            return i
