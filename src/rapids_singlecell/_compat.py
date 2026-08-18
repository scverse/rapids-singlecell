from __future__ import annotations

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csr_matrix
from dask.array import Array as DaskArray  # noqa: F401
from scipy.sparse import csc_matrix as csc_matrix_cpu
from scipy.sparse import csr_matrix as csr_matrix_cpu


def _meta_dense(dtype):
    return cp.zeros([0], dtype=dtype)


def _meta_sparse(dtype):
    return csr_matrix(cp.array((1.0,), dtype=dtype))


def _meta_dense_cpu(dtype):
    return np.zeros([0], dtype=dtype)


def _meta_sparse_csr_cpu(dtype):
    return csr_matrix_cpu(np.array((1.0,), dtype=dtype))


def _meta_sparse_csc_cpu(dtype):
    return csc_matrix_cpu(np.array((1.0,), dtype=dtype))


def _rng_kwargs(
    func: object, rng: np.random.Generator, *, always_state: bool = False
) -> dict:
    """Build ``rng=`` or ``random_state=`` kwargs for the Scanpy version.

    Stateful legacy consumers such as ``sample_comb`` opt into sharing the
    wrapped ``RandomState``. Seed-based consumers use a scalar fallback.
    """
    import inspect

    from rapids_singlecell._utils._random import (
        _legacy_random_state,
        _LegacyRng,
        _seed_from_rng,
    )

    if "rng" not in inspect.signature(func).parameters:
        random_state = (
            _legacy_random_state(rng, always_state=True)
            if always_state
            else _seed_from_rng(rng)
        )
        return {"random_state": random_state}
    if not always_state or not isinstance(rng, _LegacyRng):
        return {"rng": rng}
    from scanpy._utils.random import _LegacyRng as _ScanpyLegacyRng

    return {"rng": _ScanpyLegacyRng(rng.state)}
