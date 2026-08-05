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


def _random_state_kwargs(
    func: object,
    seed: int | np.random.RandomState,
) -> dict:
    """Build ``random_state=`` or ``rng=`` kwargs depending on the scanpy version.

    Scanpy >= 1.13 replaced the ``random_state`` parameter with ``rng``
    (a ``numpy.random.Generator``) on several internal helpers.  This function
    inspects the target callable and returns the right keyword dict so that
    rapids_singlecell works with both old and new scanpy.

    When the new API is detected we wrap the seed in scanpy's ``_LegacyRng``
    so that functions like ``sample_comb`` fall back to sklearn's
    ``sample_without_replacement``, preserving the same random stream as the
    old ``random_state`` code path.
    """
    import inspect

    sig = inspect.signature(func)
    if "rng" in sig.parameters:
        from scanpy._utils.random import _LegacyRng

        return {"rng": _LegacyRng(seed)}
    return {"random_state": seed}
