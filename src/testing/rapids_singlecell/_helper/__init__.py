"""Like fixtures, but more flexible"""

from __future__ import annotations

from functools import partial, singledispatch
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

import cupy as cp
import numpy as np
import pytest
from anndata.tests.helpers import asarray
from cupyx.scipy import sparse as cusparse
from cupyx.scipy.sparse import csr_matrix as CupyCSRMatrix
from cupyx.scipy.sparse import spmatrix as CupySparseMatrix
from dask.array import Array as DaskArray
from packaging.version import Version
from scipy import sparse
from scipy.sparse import sparray as CSArray
from scipy.sparse import spmatrix as CSMatrix

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Literal

    from _pytest.mark.structures import ParameterSet
    from numpy.typing import NDArray


def param_with(
    at: ParameterSet,
    *,
    marks: Iterable[pytest.Mark | pytest.MarkDecorator] = (),
    id: str | None = None,
) -> ParameterSet:
    return pytest.param(*at.values, marks=[*at.marks, *marks], id=id or at.id)


MAP_ARRAY_TYPES: dict[
    tuple[Literal["mem", "dask"], Literal["dense", "sparse"]],
    tuple[ParameterSet, ...],
] = {
    ("mem", "dense"): (pytest.param(asarray, id="numpy_ndarray"),),
    ("mem", "sparse"): (
        pytest.param(sparse.csr_matrix, id="scipy_csr"),
        pytest.param(sparse.csc_matrix, id="scipy_csc"),
    ),
}

ARRAY_TYPES_MEM = tuple(at for (strg, _), ats in MAP_ARRAY_TYPES.items() if strg == "mem" for at in ats)


def _half_chunk_size(a: tuple[int, ...]) -> tuple[int, ...]:
    def half_rounded_up(x):
        div, mod = divmod(x, 2)
        return div + (mod > 0)

    return tuple(half_rounded_up(x) for x in a)


@singledispatch
def as_dense_dask_array(a):
    import dask.array as da

    a = asarray(a)
    return da.asarray(a, chunks=_half_chunk_size(a.shape))


@as_dense_dask_array.register(CSMatrix)
def _(a):
    return as_dense_dask_array(a.toarray())


@as_dense_dask_array.register(DaskArray)
def _(a):
    # Map to numpy arrays blockwise, preserving dtype; meta informs dask about target array type
    return a.map_blocks(asarray, dtype=a.dtype, meta=np.empty((0,) * a.ndim, dtype=a.dtype))


@singledispatch
def _as_sparse_dask(
    a: NDArray | CSArray | CSMatrix | DaskArray,
    *,
    typ: type[CSArray | CSMatrix | CupyCSRMatrix],
    chunks: tuple[int, ...] | None = None,
) -> DaskArray:
    """Convert a to a sparse dask array, preserving sparse format and container (`cs{rc}_{array,matrix}`)."""
    raise NotImplementedError


@_as_sparse_dask.register(CSArray)
@_as_sparse_dask.register(CSMatrix)
@_as_sparse_dask.register(np.ndarray)
def _(
    a: CSArray | CSMatrix | NDArray,
    *,
    typ: type[CSArray | CSMatrix | CupyCSRMatrix],
    chunks: tuple[int, ...] | None = None,
) -> DaskArray:
    import dask.array as da

    chunks = _half_chunk_size(a.shape) if chunks is None else chunks
    return da.from_array(_as_sparse_dask_inner(a, typ=typ), chunks=chunks)


@_as_sparse_dask.register(DaskArray)
def _(
    a: DaskArray,
    *,
    typ: type[CSArray | CSMatrix | CupyCSRMatrix],
    chunks: tuple[int, ...] | None = None,
) -> DaskArray:
    assert chunks is None  # TODO: if needed we can add a .rechunk(chunks)
    return a.map_blocks(_as_sparse_dask_inner, typ=typ, dtype=a.dtype, meta=typ((2, 2)))


def _as_sparse_dask_inner(
    a: NDArray | CSArray | CSMatrix, *, typ: type[CSArray | CSMatrix | CupyCSRMatrix]
) -> CSArray | CSMatrix | CupyCSRMatrix:
    """Convert into a a sparse container that dask supports (or complain)."""
    if issubclass(typ, CSArray) and not DASK_CAN_SPARRAY:
        msg = "Dask <2025.3 doesn’t support sparse arrays"
        raise TypeError(msg)
    if issubclass(typ, CupySparseMatrix):
        a = as_cupy(a)
    return typ(a)


as_sparse_dask_array = partial(_as_sparse_dask, typ=sparse.csr_array)
as_sparse_dask_matrix = partial(_as_sparse_dask, typ=sparse.csr_matrix)


def as_sparse_cupy_dask_array(X):
    da = as_sparse_dask_array(X)
    da = da.rechunk((da.shape[0] // 2, da.shape[1]))
    da = da.map_blocks(cusparse.csr_matrix, dtype=X.dtype)
    return da


def as_dense_cupy_dask_array(X):
    X = as_dense_dask_array(X)
    X = X.map_blocks(cp.array)
    X = X.rechunk((X.shape[0] // 2, X.shape[1]))
    return X


# --- Internal helpers and feature flags ---

# Minimal feature detection: SciPy provides sparse arrays; dask support varies but
# we optimistically enable and fallback to spmatrix at runtime if needed.
DASK_CAN_SPARRAY = Version(version("dask")) >= Version("2025.3.0")


def as_cupy(a: NDArray | CSArray | CSMatrix) -> CupyCSRMatrix | cp.ndarray:
    """Convert numpy/scipy sparse to cupy/cupyx sparse where appropriate."""
    if isinstance(a, np.ndarray):
        return cp.asarray(a)
    if isinstance(a, CSMatrix):
        return cusparse.csr_matrix(a)
    if isinstance(a, CSArray):
        # Convert via SciPy csr_matrix then to Cupy
        return cusparse.csr_matrix(sparse.csr_matrix(a))
    return a
