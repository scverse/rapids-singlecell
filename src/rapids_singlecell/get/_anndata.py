from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Union

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csc_matrix as csc_matrix_gpu
from cupyx.scipy.sparse import csr_matrix as csr_matrix_gpu
from dask.array import Array as DaskArray
from scanpy.get import _get_obs_rep, _set_obs_rep
from scipy.sparse import csc_array as csc_array_cpu
from scipy.sparse import csc_matrix as csc_matrix_cpu
from scipy.sparse import csr_array as csr_array_cpu
from scipy.sparse import csr_matrix as csr_matrix_cpu

from rapids_singlecell._compat import (
    _meta_dense,
    _meta_dense_cpu,
    _meta_sparse,
    _meta_sparse_csc_cpu,
    _meta_sparse_csr_cpu,
)

if TYPE_CHECKING:
    from anndata import AnnData

GPU_ARRAY_TYPE = Union[cp.ndarray, csr_matrix_gpu, csc_matrix_gpu]  # noqa: UP007
CPU_ARRAY_TYPE = Union[np.ndarray, csr_matrix_cpu, csc_matrix_cpu]  # noqa: UP007


def anndata_to_GPU(
    adata: AnnData,
    layer: str | None = None,
    *,
    convert_all: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """
    Transfers matrices and arrays to the GPU

    Parameters
    ----------
    adata
        AnnData object

    layer
        Layer to use as input instead of `X`. If `None`, `X` is used.

    convert_all
        If True, move all supported arrays and matrices on the GPU

    copy
        Whether to return a copy or update `adata`.

    Returns
    -------
    Updates `adata` inplace or returns an updated copy
    """

    if copy:
        adata = adata.copy()

    if convert_all:
        anndata_to_GPU(adata)
        if adata.layers is not None:
            for key in adata.layers:
                anndata_to_GPU(adata, layer=key)
    else:
        X = _get_obs_rep(adata, layer=layer)
        error = layer if layer is not None else "X"
        X = X_to_GPU(X, warning=error)
        _set_obs_rep(adata, X, layer=layer)

    if copy:
        return adata


def X_to_GPU(
    X: CPU_ARRAY_TYPE | DaskArray, warning: str = "X"
) -> GPU_ARRAY_TYPE | DaskArray:
    """
    Transfers matrices and arrays to the GPU

    Parameters
    ----------
    X
        Matrix or array to transfer to the GPU
    warning
        Warning message to display if the input is not supported
    """
    if isinstance(X, GPU_ARRAY_TYPE):
        pass
    elif isinstance(X, DaskArray):
        if isinstance(X._meta, csr_matrix_cpu | csr_array_cpu | np.ndarray):
            meta = (
                _meta_sparse
                if isinstance(X._meta, csr_matrix_cpu | csr_array_cpu)
                else _meta_dense
            )
            X = X.map_blocks(X_to_GPU, meta=meta(X.dtype))
    elif isinstance(X, csr_array_cpu | csr_matrix_cpu):
        X = csr_matrix_gpu(
            (X_to_GPU(X.data), X_to_GPU(X.indices), X_to_GPU(X.indptr)), shape=X.shape
        )
    elif isinstance(X, csc_array_cpu | csc_matrix_cpu):
        X = csc_matrix_gpu(
            (X_to_GPU(X.data), X_to_GPU(X.indices), X_to_GPU(X.indptr)), shape=X.shape
        )
    elif isinstance(X, np.ndarray):
        X = cp.array(X)
    else:
        warnings.warn(
            f"{warning} not supported for GPU conversion returning {warning}", Warning
        )
    return X


def anndata_to_CPU(
    adata: AnnData,
    layer: str | None = None,
    *,
    convert_all: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """
    Transfers matrices and arrays from the GPU

    Parameters
    ----------
    adata
        AnnData object

    layer
        Layer to use as input instead of `X`. If `None`, `X` is used.

    convert_all
        If True, move all GPU based arrays and matrices to the host memory

    copy
        Whether to return a copy or update `adata`.

    Returns
    -------
    Updates `adata` inplace or returns an updated copy
    """

    if copy:
        adata = adata.copy()

    if convert_all:
        anndata_to_CPU(adata)
        if adata.layers is not None:
            for key in adata.layers:
                anndata_to_CPU(adata, layer=key)
    else:
        X = _get_obs_rep(adata, layer=layer)
        X = X_to_CPU(X)
        _set_obs_rep(adata, X, layer=layer)

    if copy:
        return adata


def X_to_CPU(X: GPU_ARRAY_TYPE | DaskArray) -> CPU_ARRAY_TYPE | DaskArray:
    """
    Transfers matrices and arrays from the GPU

    Parameters
    ----------
    X
        Matrix or array to transfer to the host memory
    """
    if isinstance(X, DaskArray):
        if isinstance(X._meta, csr_matrix_gpu):
            meta = _meta_sparse_csr_cpu
        elif isinstance(X._meta, csc_matrix_gpu):
            meta = _meta_sparse_csc_cpu
        elif isinstance(X._meta, cp.ndarray):
            meta = _meta_dense_cpu
        else:
            meta = None
        if meta is not None:
            X = X.map_blocks(X_to_CPU, meta=meta(X.dtype))
        else:
            pass
    if isinstance(X, GPU_ARRAY_TYPE):
        X = X.get()
    else:
        pass
    return X
