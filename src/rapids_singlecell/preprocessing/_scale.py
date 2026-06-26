from __future__ import annotations

from typing import Union

import cupy as cp
import dask
import dask.array as da
import numpy as np
from anndata import AnnData
from cupyx.scipy import sparse
from scanpy._utils import view_to_actual

from rapids_singlecell._compat import (
    DaskArray,
    _meta_dense,
    _meta_sparse,
)
from rapids_singlecell.get import _check_mask, _get_obs_rep, _set_obs_rep
from rapids_singlecell.preprocessing._utils import (
    _check_gpu_X,
    _get_mean_var,
    _sparse_to_dense,
)


def scale(
    adata: AnnData,
    *,
    zero_center: bool = True,
    max_value: float | None = None,
    copy: bool = False,
    layer: str | None = None,
    obsm: str | None = None,
    mask_obs: np.ndarray | str | None = None,
    inplace: bool = True,
) -> Union[sparse.spmatrix, cp.ndarray, None]:  # noqa: UP007
    """
    Scales matrix to unit variance and clips values

    Parameters
    ----------
        adata
            AnnData object

        zero_center
            If `False`, omit zero-centering variables, which allows to handle sparse
            input efficiently.

        max_value
            Clip (truncate) to this value after scaling. If `None`, do not clip.

        copy
            Whether this function should be performed inplace. If an AnnData object
            is passed, this also determines if a copy is returned.

        layer
            If provided, which element of layers to scale.

        obsm
            If provided, which element of obsm to scale.

        mask_obs
            Restrict both the derivation of scaling parameters and the scaling itself
            to a certain set of observations. The mask is specified as a boolean array
            or a string referring to an array in :attr:`~anndata.AnnData.obs`. If the matrix is in csc format and a mask is provided, the matrix will be transformed to csr format.

        inplace
            If True, update AnnData with results. Otherwise, return results. See below for details of what is returned.


    Returns
    -------
    Returns a scaled copy or updates `adata` with a scaled version of the original :attr:`~anndata.AnnData.X` and `adata.layers['layer']`, \
    depending on `inplace`.

    """
    if copy:
        if not inplace:
            raise ValueError("`copy=True` cannot be used with `inplace=False`.")
        adata = adata.copy()

    if isinstance(adata, AnnData):
        view_to_actual(adata)

    X = _get_obs_rep(adata, layer=layer, obsm=obsm)
    _check_gpu_X(X, allow_dask=True)

    str_mean_std = ("mean", "std")
    if mask_obs is not None:
        if isinstance(mask_obs, str):
            str_mean_std = (f"mean of {mask_obs}", f"std of {mask_obs}")
        else:
            str_mean_std = ("mean with mask", "std with mask")
        mask_obs = _check_mask(adata, mask_obs, "obs")

    if isinstance(X, DaskArray):
        X, means, std = _scale_dask(
            X,
            mask_obs=mask_obs,
            zero_center=zero_center,
            inplace=inplace,
            max_value=max_value,
        )

    elif isinstance(X, cp.ndarray):
        X, means, std = _scale_array(
            X,
            mask_obs=mask_obs,
            zero_center=zero_center,
            inplace=inplace,
            max_value=max_value,
        )
    elif isinstance(X, sparse.csr_matrix):
        X, means, std = _scale_sparse_csr(
            X,
            mask_obs=mask_obs,
            zero_center=zero_center,
            inplace=inplace,
            max_value=max_value,
        )
    elif isinstance(X, sparse.csc_matrix):
        X, means, std = _scale_sparse_csc(
            X,
            mask_obs=mask_obs,
            zero_center=zero_center,
            inplace=inplace,
            max_value=max_value,
        )

    if inplace:
        _set_obs_rep(adata, X, layer=layer, obsm=obsm)
        adata.var[str_mean_std[0]] = means.get()
        adata.var[str_mean_std[1]] = std.get()

    if copy:
        return adata
    elif not inplace:
        return X


def _scale_array(X, *, mask_obs=None, zero_center=True, inplace=True, max_value=None):
    if not inplace:
        X = X.copy()
    if mask_obs is None:
        mean, var = _get_mean_var(X)
        mask_array = cp.ones(X.shape[0], dtype=cp.int32)

    else:
        mean, var = _get_mean_var(X[mask_obs, :])
        mask_array = cp.array(mask_obs).astype(cp.int32)
    X = cp.ascontiguousarray(X)

    std = cp.sqrt(var)
    std[std == 0] = 1
    max_value = _get_max_value(max_value, X.dtype)
    mean = mean.astype(X.dtype)
    std = std.astype(X.dtype)
    if zero_center:
        from rapids_singlecell._cuda import _scale_cuda as _sc

        _sc.dense_scale_center_diff(
            X,
            mean,
            std,
            mask_array,
            clipper=float(max_value),
            nrows=int(X.shape[0]),
            ncols=int(X.shape[1]),
            stream=cp.cuda.get_current_stream().ptr,
        )
    else:
        from rapids_singlecell._cuda import _scale_cuda as _sc

        _sc.dense_scale_diff(
            X,
            std,
            mask_array,
            clipper=float(max_value),
            nrows=int(X.shape[0]),
            ncols=int(X.shape[1]),
            stream=cp.cuda.get_current_stream().ptr,
        )

    return X, mean, std


def _scale_sparse_csc(
    X, *, mask_obs=None, zero_center=True, inplace=True, max_value=None
):
    if zero_center:
        X = _sparse_to_dense(X, order="C")
        # inplace is True because we copied with `toarray`
        return _scale_array(
            X,
            mask_obs=mask_obs,
            zero_center=zero_center,
            inplace=True,
            max_value=max_value,
        )
    elif mask_obs is not None:
        X = X.tocsr()
        return _scale_sparse_csr(
            X,
            mask_obs=mask_obs,
            zero_center=zero_center,
            inplace=True,
            max_value=max_value,
        )
    else:
        if not inplace:
            X = X.copy()

        mean, var = _get_mean_var(X)
        std = cp.sqrt(var)
        std[std == 0] = 1
        std = std.astype(X.dtype)
        from rapids_singlecell._cuda import _scale_cuda as _sc

        _sc.csc_scale_diff(
            X.indptr,
            X.data,
            std,
            ncols=X.shape[1],
            stream=cp.cuda.get_current_stream().ptr,
        )
        if max_value is not None:
            X.data = cp.clip(X.data, a_min=None, a_max=max_value)

        return X, mean, std


def _scale_sparse_csr(
    X, *, mask_obs=None, zero_center=True, inplace=True, max_value=None
):
    if zero_center:
        X = _sparse_to_dense(X, order="C")
        # inplace is True because we copied with `toarray`
        return _scale_array(
            X,
            mask_obs=mask_obs,
            zero_center=zero_center,
            inplace=True,
            max_value=max_value,
        )
    else:
        if not inplace:
            X = X.copy()
        if mask_obs is None:
            mean, var = _get_mean_var(X)
            mask_array = cp.ones(X.shape[0], dtype=cp.int32)

        else:
            mean, var = _get_mean_var(X[mask_obs, :])
            mask_array = cp.array(mask_obs).astype(cp.int32)
        std = cp.sqrt(var)
        std[std == 0] = 1

        max_value = _get_max_value(max_value, X.dtype)
        std = std.astype(X.dtype)
        from rapids_singlecell._cuda import _scale_cuda as _sc

        _sc.csr_scale_diff(
            X.indptr,
            X.indices,
            X.data,
            std,
            mask_array,
            clipper=float(max_value),
            nrows=X.shape[0],
            stream=cp.cuda.get_current_stream().ptr,
        )

        return X, mean, std


def _scale_dask(X, *, mask_obs=None, zero_center=True, inplace=True, max_value=None):
    if not inplace:
        X = X.copy()
    mean, var = dask.compute(
        *_get_mean_var(X[mask_obs if mask_obs is not None else slice(None), :])
    )
    if mask_obs is None:
        mask_array = cp.ones(X.shape[0], dtype=cp.int32)
    else:
        mask_array = cp.array(mask_obs).astype(cp.int32)
    std = cp.sqrt(var)
    std[std == 0] = 1
    max_value = _get_max_value(max_value, X.dtype)

    mask_array = da.from_array(
        mask_array, chunks=(X.chunks[0],), meta=_meta_dense(mask_array.dtype)
    )

    if isinstance(X._meta, sparse.csr_matrix) and zero_center:

        def __dense(X_part):
            dense = _sparse_to_dense(X_part, order="C")
            return dense

        X = X.map_blocks(
            lambda x: __dense(x),
            dtype=X.dtype,
            meta=_meta_dense(X.dtype),
        )
        scale = _scale_dask_array_zc
    elif isinstance(X._meta, sparse.csr_matrix) and not zero_center:
        scale = _scale_sparse_csr_dask
    elif isinstance(X._meta, cp.ndarray) and zero_center:
        scale = _scale_dask_array_zc
    elif isinstance(X._meta, cp.ndarray) and not zero_center:
        scale = _scale_dask_array_nzc
    else:
        raise ValueError(
            "Invalid `._meta` type only supports `cupyx.scipy.sparse.csr_matrix` and `cp.ndarray`"
        )
    return scale(X, mask_array=mask_array, mean=mean, std=std, max_value=max_value)


def _scale_dask_array_zc(X, *, mask_array, mean, std, max_value):
    from rapids_singlecell._cuda import _scale_cuda as _sc

    mean_ = mean.astype(X.dtype)
    std_ = std.astype(X.dtype)

    def __scale_kernel_center(X_part, mask_part):
        _sc.dense_scale_center_diff(
            X_part,
            mean_,
            std_,
            mask_part,
            clipper=float(max_value),
            nrows=X_part.shape[0],
            ncols=X_part.shape[1],
            stream=cp.cuda.get_current_stream().ptr,
        )
        return X_part

    X = da.blockwise(
        __scale_kernel_center,
        "ij",
        X,
        "ij",
        mask_array,
        "i",
        meta=_meta_dense(X.dtype),
        dtype=X.dtype,
    )
    return X, mean, std


def _scale_dask_array_nzc(X, *, mask_array, mean, std, max_value):
    from rapids_singlecell._cuda import _scale_cuda as _sc

    std_ = std.astype(X.dtype)

    def __scale_kernel(X_part, mask_part):
        _sc.dense_scale_diff(
            X_part,
            std_,
            mask_part,
            clipper=float(max_value),
            nrows=X_part.shape[0],
            ncols=X_part.shape[1],
            stream=cp.cuda.get_current_stream().ptr,
        )

        return X_part

    X = da.blockwise(
        __scale_kernel,
        "ij",
        X,
        "ij",
        mask_array,
        "i",
        meta=_meta_dense(X.dtype),
        dtype=X.dtype,
    )
    return X, mean, std


def _scale_sparse_csr_dask(X, *, mask_array, mean, std, max_value):
    from rapids_singlecell._cuda import _scale_cuda as _sc

    std_ = std.astype(X.dtype)

    def __scale_kernel_csr(X_part, mask_part):
        _sc.csr_scale_diff(
            X_part.indptr,
            X_part.indices,
            X_part.data,
            std_,
            mask_part,
            clipper=float(max_value),
            nrows=X_part.shape[0],
            stream=cp.cuda.get_current_stream().ptr,
        )
        return X_part

    X = da.blockwise(
        __scale_kernel_csr,
        "ij",
        X,
        "ij",
        mask_array,
        "i",
        meta=_meta_sparse(X.dtype),
        dtype=X.dtype,
    )
    return X, mean, std


def _get_max_value(val, dtype):
    if val is None:
        val = np.inf
    if dtype == cp.float32:
        val = cp.float32(val)
    else:
        val = cp.float64(val)
    return val
