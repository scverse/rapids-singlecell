from __future__ import annotations

import math
from typing import Literal, Union

import cupy as cp
import dask.array as da
from anndata import AnnData
from cuml.linear_model import LinearRegression
from cupyx.scipy import sparse
from pandas.api.types import CategoricalDtype
from scanpy._utils import view_to_actual
from scanpy.get import _get_obs_rep, _set_obs_rep

from rapids_singlecell._compat import DaskArray, _meta_dense

from ._utils import _check_gpu_X, _sparse_to_dense


def regress_out(
    adata: AnnData,
    keys: str | list,
    *,
    layer: str | None = None,
    inplace: bool = True,
    batchsize: int | Literal["all"] | None = None,
    verbose: bool = False,
) -> Union[cp.ndarray, None]:  # noqa: UP007
    """
    Use linear regression to adjust for the effects of unwanted noise
    and variation.

    Parameters
    ----------
        adata
            AnnData object

        keys
            Keys for observation annotation on which to regress on.
            Supports both numerical and single categorical covariates.

        layer
            Layer to regress instead of `X`. If `None`, `X` is regressed.

        inplace
            Whether to update `adata` or return the corrected matrix of `adata.X` and `adata.layers`.

        batchsize
            Number of genes that should be processed together. \
            If `'all'` all genes will be processed together if `.n_obs` <100000. \
            If `None` each gene will be analysed separately. \
            Only used for continuous covariates.

        verbose
            Print debugging information

    Returns
    -------
    Returns a corrected copy or  updates `adata` with a corrected version of the \
    original `adata.X` and `adata.layers['layer']`, depending on `inplace`.
    """
    if batchsize != "all" and type(batchsize) not in [int, type(None)]:
        raise ValueError("batchsize must be `int`, `None` or `'all'`")

    if isinstance(adata, AnnData):
        view_to_actual(adata)

    X = _get_obs_rep(adata, layer=layer)

    _check_gpu_X(X, allow_dask=True)

    if not sparse.issparse(X) and not isinstance(X, DaskArray) and inplace is False:
        X = cp.array(X)

    # Normalize keys to list
    if isinstance(keys, str):
        keys = [keys]

    # Check for categorical covariates
    categorical = any(
        isinstance(adata.obs[key].dtype, CategoricalDtype) for key in keys
    )

    if categorical:
        if len(keys) > 1:
            raise ValueError(
                "Only a single categorical variable is supported for regress_out."
            )
        if isinstance(X, DaskArray):
            X = _regress_out_categorical_dask(X, adata.obs[keys[0]])
        else:
            X = _regress_out_categorical(X, adata.obs[keys[0]])
    else:
        if isinstance(X, DaskArray):
            X = _regress_out_continuous_dask(X, adata, keys)
        else:
            X = _regress_out_continuous(X, adata, keys, batchsize)

    if inplace:
        _set_obs_rep(adata, X, layer=layer)
    else:
        return X


def _regress_out_categorical(X, categorical_series):
    """Regress out a categorical variable by subtracting group means."""
    if sparse.issparse(X):
        X = _sparse_to_dense(X, order="C")
    else:
        X = cp.ascontiguousarray(X)

    codes = cp.array(categorical_series.cat.codes.to_numpy(), dtype=cp.int32)
    n_cats = len(categorical_series.cat.categories)

    # Compute group means via scatter_add into compact (n_cats, n_genes) matrix
    means = cp.zeros((n_cats, X.shape[1]), dtype=cp.float64)
    cp.add.at(means, codes, X)
    counts = cp.bincount(codes, minlength=n_cats).reshape(-1, 1)
    means = (means / counts).astype(X.dtype)

    # Subtract group means in-place
    X -= means[codes]
    return X


def _regress_out_continuous(X, adata, keys, batchsize):
    """Regress out continuous variables using linear regression."""
    dim_regressor = len(keys) + 1

    regressors = cp.ones(X.shape[0] * dim_regressor, dtype=X.dtype).reshape(
        (X.shape[0], dim_regressor), order="F"
    )
    for i in range(len(keys)):
        regressors[:, i + 1] = cp.array(adata.obs[keys[i]], dtype=X.dtype).ravel()

    # Set default batch size based on the number of samples in X
    DEFAULT_GENE_BATCH = 100
    BATCH_CELL_THRESHOLD = 100_000
    if batchsize is None:
        batchsize = DEFAULT_GENE_BATCH if X.shape[0] > BATCH_CELL_THRESHOLD else "all"

    # Validate the choice of "all" batch size
    if batchsize == "all":
        if cp.linalg.det(regressors.T @ regressors) == 0:
            batchsize = DEFAULT_GENE_BATCH

    # Do regression
    if batchsize == "all":
        if sparse.issparse(X):
            X = _sparse_to_dense(X, order="C")
        else:
            X = cp.ascontiguousarray(X)
        inv_gram_matrix = cp.linalg.inv(regressors.T @ regressors)
        coeff = inv_gram_matrix @ (regressors.T @ X)
        cp.cublas.gemm("N", "N", regressors, coeff, alpha=-1, beta=1, out=X)

    else:
        if sparse.issparse(X):
            X = _sparse_to_dense(X, order="F")
        else:
            X = cp.asfortranarray(X)
        n_batches = math.ceil(X.shape[1] / batchsize)
        for batch in range(n_batches):
            start_idx = batch * batchsize
            stop_idx = min(batch * batchsize + batchsize, X.shape[1])

            arr_batch = X[:, start_idx:stop_idx].copy()
            lr = LinearRegression(
                fit_intercept=False, output_type="cupy", algorithm="svd"
            )
            lr.fit(regressors, arr_batch, convert_dtype=True)
            X[:, start_idx:stop_idx] = arr_batch - lr.predict(regressors)

    return X


def _maybe_densify(X):
    """Phase 0: If X is a sparse dask array, densify all chunks upfront."""
    if sparse.issparse(X._meta):
        X = X.map_blocks(
            _sparse_to_dense, order="C", dtype=X.dtype, meta=_meta_dense(X.dtype)
        )
    return X


def _regress_out_categorical_dask(X, categorical_series):
    """Regress out a categorical variable from a dask array by subtracting group means."""
    X = _maybe_densify(X)

    codes_cp = cp.asarray(categorical_series.cat.codes.to_numpy(), dtype=cp.int32)
    n_cats = len(categorical_series.cat.categories)
    n_genes = X.shape[1]

    # Build a dask array of codes aligned with X's row chunks
    codes_dask = da.from_array(codes_cp, chunks=(X.chunks[0],))

    # Phase 1: Compute partial group sums per chunk (float64 matches non-dask path)
    def _partial_sums(X_part, codes_part):
        codes_part = codes_part.ravel()
        partial = cp.zeros((n_cats, n_genes), dtype=cp.float64)
        cp.add.at(partial, codes_part, X_part)
        return partial[None, ...]  # (1, n_cats, n_genes)

    n_blocks = len(X.chunks[0])
    partial_sums = da.map_blocks(
        _partial_sums,
        X,
        codes_dask[..., None],
        new_axis=(1,),
        chunks=((1,) * n_blocks, (n_cats,), (n_genes,)),
        dtype=cp.float64,
        meta=cp.array([]),
    )
    global_sums = partial_sums.sum(axis=0).compute()

    counts = cp.bincount(codes_cp, minlength=n_cats).reshape(-1, 1)
    means = (global_sums / counts).astype(X.dtype)

    # Phase 2: Subtract group means per chunk
    def _subtract_means(X_part, codes_part):
        codes_part = codes_part.ravel()
        return X_part - means[codes_part]

    return da.map_blocks(
        _subtract_means,
        X,
        codes_dask[..., None],
        dtype=X.dtype,
        meta=_meta_dense(X.dtype),
    )


def _regress_out_continuous_dask(X, adata, keys):
    """Regress out continuous variables from a dask array using linear regression."""
    X = _maybe_densify(X)

    dim_regressor = len(keys) + 1
    n_genes = X.shape[1]

    # Build full regressors array on GPU (use X.dtype, matching non-dask path)
    regressors = cp.ones((X.shape[0], dim_regressor), dtype=X.dtype)
    for i, key in enumerate(keys):
        regressors[:, i + 1] = cp.array(adata.obs[key], dtype=X.dtype).ravel()

    # Check Gram matrix is invertible
    gram = regressors.T @ regressors
    if cp.linalg.det(gram) == 0:
        raise ValueError(
            "The Gram matrix (R^T R) is singular. "
            "The regressor variables are linearly dependent."
        )

    inv_gram = cp.linalg.inv(gram)

    # Build a dask array of regressors aligned with X's row chunks
    regressors_dask = da.from_array(regressors, chunks=(X.chunks[0], (dim_regressor,)))

    # Phase 1: Compute partial R^T @ X per chunk
    def _partial_rtx(X_part, R_part):
        return (R_part.T @ X_part)[None, ...]  # (1, dim_regressor, n_genes)

    n_blocks = len(X.chunks[0])
    partial_rtx = da.map_blocks(
        _partial_rtx,
        X,
        regressors_dask,
        new_axis=(1,),
        chunks=((1,) * n_blocks, (dim_regressor,), (n_genes,)),
        dtype=X.dtype,
        meta=cp.array([]),
    )
    global_rtx = partial_rtx.sum(axis=0).compute()

    # Solve for coefficients: coeff = inv(R^T R) @ R^T X
    coeff = inv_gram @ global_rtx

    # Phase 2: Subtract R @ coeff per chunk
    def _subtract_fit(X_part, R_part):
        return X_part - R_part @ coeff

    return da.map_blocks(
        _subtract_fit,
        X,
        regressors_dask,
        dtype=X.dtype,
        meta=_meta_dense(X.dtype),
    )
