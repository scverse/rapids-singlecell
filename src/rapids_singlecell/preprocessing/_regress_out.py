from __future__ import annotations

import math
from collections.abc import Collection
from typing import TYPE_CHECKING, Literal, Union

import cupy as cp
import dask.array as da
import numpy as np
import pandas as pd
from cuml.linear_model import LinearRegression
from cupyx.scipy import sparse
from pandas.api.types import CategoricalDtype
from scanpy._utils import view_to_actual
from scverse_misc import Deprecation, deprecated_arg

from rapids_singlecell._compat import DaskArray, _meta_dense
from rapids_singlecell.get import _get_arr, _get_vec
from rapids_singlecell.get._out import (
    DEPR_INPLACE,
    DEPR_USE,
    _legacy_out,
    _use_args,
    _write_out,
    deprecated_copy,
)
from rapids_singlecell.get._utils import AdRef, _to_numpy_1d

from ._utils import _check_gpu_X, _sparse_to_dense

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData

    from rapids_singlecell.get._utils import LayerAcc, MultiAcc


@deprecated_arg("layer", DEPR_USE)
@deprecated_arg("keys", Deprecation("0.19.0", "Use `covariates` instead."))
@deprecated_arg("inplace", DEPR_INPLACE)
@deprecated_copy
def regress_out(
    adata: AnnData,
    covariates: AdRef | str | Sequence[AdRef | str] | None = None,
    *,
    use: LayerAcc | MultiAcc | str | None = None,
    out: LayerAcc | MultiAcc | str | bool = False,
    batchsize: int | Literal["all"] | None = None,
    verbose: bool = False,
    # deprecated
    layer: str | None = None,
    inplace: bool = True,
    copy: bool = False,
    keys: str | Sequence[str] | None = None,
) -> Union[AnnData, cp.ndarray, None]:  # noqa: UP007
    """Use linear regression to adjust for the effects of unwanted noise and variation.

    Parameters
    ----------
        adata
            AnnData object

        covariates
            Observation annotation(s) to regress on, e.g. `A.obs['n_counts']`.
            Strings name `.obs` columns (resolved by :meth:`anndata.acc.AdAcc.resolve` with the
            `scanpy-v2-preview` preset). Supports both numerical and single categorical covariates.

        use
            Which matrix to regress, e.g. `A.X` or `A.layers['counts']`.
            Strings are resolved by :meth:`anndata.acc.AdAcc.resolve`, e.g. `'layers.counts'`.

        out
            Where to write the result, e.g. `A.layers['k']` (or `'layers.k'`).
            If `True`, the result is returned instead of written.
            If `False`, it is written to the place `use` reads from.

        batchsize
            Number of genes that should be processed together. \
            If `'all'` all genes will be processed together if `.n_obs` <100000. \
            If `None` each gene will be analysed separately. \
            Only used for continuous covariates.

        verbose
            Print debugging information

    Returns
    -------
    The corrected matrix if `out=True`, else `None`.
    """
    if batchsize != "all" and type(batchsize) not in [int, type(None)]:
        raise ValueError("batchsize must be `int`, `None` or `'all'`")

    if keys is not None:
        if covariates is not None:
            raise TypeError("Pass either `covariates` or `keys`, not both.")
        covariates = keys
    if covariates is None:
        raise TypeError("regress_out() missing required argument: 'covariates'")
    out = _legacy_out(out, "inplace", None if inplace else True)
    src = _use_args(use, layer=layer)
    view_to_actual(adata)

    X = _get_arr(adata, dim="obs", **src)

    _check_gpu_X(X, allow_dask=True)

    if not sparse.issparse(X) and not isinstance(X, DaskArray) and out is not False:
        X = cp.array(X)

    if isinstance(covariates, AdRef | str) or not isinstance(covariates, Collection):
        covariates = [covariates]
    cov = pd.DataFrame({str(c): _to_numpy_1d(_get_vec(adata, c, dim="obs")) for c in covariates}, index=adata.obs_names)

    if any(isinstance(dtype, CategoricalDtype) for dtype in cov.dtypes):
        if cov.shape[1] > 1:
            raise ValueError("Only a single categorical variable is supported for regress_out.")
        if isinstance(X, DaskArray):
            X = _regress_out_categorical_dask(X, cov.iloc[:, 0])
        else:
            X = _regress_out_categorical(X, cov.iloc[:, 0])
    else:
        if isinstance(X, DaskArray):
            X = _regress_out_continuous_dask(X, cov)
        else:
            X = _regress_out_continuous(X, cov, batchsize)

    return _write_out(adata, X, out=out, dest=src)


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


def _prepare_regressors(cov, *, n_obs, dtype, order):
    """Build the design matrix ``[1, *keys]`` and its Gram matrix.

    Centring the columns and scaling them to unit length leaves the column
    space -- and so the result -- unchanged, but keeps ``cond(R.T @ R)`` at
    O(1) rather than the ~1e11 that covariates on very different scales give,
    which float32 cannot carry. The Gram matrix is accumulated in float64 too.
    """
    dim_regressor = cov.shape[1] + 1
    regressors = cp.ones((n_obs, dim_regressor), dtype=cp.float64, order=order)
    for i, (_, col) in enumerate(cov.items()):
        regressors[:, i + 1] = cp.array(col).ravel()
    regressors[:, 1:] -= regressors[:, 1:].mean(axis=0)
    norms = cp.linalg.norm(regressors, axis=0)
    regressors /= cp.where(norms > 0, norms, 1.0)  # a constant covariate stays 0

    # k gemv calls: cuBLAS is ~20x slower on the (k, n_obs) @ (n_obs, k) gemm
    cols = [regressors.T @ regressors[:, j] for j in range(dim_regressor)]
    gram = cp.stack(cols, axis=1)
    return regressors.astype(dtype, order=order, copy=False), gram


def _regress_out_continuous(X, cov, batchsize):
    """Regress out continuous variables using linear regression."""
    regressors, gram = _prepare_regressors(cov, n_obs=X.shape[0], dtype=X.dtype, order="F")

    # Set default batch size based on the number of samples in X
    DEFAULT_GENE_BATCH = 100
    BATCH_CELL_THRESHOLD = 100_000
    if batchsize is None:
        batchsize = DEFAULT_GENE_BATCH if X.shape[0] > BATCH_CELL_THRESHOLD else "all"

    # Validate the choice of "all" batch size
    if batchsize == "all":
        if np.linalg.det(cp.asnumpy(gram)) == 0:
            batchsize = DEFAULT_GENE_BATCH

    # Do regression
    if batchsize == "all":
        if sparse.issparse(X):
            X = _sparse_to_dense(X, order="C")
        else:
            X = cp.ascontiguousarray(X)
        # gram is float64, so the solve is too: it is what keeps near-collinear
        # covariates from losing the coefficients to float32 rounding
        coeff = cp.linalg.solve(gram, regressors.T @ X).astype(X.dtype, copy=False)
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
            lr = LinearRegression(fit_intercept=False, output_type="cupy", algorithm="svd")
            lr.fit(regressors, arr_batch)
            X[:, start_idx:stop_idx] = arr_batch - lr.predict(regressors)

    return X


def _maybe_densify(X):
    """Phase 0: If X is a sparse dask array, densify all chunks upfront."""
    if sparse.issparse(X._meta):
        X = X.map_blocks(_sparse_to_dense, order="C", dtype=X.dtype, meta=_meta_dense(X.dtype))
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


def _regress_out_continuous_dask(X, cov):
    """Regress out continuous variables from a dask array using linear regression."""
    X = _maybe_densify(X)

    dim_regressor = cov.shape[1] + 1
    n_genes = X.shape[1]

    # Build full regressors array on GPU (matching the non-dask path)
    regressors, gram = _prepare_regressors(cov, n_obs=X.shape[0], dtype=X.dtype, order="C")

    # Check Gram matrix is invertible
    if np.linalg.det(cp.asnumpy(gram)) == 0:
        raise ValueError("The Gram matrix (R^T R) is singular. The regressor variables are linearly dependent.")

    # Build a dask array of regressors aligned with X's row chunks
    regressors_dask = da.from_array(regressors, chunks=(X.chunks[0], (dim_regressor,)))

    # Phase 1: Compute partial R^T @ X per chunk, widened so that summing the
    # blocks does not make the result depend on how X is chunked
    def _partial_rtx(X_part, R_part):
        return (R_part.T @ X_part).astype(cp.float64)[None, ...]

    n_blocks = len(X.chunks[0])
    partial_rtx = da.map_blocks(
        _partial_rtx,
        X,
        regressors_dask,
        new_axis=(1,),
        chunks=((1,) * n_blocks, (dim_regressor,), (n_genes,)),
        dtype=cp.float64,
        meta=cp.array([]),
    )
    global_rtx = partial_rtx.sum(axis=0).compute()

    # Solve the normal equations for the coefficients
    coeff = cp.linalg.solve(gram, global_rtx).astype(X.dtype, copy=False)

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
