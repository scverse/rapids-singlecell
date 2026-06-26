from __future__ import annotations

import math
import warnings
from functools import partial
from typing import TYPE_CHECKING, Union

import cupy as cp
from cupyx.scipy import sparse
from cupyx.scipy.sparse import csr_matrix
from scanpy.get import _get_obs_rep, _set_obs_rep

from rapids_singlecell._compat import (
    DaskArray,
    _meta_dense,
    _meta_sparse,
)

from ._utils import _check_gpu_X, _check_nonnegative_integers

if TYPE_CHECKING:
    from anndata import AnnData
    from cupyx.scipy.sparse import spmatrix

    from rapids_singlecell._utils import ArrayTypesDask


def normalize_total(
    adata: AnnData,
    *,
    target_sum: float | None = None,
    exclude_highly_expressed: bool = False,
    max_fraction: float = 0.05,
    layer: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> Union[AnnData, csr_matrix, cp.ndarray, None]:  # noqa: UP007
    """\
    Normalizes rows in matrix so they sum to `target_sum`.

    Parameters
    ----------
        adata
            AnnData object

        target_sum
            If `None`, after normalization, each observation (cell) has a total count
            equal to the median of total counts for observations (cells) before normalization.

        exclude_highly_expressed
            Exclude (very) highly expressed genes for the computation of the
            normalization factor (size factor) for each cell. A gene is considered
            highly expressed, if it has more than `max_fraction` of the total counts
            in at least one cell. The not-excluded genes will sum up to
            `target_sum`.

        max_fraction
            If `exclude_highly_expressed=True`, consider cells as highly expressed
            that have more counts than `max_fraction` of the original total counts
            in at least one cell.

        layer
            Layer to normalize instead of `X`. If `None`, `X` is normalized.

        inplace
            Whether to update `adata` or return the matrix.

        copy
            Whether to return a copy or update `adata`. Not compatible with inplace=False.
    Returns
    -------
        Returns a normalized copy or  updates `adata` with a normalized version of
        the original `adata.X` and `adata.layers['layer']`, depending on `inplace`.
    """
    if copy:
        if not inplace:
            raise ValueError("`copy=True` cannot be used with `inplace=False`.")
        adata = adata.copy()
    X = _get_obs_rep(adata, layer=layer)

    _check_gpu_X(X, allow_dask=True)

    if not inplace:
        X = X.copy()

    if sparse.isspmatrix_csc(X):
        X = X.tocsr()

    if exclude_highly_expressed:
        if isinstance(X, DaskArray):
            raise NotImplementedError(
                "`exclude_highly_expressed` is not supported for Dask arrays."
            )
        if not 0 < max_fraction < 1:
            raise ValueError(
                f"`max_fraction` must be between 0 and 1, got {max_fraction}."
            )

    if isinstance(X, DaskArray):
        X = _normalize_total_dask(X, target_sum)
    elif isinstance(X, sparse.csr_matrix):
        X = _normalize_total_csr(
            X,
            target_sum,
            exclude_highly_expressed=exclude_highly_expressed,
            max_fraction=max_fraction,
        )
    elif isinstance(X, cp.ndarray):
        X = _normalize_total_dense(
            X,
            target_sum,
            exclude_highly_expressed=exclude_highly_expressed,
            max_fraction=max_fraction,
        )
    else:
        raise ValueError(f"Cannot normalize {type(X)}")

    if inplace:
        _set_obs_rep(adata, X, layer=layer)

    if copy:
        return adata
    elif not inplace:
        return X


def _counts_to_scales(
    counts_per_cell: cp.ndarray, target_sum: float | None = None
) -> cp.ndarray:
    """Compute per-cell scale factors. Uses median of nonzero counts if target_sum is None."""
    nonzero = counts_per_cell > 0
    if target_sum is None:
        target_sum = cp.median(counts_per_cell[nonzero])
    scales = cp.zeros_like(counts_per_cell)
    scales[nonzero] = (
        cp.array(target_sum, dtype=counts_per_cell.dtype) / counts_per_cell[nonzero]
    )
    return scales


def _normalize_total_csr(
    X: sparse.csr_matrix,
    target_sum: float | None,
    *,
    exclude_highly_expressed: bool,
    max_fraction: float,
) -> sparse.csr_matrix:
    n_cells, n_genes = X.shape
    gene_is_hi = None

    if exclude_highly_expressed:
        from rapids_singlecell._cuda import _norm_cuda as _nc

        gene_is_hi = cp.zeros(n_genes, dtype=cp.bool_)
        _nc.find_hi_genes_csr(
            X.indptr,
            X.indices,
            X.data,
            gene_is_hi=gene_is_hi,
            max_fraction=float(max_fraction),
            nrows=n_cells,
            stream=cp.cuda.get_current_stream().ptr,
        )

    if target_sum is not None and gene_is_hi is None:
        # Fused: row sum + scale in one pass
        from rapids_singlecell._cuda import _norm_cuda as _nc

        _nc.mul_csr(
            X.indptr,
            X.data,
            nrows=n_cells,
            target_sum=float(target_sum),
            stream=cp.cuda.get_current_stream().ptr,
        )
    elif target_sum is not None:
        # Fused: masked row sum + scale in one pass
        from rapids_singlecell._cuda import _norm_cuda as _nc

        _nc.masked_mul_csr(
            X.indptr,
            X.indices,
            X.data,
            gene_mask=gene_is_hi,
            nrows=n_cells,
            tsum=float(target_sum),
            stream=cp.cuda.get_current_stream().ptr,
        )
    else:
        # Two-pass: compute counts → median → prescaled multiply
        from rapids_singlecell._cuda import _norm_cuda as _nc

        if gene_is_hi is None:
            counts = cp.zeros(n_cells, dtype=X.dtype)
            _nc.sum_major(
                X.indptr,
                X.data,
                sums=counts,
                major=n_cells,
                stream=cp.cuda.get_current_stream().ptr,
            )
        else:
            counts = cp.zeros(n_cells, dtype=X.dtype)
            _nc.masked_sum_major(
                X.indptr,
                X.indices,
                X.data,
                gene_mask=gene_is_hi,
                sums=counts,
                major=n_cells,
                stream=cp.cuda.get_current_stream().ptr,
            )

        scales = _counts_to_scales(counts)
        _nc.prescaled_mul_csr(
            X.indptr,
            X.data,
            scales=scales,
            nrows=n_cells,
            stream=cp.cuda.get_current_stream().ptr,
        )

    return X


def _normalize_total_dense(
    X: cp.ndarray,
    target_sum: float | None,
    *,
    exclude_highly_expressed: bool,
    max_fraction: float,
) -> cp.ndarray:
    if not X.flags.c_contiguous:
        X = cp.asarray(X, order="C")

    n_cells, n_cols = X.shape

    if target_sum is not None and not exclude_highly_expressed:
        # Fused: row sum + scale in one pass
        from rapids_singlecell._cuda import _norm_cuda as _nc

        _nc.mul_dense(
            X,
            nrows=n_cells,
            ncols=n_cols,
            target_sum=float(target_sum),
            stream=cp.cuda.get_current_stream().ptr,
        )
    else:
        # Compute per-cell counts, then prescaled multiply
        from rapids_singlecell._cuda import _norm_cuda as _nc

        counts_per_cell = X.sum(axis=1)
        if exclude_highly_expressed:
            hi_exp = X > max_fraction * counts_per_cell.reshape(-1, 1)
            gene_subset = ~hi_exp.any(axis=0)
            counts_per_cell = X[:, gene_subset].sum(axis=1)

        scales = _counts_to_scales(counts_per_cell, target_sum)
        _nc.prescaled_mul_dense(
            X,
            scales=scales,
            nrows=n_cells,
            ncols=n_cols,
            stream=cp.cuda.get_current_stream().ptr,
        )

    return X


def _normalize_total_dask(X: DaskArray, target_sum: float | None) -> DaskArray:
    if target_sum is None:
        target_sum = _get_target_sum_dask(X)

    if isinstance(X._meta, sparse.csr_matrix):
        from rapids_singlecell._cuda import _norm_cuda as _nc

        def __mul(X_part):
            _nc.mul_csr(
                X_part.indptr,
                X_part.data,
                nrows=X_part.shape[0],
                target_sum=float(target_sum),
                stream=cp.cuda.get_current_stream().ptr,
            )
            return X_part

        X = X.map_blocks(__mul, meta=_meta_sparse(X.dtype))
    elif isinstance(X._meta, cp.ndarray):
        from rapids_singlecell._cuda import _norm_cuda as _nc

        def __mul(X_part):
            _nc.mul_dense(
                X_part,
                nrows=X_part.shape[0],
                ncols=X_part.shape[1],
                target_sum=float(target_sum),
                stream=cp.cuda.get_current_stream().ptr,
            )
            return X_part

        X = X.map_blocks(__mul, meta=_meta_dense(X.dtype))
    else:
        raise ValueError(f"Cannot normalize {type(X)}")
    return X


def _get_target_sum_dask(X: DaskArray) -> int:
    if isinstance(X._meta, sparse.csr_matrix):
        from rapids_singlecell._cuda import _norm_cuda as _nc

        def __sum(X_part):
            counts_per_cell = cp.zeros(X_part.shape[0], dtype=X_part.dtype)
            _nc.sum_major(
                X_part.indptr,
                X_part.data,
                sums=counts_per_cell,
                major=X_part.shape[0],
                stream=cp.cuda.get_current_stream().ptr,
            )
            return counts_per_cell

    elif isinstance(X._meta, cp.ndarray):

        def __sum(X_part):
            return X_part.sum(axis=1)
    else:
        raise ValueError(f"Cannot compute target sum for {type(X)}")
    target_sum_chunk_matrices = X.map_blocks(
        __sum,
        meta=cp.array((1.0,), dtype=X.dtype),
        dtype=X.dtype,
        chunks=(X.chunksize[0],),
        drop_axis=1,
    )
    counts_per_cell = target_sum_chunk_matrices.compute()
    counts_per_cell = counts_per_cell[counts_per_cell > 0]
    target_sum = cp.median(counts_per_cell)
    return target_sum


def _calc_log1p(X: ArrayTypesDask, base: float | None = None) -> ArrayTypesDask:
    if isinstance(X, DaskArray):
        meta = _meta_sparse if isinstance(X._meta, csr_matrix) else _meta_dense
        X = X.map_blocks(partial(_calc_log1p, base=base), meta=meta(X.dtype))
    else:
        X = X.copy()
        if sparse.issparse(X):
            X = X.log1p()
            if base is not None:
                X.data /= cp.log(base)
        else:
            X = cp.log1p(X)
            if base is not None:
                X /= cp.log(base)
    return X


def log1p(
    adata: AnnData,
    *,
    base: float | None = None,
    layer: str | None = None,
    obsm: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> Union[AnnData, spmatrix, cp.ndarray, None]:  # noqa: UP007
    """\
    Logarithmize the data matrix.

    Computes :math:`X = \\log(X + 1)`, where :math:`log` denotes the natural logarithm
    unless a different `base` is given.

    Parameters
    ----------
        adata
            AnnData object
        base
            Base of the logarithm. Natural logarithm is used by default.
        layer
            Layer to normalize instead of `X`. If `None`, `X` is normalized.
        obsm
            Entry of `.obsm` to transform.
        inplace
            Whether to update `adata` or return the matrix.
        copy
            Whether to return a copy or update `adata`. Not compatible with `inplace=False`.

    Returns
    -------
    The resulting matrix after applying the logarithm of one plus the input matrix. \
    If `copy` is set to True, returns the modified AnnData. Otherwise, updates the `adata` object \
    in-place and returns None.

    """
    if copy:
        if not inplace:
            raise ValueError("`copy=True` cannot be used with `inplace=False`.")
        adata = adata.copy()
    X = _get_obs_rep(adata, layer=layer, obsm=obsm)

    _check_gpu_X(X, allow_dask=True)

    if not inplace:
        X = X.copy()

    X = _calc_log1p(X, base=base)
    adata.uns["log1p"] = {"base": base}
    if inplace:
        _set_obs_rep(adata, X, layer=layer, obsm=obsm)

    if copy:
        return adata
    elif not inplace:
        return X


def normalize_pearson_residuals(
    adata: AnnData,
    *,
    theta: float = 100,
    clip: float | None = None,
    check_values: bool = True,
    layer: str | None = None,
    inplace: bool = True,
) -> Union[cp.ndarray, None]:  # noqa: UP007
    """\
    Applies analytic Pearson residual normalization :cite:p:`Lause2021`.
    The residuals are based on a negative binomial offset model with overdispersion
    `theta` shared across genes. By default, residuals are clipped to `sqrt(n_obs)`
    and overdispersion `theta=100` is used.

    Parameters
    ----------
        adata
            AnnData object
        theta
            The negative binomial overdispersion parameter theta for Pearson residuals.
            Higher values correspond to less overdispersion `(var = mean + mean^2/theta)`, and `theta=np.Inf` corresponds to a Poisson model.
        clip
            Determines if and how residuals are clipped:
            If None, residuals are clipped to the interval [-sqrt(n_obs), sqrt(n_obs)], where n_obs is the number of cells in the dataset (default behavior).
            If any scalar c, residuals are clipped to the interval `[-c, c]`. Set `clip=np.Inf` for no clipping.
        check_values
            If True, checks if counts in selected layer are integers as expected by this function,
            and return a warning if non-integers are found. Otherwise, proceed without checking. Setting this to False can speed up code for large datasets.
        layer
            Layer to use as input instead of :attr:`~anndata.AnnData.X`. If None, :attr:`~anndata.AnnData.X` is used.
        inplace
            If True, update AnnData with results. Otherwise, return results. See below for details of what is returned.

    Returns
    -------
        If `inplace=True`, :attr:`~anndata.AnnData.X` or the selected layer in :attr:`~anndata.AnnData.layers` is updated with the normalized values. \
        If `inplace=False` the normalized matrix is returned.

    """
    X = _get_obs_rep(adata, layer=layer)

    _check_gpu_X(X, require_cf=True)

    if check_values and not _check_nonnegative_integers(X):
        warnings.warn(
            "`flavor='pearson_residuals'` expects raw count data, but non-integers were found.",
            UserWarning,
        )
    computed_on = layer if layer else "adata.X"
    settings_dict = {"theta": theta, "clip": clip, "computed_on": computed_on}
    if theta <= 0:
        raise ValueError("Pearson residuals require theta > 0")
    if clip is None:
        clip = math.sqrt(X.shape[0])
    if clip < 0:
        raise ValueError("Pearson residuals require `clip>=0` or `clip=None`.")

    from rapids_singlecell._cuda import _pr_cuda as _pr

    inv_theta = 1.0 / theta
    n_cells, n_genes = X.shape
    stream = cp.cuda.get_current_stream().ptr

    if sparse.issparse(X):
        residuals = cp.zeros(X.shape, dtype=X.dtype)
        if sparse.isspmatrix_csc(X):
            sums_genes = cp.zeros(n_genes, dtype=X.dtype)
            sums_cells = cp.zeros(n_cells, dtype=X.dtype)
            _pr.sparse_sum_csc(
                X.indptr,
                X.indices,
                X.data,
                sums_genes=sums_genes,
                sums_cells=sums_cells,
                n_genes=n_genes,
                stream=stream,
            )
            inv_sum_total = float(1.0 / sums_genes.sum())
            _pr.sparse_norm_res_csc(
                X.indptr,
                X.indices,
                X.data,
                sums_cells=sums_cells,
                sums_genes=sums_genes,
                residuals=residuals,
                inv_sum_total=inv_sum_total,
                clip=float(clip),
                inv_theta=inv_theta,
                n_cells=n_cells,
                n_genes=n_genes,
                stream=stream,
            )
        elif sparse.isspmatrix_csr(X):
            sums_cells = cp.array(X.sum(axis=1), dtype=X.dtype).ravel()
            sums_genes = cp.array(X.sum(axis=0), dtype=X.dtype).ravel()
            inv_sum_total = float(1.0 / sums_genes.sum())
            _pr.sparse_norm_res_csr(
                X.indptr,
                X.indices,
                X.data,
                sums_cells=sums_cells,
                sums_genes=sums_genes,
                residuals=residuals,
                inv_sum_total=inv_sum_total,
                clip=float(clip),
                inv_theta=inv_theta,
                n_cells=n_cells,
                n_genes=n_genes,
                stream=stream,
            )
        else:
            raise ValueError(
                "Please transform you sparse matrix into CSR or CSC format."
            )
    else:
        residuals = cp.zeros(X.shape, dtype=X.dtype)
        sums_cells = X.sum(axis=1).astype(X.dtype)
        sums_genes = X.sum(axis=0).astype(X.dtype)
        inv_sum_total = float(1.0 / sums_genes.sum())
        _pr.dense_norm_res(
            X,
            residuals=residuals,
            sums_cells=sums_cells,
            sums_genes=sums_genes,
            inv_sum_total=inv_sum_total,
            clip=float(clip),
            inv_theta=inv_theta,
            n_cells=n_cells,
            n_genes=n_genes,
            stream=stream,
        )

    if inplace is True:
        adata.uns["pearson_residuals_normalization"] = settings_dict
        _set_obs_rep(adata, residuals, layer=layer)
    else:
        return residuals
