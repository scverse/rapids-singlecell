from __future__ import annotations

import math
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict, Union

import cupy as cp
from anndata import AnnData
from cupyx.scipy import sparse
from cupyx.scipy.sparse import csr_matrix
from scanpy._utils import view_to_actual
from scverse_misc import deprecated_arg

from rapids_singlecell._compat import (
    DaskArray,
    _meta_dense,
    _meta_sparse,
)
from rapids_singlecell.get import _get_arr
from rapids_singlecell.get._out import (
    DEPR_INPLACE,
    DEPR_KEY_ADDED,
    DEPR_USE,
    _dest,
    _key,
    _legacy_out,
    _out_dests,
    _use_args,
    _write_dests,
    _write_out,
    deprecated_copy,
)

from ._utils import _check_gpu_X, _check_nonnegative_integers, _get_mean_var

if TYPE_CHECKING:
    from cupyx.scipy.sparse import spmatrix

    from rapids_singlecell._utils import ArrayTypesDask
    from rapids_singlecell.get._utils import AdRef, LayerAcc, MultiAcc


class NormalizeTotalResult(NamedTuple):
    """Outputs of :func:`~rapids_singlecell.pp.normalize_total`."""

    normalized: Any
    """The normalized matrix."""
    norm_factor: Any
    """Counts per cell before normalization divided by `target_sum`."""


class NormalizeTotalOut(TypedDict, total=False):
    """Where a mapping `out` of :func:`~rapids_singlecell.pp.normalize_total` writes each output."""

    normalized: LayerAcc | MultiAcc | str
    norm_factor: AdRef | str


@deprecated_arg("layer", DEPR_USE)
@deprecated_arg("inplace", DEPR_INPLACE)
@deprecated_arg("key_added", DEPR_KEY_ADDED)
@deprecated_copy
def normalize_total(
    adata: AnnData,
    *,
    target_sum: float | None = None,
    exclude_highly_expressed: bool = False,
    max_fraction: float = 0.05,
    use: LayerAcc | MultiAcc | str | None = None,
    out: LayerAcc | MultiAcc | str | NormalizeTotalOut | bool = False,
    # deprecated
    key_added: str | None = None,
    layer: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> Union[AnnData, NormalizeTotalResult, None]:  # noqa: UP007
    """Normalizes rows in matrix so they sum to `target_sum`.

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

        use
            Which matrix to normalize, e.g. `A.X`, `A.layers['counts']` or `A.obsm['x']`.
            Strings are resolved by :meth:`anndata.acc.AdAcc.resolve`, e.g. `'layers.counts'`.

        out
            Where to write the outputs. A single accessor writes `normalized` to it,
            a mapping (:class:`~rapids_singlecell.pp.NormalizeTotalOut`) writes only the outputs it names,
            e.g. `{'normalized': A.layers['norm'], 'norm_factor': A.obs['nf']}`.
            If `True`, they are returned as a :class:`~rapids_singlecell.pp.NormalizeTotalResult`.
            If `False`, `normalized` is written to the place `use` reads from and `norm_factor` isn't written.

    Returns
    -------
        `None`, or a :class:`~rapids_singlecell.pp.NormalizeTotalResult` if `out=True`.
    """
    out = _legacy_out(out, "inplace", None if inplace else True)
    src = _use_args(use, layer=layer)
    dests = _out_dests(
        out,
        key_added,
        dests=lambda key: NormalizeTotalResult(src, None if key is None else {"obs": key}),
        kinds=NormalizeTotalResult({"layer", "obsm"}, {"obs"}),
    )
    view_to_actual(adata)

    X = _get_arr(adata, dim="obs", **src)

    _check_gpu_X(X, allow_dask=True)

    if X.dtype.kind in "iu":
        X = X.astype(cp.float32)
    elif dests is True or dests[1].normalized != src:
        X = X.copy()
    need_factor = dests is True or dests[1].norm_factor is not None
    if need_factor:
        before = X.sum(axis=1).ravel()

    if sparse.isspmatrix_csc(X):
        X = X.tocsr()

    if exclude_highly_expressed:
        if isinstance(X, DaskArray):
            raise NotImplementedError("`exclude_highly_expressed` is not supported for Dask arrays.")
        if not 0 < max_fraction < 1:
            raise ValueError(f"`max_fraction` must be between 0 and 1, got {max_fraction}.")

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

    norm_factor = None
    if need_factor:  # each row was scaled uniformly, so the ratio of row sums is the factor
        after = X.sum(axis=1).ravel()
        norm_factor = before / (after + (after == 0)) * (after != 0)
    vals = NormalizeTotalResult(X, norm_factor)
    if dests is True:
        return vals if inplace else {"X": X, "norm_factor": norm_factor}
    _write_dests(adata, vals, dests[1])
    return None


def _counts_to_scales(counts_per_cell: cp.ndarray, target_sum: float | None = None) -> cp.ndarray:
    """Compute per-cell scale factors. Uses median of nonzero counts if target_sum is None."""
    nonzero = counts_per_cell > 0
    if target_sum is None:
        target_sum = cp.median(counts_per_cell[nonzero])
    scales = cp.zeros_like(counts_per_cell)
    scales[nonzero] = cp.array(target_sum, dtype=counts_per_cell.dtype) / counts_per_cell[nonzero]
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


@deprecated_arg("layer", DEPR_USE)
@deprecated_arg("obsm", DEPR_USE)
@deprecated_arg("inplace", DEPR_INPLACE)
@deprecated_copy
def log1p(
    data: AnnData | ArrayTypesDask,
    *,
    base: float | None = None,
    use: LayerAcc | MultiAcc | str | None = None,
    out: LayerAcc | MultiAcc | str | bool = False,
    # deprecated
    layer: str | None = None,
    obsm: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> Union[AnnData, spmatrix, cp.ndarray, None]:  # noqa: UP007
    r"""Logarithmize the data matrix.

    Computes :math:`X = \log(X + 1)`, where :math:`log` denotes the natural logarithm
    unless a different `base` is given.

    Parameters
    ----------
        data
            The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond
            to cells and columns to genes. If a matrix is passed instead of an
            :class:`~anndata.AnnData` object, the transformed matrix is returned.
        base
            Base of the logarithm. Natural logarithm is used by default.
        use
            Which matrix to transform, e.g. `A.X`, `A.layers['counts']` or `A.obsm['x']`.
            Strings are resolved by :meth:`anndata.acc.AdAcc.resolve`, e.g. `'layers.counts'`.
        out
            Where to write the result, e.g. `A.layers['k']` (or `'layers.k'`).
            If `True`, the result is returned instead of written.
            If `False`, it is written to the place `use` reads from.

    Returns
    -------
    The transformed matrix if a matrix is passed or `out=True`, else `None`.

    """
    if not isinstance(data, AnnData):
        if layer is not None or obsm is not None or use is not None or out is not False:
            raise ValueError("`use` and `out` can only be used with an AnnData object.")
        X = data
        _check_gpu_X(X, allow_dask=True)
        return _calc_log1p(X, base=base)

    adata = data
    out = _legacy_out(out, "inplace", None if inplace else True)
    src = _use_args(use, layer=layer, obsm=obsm)
    X = _get_arr(adata, dim="obs", **src)

    _check_gpu_X(X, allow_dask=True)

    X = _calc_log1p(X, base=base)
    if out is True:
        return X
    _write_out(adata, X, out=out, dest=src)
    if (src if out is False else _dest(out)) == {}:
        adata.uns["log1p"] = {"base": base}
    return None


def _calc_sqrt(X: ArrayTypesDask) -> ArrayTypesDask:
    if isinstance(X, DaskArray):
        meta = _meta_sparse if isinstance(X._meta, csr_matrix) else _meta_dense
        X = X.map_blocks(_calc_sqrt, meta=meta(X.dtype))
    else:
        X = X.copy()
        if sparse.issparse(X):
            X = X.sqrt()
        else:
            X = cp.sqrt(X)
    return X


@deprecated_arg("layer", DEPR_USE)
@deprecated_arg("obsm", DEPR_USE)
@deprecated_arg("inplace", DEPR_INPLACE)
@deprecated_copy
def sqrt(
    data: AnnData | ArrayTypesDask,
    *,
    use: LayerAcc | MultiAcc | str | None = None,
    out: LayerAcc | MultiAcc | str | bool = False,
    # deprecated
    layer: str | None = None,
    obsm: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> Union[AnnData, spmatrix, cp.ndarray, None]:  # noqa: UP007
    r"""Take the square root of the data matrix.

    Computes :math:`X = \sqrt{X}`.

    Parameters
    ----------
        data
            The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond
            to cells and columns to genes. If a matrix is passed instead of an
            :class:`~anndata.AnnData` object, the transformed matrix is returned.
        use
            Which matrix to transform, e.g. `A.X`, `A.layers['counts']` or `A.obsm['x']`.
            Strings are resolved by :meth:`anndata.acc.AdAcc.resolve`, e.g. `'layers.counts'`.
        out
            Where to write the result, e.g. `A.layers['k']` (or `'layers.k'`).
            If `True`, the result is returned instead of written.
            If `False`, it is written to the place `use` reads from.

    Returns
    -------
    The transformed matrix if a matrix is passed or `out=True`, else `None`.

    """
    if not isinstance(data, AnnData):
        if layer is not None or obsm is not None or use is not None or out is not False:
            raise ValueError("`use` and `out` can only be used with an AnnData object.")
        X = data
        _check_gpu_X(X, allow_dask=True)
        return _calc_sqrt(X)

    adata = data
    out = _legacy_out(out, "inplace", None if inplace else True)
    src = _use_args(use, layer=layer, obsm=obsm)
    X = _get_arr(adata, dim="obs", **src)

    _check_gpu_X(X, allow_dask=True)

    return _write_out(adata, _calc_sqrt(X), out=out, dest=src)


@deprecated_arg("layer", DEPR_USE)
@deprecated_arg("inplace", DEPR_INPLACE)
@deprecated_copy
def normalize_pearson_residuals(
    adata: AnnData,
    *,
    theta: float = 100,
    clip: float | None = None,
    check_values: bool = True,
    use: LayerAcc | MultiAcc | str | None = None,
    out: LayerAcc | MultiAcc | str | bool = False,
    # deprecated
    layer: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> Union[AnnData, cp.ndarray, None]:  # noqa: UP007
    """Applies analytic Pearson residual normalization :cite:p:`Lause2021`.

    The residuals are based on a negative binomial offset model with overdispersion
    `theta` shared across genes. By default, residuals are clipped to `sqrt(n_obs)`
    and overdispersion `theta=100` is used.

    Parameters
    ----------
        adata
            AnnData object
        theta
            The negative binomial overdispersion parameter theta for Pearson residuals.
            Higher values correspond to less overdispersion `(var = mean + mean^2/theta)`, and `theta=np.inf` corresponds to a Poisson model.
        clip
            Determines if and how residuals are clipped:
            If None, residuals are clipped to the interval [-sqrt(n_obs), sqrt(n_obs)], where n_obs is the number of cells in the dataset (default behavior).
            If any scalar c, residuals are clipped to the interval `[-c, c]`. Set `clip=np.inf` for no clipping.
        check_values
            If True, checks if counts in selected layer are integers as expected by this function,
            and return a warning if non-integers are found. Otherwise, proceed without checking. Setting this to False can speed up code for large datasets.
        use
            Which matrix to normalize, e.g. `A.X`, `A.layers['counts']` or `A.obsm['x']`.
            Strings are resolved by :meth:`anndata.acc.AdAcc.resolve`, e.g. `'layers.counts'`.
        out
            Where to write the result, e.g. `A.layers['k']` (or `'layers.k'`).
            If `True`, the result is returned instead of written.
            If `False`, it is written to the place `use` reads from.

    Returns
    -------
        The normalized matrix if `out=True`. Otherwise writes it to `out` and the settings to
        `.uns['pearson_residuals_normalization']` (or `.uns[k]` for `out=A.layers[k]`).

    """
    out = _legacy_out(out, "inplace", None if inplace else True)
    src = _use_args(use, layer=layer)
    X = _get_arr(adata, dim="obs", **src)

    _check_gpu_X(X, require_cf=True)

    if check_values and not _check_nonnegative_integers(X):
        warnings.warn(
            "`flavor='pearson_residuals'` expects raw count data, but non-integers were found.",
            UserWarning,
            stacklevel=2,
        )
    computed_on = layer or (repr(use) if use is not None else "adata.X")
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
            raise ValueError("Please transform you sparse matrix into CSR or CSC format.")
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

    if out is True:
        return residuals
    _write_out(adata, residuals, out=out, dest=src)
    adata.uns[(out is not False and _key(_dest(out))) or "pearson_residuals_normalization"] = settings_dict
    return None


def _estimate_nb_overdispersion(X: ArrayTypesDask) -> float:
    r"""Closed-form least squares of :math:`\mathrm{Var}_g = μ_g + α μ_g^2` across genes."""
    mu, var = _get_mean_var(X, axis=0, correction=0)
    mu2 = mu**2
    numerator = cp.sum((var - mu) * mu2)
    denominator = cp.sum(mu2 * mu2)
    if isinstance(X, DaskArray):
        import dask

        numerator, denominator = dask.compute(numerator, denominator)
    if float(denominator) == 0.0:
        msg = "Cannot estimate overdispersion: every gene has zero mean. Pass a positive `alpha` explicitly."
        raise ValueError(msg)
    alpha = float(numerator / denominator)
    if not alpha > 0:
        msg = f"Estimated overdispersion is non-positive (alpha = {alpha}); pass a positive `alpha` explicitly."
        raise ValueError(msg)
    return alpha


def _shifted_clr(X: cp.ndarray | sparse.csr_matrix, scale: float) -> cp.ndarray:
    """Dense ``log1p(scale * X)`` centered on its per-cell mean."""
    from rapids_singlecell._cuda import _norm_cuda as _nc

    n_rows, n_cols = X.shape
    out = cp.empty((n_rows, n_cols), dtype=X.dtype)
    if n_rows == 0 or n_cols == 0:
        return out
    if isinstance(X, cp.ndarray):
        _nc.shifted_clr_dense(
            cp.ascontiguousarray(X),
            out=out,
            nrows=n_rows,
            ncols=n_cols,
            scale=float(scale),
            stream=cp.cuda.get_current_stream().ptr,
        )
    else:
        _nc.shifted_clr_csr(
            X.indptr,
            X.indices,
            X.data,
            out=out,
            nrows=n_rows,
            ncols=n_cols,
            scale=float(scale),
            stream=cp.cuda.get_current_stream().ptr,
        )
    return out


@deprecated_arg("layer", DEPR_USE)
@deprecated_arg("inplace", DEPR_INPLACE)
@deprecated_copy
def normalize_shifted_clr(
    adata: AnnData,
    *,
    alpha: float | None = None,
    use: LayerAcc | MultiAcc | str | None = None,
    out: LayerAcc | MultiAcc | str | bool = False,
    # deprecated
    layer: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> Union[AnnData, cp.ndarray, DaskArray, dict[str, cp.ndarray | DaskArray], None]:  # noqa: UP007
    r"""Normalize counts with the shifted centered log-ratio (PFlog) transform :cite:p:`Booeshaghi2022`.

    If `alpha` is not provided, it is estimated from the input matrix. PFlog is
    then computed as

    .. math::
        T(x)_i = \log(1 + 4 α x_i)
            - \frac{1}{D} \sum_{j=1}^D \log(1 + 4 α x_j),

    which is equivalent to centering :math:`\log(x_i + 1 / (4 α))` because the
    constant :math:`\log(4 α)` cancels during CLR centering.

    .. note::
        CLR centering generally maps zeros to non-zero values, so the resulting
        matrix is dense even when the input is sparse.

    Parameters
    ----------
        adata
            The annotated data matrix of shape `n_obs` × `n_vars`.
            Rows correspond to cells and columns to genes.
        alpha
            Negative-binomial overdispersion of the dataset (``var = μ + α·μ²``).
            If `None`, it is estimated from the input matrix. A positive numeric
            value uses that value directly. PFlog applies ``log1p(4 * alpha * x)``
            before CLR centering.
        use
            Which matrix to normalize, e.g. `A.X`, `A.layers['counts']` or `A.obsm['x']`.
            Strings are resolved by :meth:`anndata.acc.AdAcc.resolve`, e.g. `'layers.counts'`.
        out
            Where to write the result, e.g. `A.layers['k']` (or `'layers.k'`).
            If `True`, the result is returned instead of written.
            If `False`, it is written to the place `use` reads from.

    Returns
    -------
    The normalized matrix if `out=True`, else `None`.
    """
    src = _use_args(use, layer=layer)
    view_to_actual(adata)

    X = _get_arr(adata, dim="obs", **src)
    _check_gpu_X(X, allow_dask=True)
    if sparse.isspmatrix_csc(X):
        X = X.tocsr()
    if X.dtype.kind in "iub":
        X = X.astype(cp.float64)

    if alpha is None:
        alpha = _estimate_nb_overdispersion(X)
    elif not alpha > 0:
        msg = f"`alpha` must be positive to compute PFlog, got {alpha}. The data may be underdispersed."
        raise ValueError(msg)
    scale = 4.0 * float(alpha)

    if isinstance(X, DaskArray):
        if len(X.chunks[1]) > 1:
            raise ValueError("Dask arrays must be chunked along the first axis only.")
        X = X.map_blocks(_shifted_clr, scale, meta=_meta_dense(X.dtype), dtype=X.dtype)
    else:
        cell_depths = X.sum(axis=1)
        if not bool((cell_depths > 0).all()):
            warnings.warn("Some cells have zero counts", UserWarning, stacklevel=2)
        X = _shifted_clr(X, scale)

    if not inplace:  # deprecated, returns what it did before `out`
        return {"X": X}
    return _write_out(adata, X, out=out, dest=src)
