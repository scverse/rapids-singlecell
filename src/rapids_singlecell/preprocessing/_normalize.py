from __future__ import annotations

import math
import warnings
from functools import partial
from typing import TYPE_CHECKING, Literal, Union

import cupy as cp
from anndata import AnnData
from cupyx.scipy import sparse
from cupyx.scipy.sparse import csr_matrix
from scanpy.get import _get_obs_rep, _set_obs_rep

from rapids_singlecell._compat import (
    DaskArray,
    _meta_dense,
    _meta_sparse,
)

from ._utils import _check_gpu_X, _check_nonnegative_integers, _get_mean_var

if TYPE_CHECKING:
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


def _normalize_total(
    X: ArrayTypesDask,
    target_sum: float | None,
    *,
    exclude_highly_expressed: bool = False,
    max_fraction: float = 0.05,
) -> ArrayTypesDask:
    if isinstance(X, DaskArray):
        return _normalize_total_dask(X, target_sum)
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
    return X


def _sum_axis1(X: ArrayTypesDask) -> cp.ndarray | DaskArray:
    """Per-cell counts (sum over axis=1) for a CSR matrix, dense cupy array, or Dask array."""
    if isinstance(X, DaskArray):
        return X.map_blocks(
            _sum_axis1,
            meta=cp.array((1.0,), dtype=X.dtype),
            dtype=X.dtype,
            chunks=(X.chunksize[0],),
            drop_axis=1,
        )
    if isinstance(X, sparse.csr_matrix):
        from rapids_singlecell._cuda import _norm_cuda as _nc

        counts = cp.zeros(X.shape[0], dtype=X.dtype)
        _nc.sum_major(
            X.indptr,
            X.data,
            sums=counts,
            major=X.shape[0],
            stream=cp.cuda.get_current_stream().ptr,
        )
        return counts
    elif isinstance(X, cp.ndarray):
        return X.sum(axis=1)
    raise ValueError(f"Cannot compute row sums for {type(X)}")


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
            counts = _sum_axis1(X)
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

        counts_per_cell = _sum_axis1(X)
        if exclude_highly_expressed:
            hi_exp = X > max_fraction * counts_per_cell.reshape(-1, 1)
            gene_subset = ~hi_exp.any(axis=0)
            counts_per_cell = _sum_axis1(X[:, gene_subset])

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
    counts_per_cell = _sum_axis1(X).compute()
    counts_per_cell = counts_per_cell[counts_per_cell > 0]
    target_sum = cp.median(counts_per_cell)
    return target_sum


def normalize_clr(
    adata: AnnData,
    *,
    target_sum: float | None = None,
    alpha: float | Literal["auto"] | None = None,
    layer: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> Union[AnnData, spmatrix, cp.ndarray, None]:  # noqa: UP007
    r"""\
    Normalize counts with the shifted centered log-ratio (PFlog1pPF) transform.

    Computes the shifted centered log-ratio (CLR) transform

    .. math::
        T(x)_i = \log(u_i + 1) - \frac{1}{D} \sum_{j=1}^D \log(u_j + 1),

    where :math:`u_i = K \, x_i / \sum_j x_j` are the depth-normalized counts
    (proportional fitting to a target depth :math:`K`) and :math:`D` is the number
    of genes. Equivalently this is proportional fitting, then ``log1p``, then
    per-cell mean-centering in log space (the centered-log-ratio step). The
    transform is simultaneously variance-stabilizing, depth-invariant, and
    rank-preserving :cite:p:`Booeshaghi2026`.

    To avoid densifying the matrix, the centering term is *not* subtracted in
    place: ``adata.X`` (or `layer`) holds the sparse :math:`\log(u + 1)`, while the
    per-cell centering offset :math:`\frac{1}{D}\sum_j \log(u_j + 1)` is written to
    ``adata.obsm["clr_residuals"]`` and the raw per-cell depths to
    ``adata.obsm["clr_cell_depths"]``. The full centered CLR is recovered as
    ``adata.X - adata.obsm["clr_residuals"][:, None]``.

    .. note::
        When `adata.X` is a Dask array, deriving the proportional-fitting target
        :math:`K` from the data requires a global reduction and therefore triggers
        a blocking ``.compute()`` (for the default mean-depth target, and for
        `alpha`, including ``alpha="auto"``). Only the scalar reduction is
        materialized, not the matrix. Passing `target_sum` explicitly keeps it lazy.

    Parameters
    ----------
        adata
            The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond
            to cells and columns to genes.
        target_sum
            Target depth :math:`K` for the proportional-fitting step. If `None`
            (and `alpha` is not given), the empirical mean cell depth is used. This
            is only an *intermediate* target: the subsequent log and centering steps
            put each cell on the zero-sum hyperplane regardless of `K`.
        alpha
            Negative-binomial overdispersion of the dataset (``var = μ + α·μ²``).
            When given, it overrides `target_sum` and sets :math:`K = 4 \cdot α \cdot s`
            by the delta method, where :math:`s` is the mean cell depth, calibrating
            the count-scale pseudocount to the variance-stabilizing value
            ``y0 = 1/(4·α)`` :cite:p:`Booeshaghi2026`. Pass ``"auto"`` to estimate
            :math:`α` from the data (closed-form least squares of ``var = μ + α·μ²``
            across genes). Raises a :class:`ValueError` if the estimated or supplied
            :math:`α` is not positive (e.g. underdispersed data); pass `target_sum`
            instead.
        layer
            Layer to normalize instead of `X`. If `None`, `X` is normalized.
        inplace
            Whether to update `adata` or return the result.
        copy
            Whether to return a copy or update `adata`. Not compatible with
            `inplace=False`.

    Returns
    -------
        Depending on `inplace`:

        - `inplace=True` (default): updates `adata.X` (or `layer`) with the sparse
          :math:`\log(u + 1)`, writes ``adata.obsm["clr_cell_depths"]`` and
          ``adata.obsm["clr_residuals"]``, and returns `None`.
        - `copy=True`: performs the in-place update on a copy and returns it.
        - `inplace=False`: returns the tuple ``(X, cell_depths, residuals)`` and
          leaves `adata` untouched.
    """
    if copy:
        if not inplace:
            msg = "`copy=True` cannot be used with `inplace=False`."
            raise ValueError(msg)
        adata = adata.copy()
    X = _get_obs_rep(adata, layer=layer)
    _check_gpu_X(X, allow_dask=True)
    if not inplace:
        X = X.copy()
    if sparse.isspmatrix_csc(X):
        X = X.tocsr()
    X, cell_depths, residuals = _normalize_clr(X, target_sum=target_sum, alpha=alpha)
    if inplace:
        _set_obs_rep(adata, X, layer=layer)
        adata.obsm["clr_cell_depths"] = cell_depths
        adata.obsm["clr_residuals"] = residuals
    if copy:
        return adata
    if not inplace:
        return X, cell_depths, residuals
    return None


def _estimate_overdispersion(X: ArrayTypesDask) -> tuple[float, cp.ndarray]:
    mean, var = _get_mean_var(X, axis=0, correction=0)
    cell_depths = _sum_axis1(X)
    if isinstance(X, DaskArray):
        import dask

        mean, var, cell_depths = dask.compute(mean, var, cell_depths)
    mean_sq = mean**2
    numerator = cp.sum((var - mean) * mean_sq)
    denominator = cp.sum(mean_sq**2)
    if float(denominator) == 0:
        msg = "Cannot estimate overdispersion: all gene means are zero."
        raise ValueError(msg)
    return numerator / denominator, cell_depths


def _normalize_clr(
    X: ArrayTypesDask, target_sum: float | None, alpha: float | Literal["auto"] | None
) -> tuple[ArrayTypesDask, cp.ndarray, cp.ndarray]:
    if alpha == "auto":
        alpha, cell_depths = _estimate_overdispersion(X)
    else:
        cell_depths = _sum_axis1(X)
        if isinstance(cell_depths, DaskArray):
            cell_depths = cell_depths.compute()
    if bool((cell_depths == 0).any()):
        warnings.warn("Some cells have zero counts", UserWarning)
    if alpha is not None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        target_sum = 4.0 * alpha * float(cell_depths.mean())
    elif target_sum is None:
        target_sum = float(cell_depths.mean())

    X = _normalize_total(X, target_sum)
    X = _calc_log1p(X)
    # Centering offset = per-cell mean of log1p(PF) = row sum / n_genes.
    # `_sum_axis1` already covers CSR/dense/Dask; avoids the unused variance pass.
    residuals = _sum_axis1(X) / X.shape[1]

    return X, cell_depths, residuals


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
    data: AnnData | ArrayTypesDask,
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
        data
            The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond
            to cells and columns to genes. If a matrix is passed instead of an
            :class:`~anndata.AnnData` object, the transformed matrix is returned.
        base
            Base of the logarithm. Natural logarithm is used by default.
        layer
            Layer to normalize instead of `X`. If `None`, `X` is normalized.
        obsm
            Entry of `.obsm` to transform.
        inplace
            Whether to update `data` or return the matrix. Only applies to
            :class:`~anndata.AnnData` input.
        copy
            Whether to return a copy or update `data`. Not compatible with `inplace=False`.
            Only applies to :class:`~anndata.AnnData` input.

    Returns
    -------
    The resulting matrix after applying the logarithm of one plus the input matrix. \
    If a matrix is passed, the transformed matrix is returned. If an AnnData object is \
    passed and `copy` is `True`, returns the modified AnnData. Otherwise, updates the \
    `data` object in-place and returns None.

    """
    if not isinstance(data, AnnData):
        if layer is not None or obsm is not None:
            raise ValueError(
                "`layer` and `obsm` can only be used with an AnnData object."
            )
        X = data
        _check_gpu_X(X, allow_dask=True)
        return _calc_log1p(X, base=base)

    adata = data
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


def sqrt(
    data: AnnData | ArrayTypesDask,
    *,
    layer: str | None = None,
    obsm: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> Union[AnnData, spmatrix, cp.ndarray, None]:  # noqa: UP007
    """\
    Take the square root of the data matrix.

    Computes :math:`X = \\sqrt{X}`.

    Parameters
    ----------
        data
            The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond
            to cells and columns to genes. If a matrix is passed instead of an
            :class:`~anndata.AnnData` object, the transformed matrix is returned.
        layer
            Layer to transform instead of `X`. If `None`, `X` is transformed.
        obsm
            Entry of `.obsm` to transform.
        inplace
            Whether to update `data` or return the matrix. Only applies to
            :class:`~anndata.AnnData` input.
        copy
            Whether to return a copy or update `data`. Not compatible with `inplace=False`.
            Only applies to :class:`~anndata.AnnData` input.

    Returns
    -------
    The resulting matrix after applying the square root to the input matrix. \
    If a matrix is passed, the transformed matrix is returned. If an AnnData object is \
    passed and `copy` is `True`, returns the modified AnnData. Otherwise, updates the \
    `data` object in-place and returns None.

    """
    if not isinstance(data, AnnData):
        if layer is not None or obsm is not None:
            raise ValueError(
                "`layer` and `obsm` can only be used with an AnnData object."
            )
        X = data
        _check_gpu_X(X, allow_dask=True)
        return _calc_sqrt(X)

    adata = data
    if copy:
        if not inplace:
            raise ValueError("`copy=True` cannot be used with `inplace=False`.")
        adata = adata.copy()
    X = _get_obs_rep(adata, layer=layer, obsm=obsm)

    _check_gpu_X(X, allow_dask=True)

    if not inplace:
        X = X.copy()

    X = _calc_sqrt(X)
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
