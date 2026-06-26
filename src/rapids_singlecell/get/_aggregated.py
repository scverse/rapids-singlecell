from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union, get_args

import cupy as cp
from anndata import AnnData
from cupyx.scipy import sparse as cp_sparse
from scanpy._utils import _resolve_axis
from scanpy.get._aggregated import _combine_categories

from rapids_singlecell._compat import DaskArray, _meta_dense
from rapids_singlecell._cuda import _aggr_cuda
from rapids_singlecell.get import _check_mask
from rapids_singlecell.preprocessing._utils import _check_gpu_X

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray

Array = Union[cp.ndarray, cp_sparse.csc_matrix, cp_sparse.csr_matrix]  # noqa: UP007
AggType = Literal["count_nonzero", "mean", "sum", "var", "sq_sum"]
_ALL_FUNCS = frozenset(get_args(AggType))


def _planes_for_funcs(funcs):
    """Map requested metrics to the raw accumulator planes they require.

    Returns a ``(need_sum, need_count, need_sqsum)`` tuple. The ``sum`` plane
    backs ``sum``/``mean``/``var``, the ``count`` plane backs ``count_nonzero``,
    and the ``sqsum`` plane backs ``var``/``sq_sum``.
    """
    need_sum = not funcs.isdisjoint(("sum", "mean", "var"))
    need_count = "count_nonzero" in funcs
    need_sqsum = not funcs.isdisjoint(("var", "sq_sum"))
    return need_sum, need_count, need_sqsum


class Aggregate:
    """\
    Functionality for generic grouping and aggregating.

    There is currently support for count_nonzero, sum, mean, and variance.

    Params
    ------
    groupby
        :class:`~pandas.Categorical` containing values for grouping by.
    data
        Data matrix for aggregation.
    mask
        Mask to be used for aggregation.
    """

    def __init__(
        self,
        groupby: pd.Categorical,
        data: Array,
        *,
        mask: NDArray[np.bool_] | None = None,
    ) -> None:
        self.mask = mask
        self.groupby = cp.array(groupby.codes, dtype=cp.int32)
        self.n_cells = cp.array(cp.bincount(self.groupby), dtype=cp.float64).reshape(
            -1, 1
        )
        if data.dtype.kind != "f" and not isinstance(data, DaskArray):
            data = data.astype(cp.float32, copy=False)
        self.data = data

    groupby: cp.ndarray
    data: Array

    def _get_mask(self):
        if self.mask is not None:
            return cp.array(self.mask)
        else:
            return cp.ones(self.data.shape[0], dtype=bool)

    def _build_result(self, funcs, *, sums, counts, sq_sums, dof):
        """Compute the requested derived statistics from raw accumulators.

        Only the planes a metric depends on need to be provided: ``sum``/``mean``
        need ``sums``, ``var`` needs ``sums`` and ``sq_sums``, ``sq_sum`` needs
        ``sq_sums``, and ``count_nonzero`` needs ``counts``.
        """
        results = {}
        if "sum" in funcs:
            results["sum"] = sums
        if "sq_sum" in funcs:
            results["sq_sum"] = sq_sums
        if "count_nonzero" in funcs:
            results["count_nonzero"] = counts.astype(cp.int32)
        if "mean" in funcs or "var" in funcs:
            means = sums / self.n_cells
            if "mean" in funcs:
                results["mean"] = means
            if "var" in funcs:
                var = sq_sums / self.n_cells - means**2
                var *= self.n_cells / (self.n_cells - dof)
                results["var"] = var
        return results

    def count_mean_var(self, funcs=None, *, dof: int = 1, split_every: int = 2):
        """Dispatch to the appropriate count_mean_var method based on data type.

        ``funcs`` is the set of requested metrics; only the accumulators those
        metrics need are allocated and computed. ``None`` computes everything.
        """
        funcs = _ALL_FUNCS if funcs is None else set(funcs)
        if isinstance(self.data, DaskArray):
            return self.count_mean_var_dask(funcs, dof=dof, split_every=split_every)
        elif cp_sparse.issparse(self.data):
            return self.count_mean_var_sparse(funcs, dof=dof)
        else:
            return self.count_mean_var_dense(funcs, dof=dof)

    def count_mean_var_dask(self, funcs=None, *, dof: int = 1, split_every: int = 2):
        """
        This function is used to calculate the sum, mean, and variance of the data matrix.
        It automatically detects sparse vs dense matrices and uses the appropriate
        CUDA kernel for aggregation. Only the accumulator planes the requested
        ``funcs`` need are allocated and reduced across blocks.
        """
        import dask.array as da

        assert dof >= 0
        funcs = _ALL_FUNCS if funcs is None else set(funcs)
        need_sum, need_count, need_sqsum = _planes_for_funcs(funcs)

        # Compact plane layout: only the requested accumulators, canonical order.
        plane_order = []
        if need_sum:
            plane_order.append("sum")
        if need_count:
            plane_order.append("count")
        if need_sqsum:
            plane_order.append("sqsum")
        n_planes = len(plane_order)
        i_sum = plane_order.index("sum") if need_sum else None
        i_count = plane_order.index("count") if need_count else None
        i_sqsum = plane_order.index("sqsum") if need_sqsum else None

        is_sparse = not isinstance(self.data._meta, cp.ndarray)
        n_groups = self.n_cells.shape[0]

        def __aggregate_dask(X_part, mask_part, groupby_part):
            out = cp.zeros(
                (1, n_planes, n_groups, self.data.shape[1]), dtype=cp.float64
            )
            gb = groupby_part.ravel()
            mk = mask_part.ravel()
            out_sum = out[0, i_sum] if need_sum else None
            out_count = out[0, i_count] if need_count else None
            out_sqsum = out[0, i_sqsum] if need_sqsum else None

            if is_sparse:
                _aggr_cuda.sparse_aggr(
                    X_part.indptr,
                    X_part.indices,
                    X_part.data,
                    out_sum=out_sum,
                    out_count=out_count,
                    out_sqsum=out_sqsum,
                    cats=gb,
                    mask=mk,
                    n_cells=X_part.shape[0],
                    n_genes=X_part.shape[1],
                    is_csc=False,
                )
            else:
                _aggr_cuda.dense_aggr(
                    X_part,
                    out_sum=out_sum,
                    out_count=out_count,
                    out_sqsum=out_sqsum,
                    cats=gb,
                    mask=mk,
                    n_cells=X_part.shape[0],
                    n_genes=X_part.shape[1],
                    is_fortran=X_part.flags.f_contiguous,
                )
            return out

        # Prepare Dask arrays
        mask = self._get_mask()
        mask_dask = da.from_array(
            mask, chunks=(self.data.chunks[0]), meta=_meta_dense(mask.dtype)
        )
        groupby_dask = da.from_array(
            self.groupby,
            chunks=(self.data.chunks[0]),
            meta=_meta_dense(self.groupby.dtype),
        )

        # Apply aggregation across all blocks
        out = da.map_blocks(
            __aggregate_dask,
            self.data,
            mask_dask[..., None],
            groupby_dask[..., None],
            meta=cp.empty([], dtype=cp.float64),
            dtype=cp.float64,
            new_axis=(1, 2),
            chunks=(
                (1,) * self.data.blocks.size,
                (n_planes,),
                (n_groups,),
                (self.data.shape[1],),
            ),
        )

        # Compute final aggregated results
        out = out.sum(axis=0, split_every=split_every).compute()
        sums = out[i_sum] if need_sum else None
        counts = out[i_count] if need_count else None
        sq_sums = out[i_sqsum] if need_sqsum else None
        return self._build_result(
            funcs, sums=sums, counts=counts, sq_sums=sq_sums, dof=dof
        )

    def count_mean_var_sparse(self, funcs=None, *, dof: int = 1):
        """
        This function is used to calculate the sum, mean, and variance of the sparse data matrix.
        It uses a custom cuda-kernel to perform the aggregation. Only the
        accumulator planes the requested ``funcs`` need are allocated.
        """

        assert dof >= 0
        funcs = _ALL_FUNCS if funcs is None else set(funcs)
        need_sum, need_count, need_sqsum = _planes_for_funcs(funcs)
        n_groups, n_genes = self.n_cells.shape[0], self.data.shape[1]

        sums = cp.zeros((n_groups, n_genes), dtype=cp.float64) if need_sum else None
        counts = cp.zeros((n_groups, n_genes), dtype=cp.float64) if need_count else None
        sq_sums = (
            cp.zeros((n_groups, n_genes), dtype=cp.float64) if need_sqsum else None
        )
        mask = self._get_mask()

        _aggr_cuda.sparse_aggr(
            self.data.indptr,
            self.data.indices,
            self.data.data,
            out_sum=sums,
            out_count=counts,
            out_sqsum=sq_sums,
            cats=self.groupby,
            mask=mask,
            n_cells=self.data.shape[0],
            n_genes=n_genes,
            is_csc=self.data.format == "csc",
        )
        return self._build_result(
            funcs, sums=sums, counts=counts, sq_sums=sq_sums, dof=dof
        )

    def count_mean_var_sparse_sparse(self, funcs, dof: int = 1):
        """
        This function is used to calculate the sum, mean, and variance of the sparse data matrix.
        It uses a custom cuda-kernel to perform the aggregation.
        """

        assert dof >= 0
        from ._kernels._aggr_elementwise import (
            _scatter_count_nonzero,
            _scatter_mean_var,
            _scatter_sum,
            _sum_duplicates_assign,
            _sum_duplicates_diff,
        )

        if self.data.format == "csc":
            self.data = self.data.tocsr()

        index_dtype = cp.result_type(self.data.indptr.dtype, self.data.indices.dtype)
        if index_dtype.itemsize < 8 and self.data.nnz > cp.iinfo(cp.int32).max:
            index_dtype = cp.dtype(cp.int64)

        src_row = cp.zeros(self.data.nnz, dtype=index_dtype)
        src_col = cp.zeros(self.data.nnz, dtype=index_dtype)
        src_data = cp.zeros(self.data.nnz, dtype=cp.float64)
        mask = self._get_mask()

        _aggr_cuda.csr_to_coo(
            self.data.indptr,
            self.data.indices,
            self.data.data,
            out_row=src_row,
            out_col=src_col,
            out_data=src_data,
            cats=self.groupby,
            mask=mask,
            n_cells=self.data.shape[0],
        )
        n_groups = self.n_cells.shape[0]
        fused_key = src_row.astype(cp.int64) * self.data.shape[1] + src_col
        order = cp.argsort(fused_key)

        src_row = src_row[order]
        src_col = src_col[order]
        src_data = src_data[order]

        diff = _sum_duplicates_diff(src_row, src_col, size=src_row.size)
        index = cp.cumsum(diff, dtype=index_dtype)
        nnz = int(index[-1].get())

        # calculate the rows and indices
        rows = cp.zeros(nnz + 1, dtype=index_dtype)
        indices = cp.zeros(nnz + 1, dtype=index_dtype)

        _sum_duplicates_assign(src_row, src_col, index, rows, indices)

        # Calculate the indptr using searchsorted to handle empty groups
        group_boundaries = cp.arange(n_groups + 1, dtype=index_dtype)
        indptr = cp.searchsorted(rows[: nnz + 1], group_boundaries).astype(index_dtype)
        indptr[-1] = nnz + 1

        # Calculate the the sparse matrices
        results = {}

        if "sum" in funcs:
            sums = cp.zeros(nnz + 1, dtype=cp.float64)
            _scatter_sum(src_data, index, sums)

            results["sum"] = cp_sparse.csr_matrix(
                (sums, indices, indptr),
                shape=(self.n_cells.shape[0], self.data.shape[1]),
            )
        if "sq_sum" in funcs:
            sq_sums = cp.zeros(nnz + 1, dtype=cp.float64)
            _scatter_sum(src_data * src_data, index, sq_sums)
            results["sq_sum"] = cp_sparse.csr_matrix(
                (sq_sums, indices, indptr),
                shape=(self.n_cells.shape[0], self.data.shape[1]),
            )
        if "var" in funcs or "mean" in funcs:
            means = cp.zeros(nnz + 1, dtype=cp.float64)
            var = cp.zeros(nnz + 1, dtype=cp.float64)
            _scatter_mean_var(src_data, index, means, var)
            n_cells_sparse = self.n_cells[rows[: nnz + 1]].ravel()
            means /= n_cells_sparse
            var /= n_cells_sparse
            if "mean" in funcs:
                results["mean"] = cp_sparse.csr_matrix(
                    (means, indices, indptr),
                    shape=(self.n_cells.shape[0], self.data.shape[1]),
                )
            if "var" in funcs:
                var = cp_sparse.csr_matrix(
                    (var, indices, indptr),
                    shape=(self.n_cells.shape[0], self.data.shape[1]),
                )

                _aggr_cuda.sparse_var(
                    var.indptr,
                    var.indices,
                    var.data,
                    means=means,
                    n_cells=self.n_cells,
                    dof=int(dof),
                    n_groups=var.shape[0],
                    stream=cp.cuda.get_current_stream().ptr,
                )
                results["var"] = var
        if "count_nonzero" in funcs:
            counts = cp.zeros(nnz + 1, dtype=cp.float32)
            _scatter_count_nonzero(src_data, index, counts)
            results["count_nonzero"] = cp_sparse.csr_matrix(
                (counts, indices, indptr),
                shape=(self.n_cells.shape[0], self.data.shape[1]),
            )

        return results

    def count_mean_var_dense(self, funcs=None, *, dof: int = 1):
        """
        This function is used to calculate the sum, mean, and variance of the dense data matrix.
        It uses a custom cuda-kernel to perform the aggregation. Only the
        accumulator planes the requested ``funcs`` need are allocated.
        """

        assert dof >= 0
        funcs = _ALL_FUNCS if funcs is None else set(funcs)
        need_sum, need_count, need_sqsum = _planes_for_funcs(funcs)
        n_groups, n_genes = self.n_cells.shape[0], self.data.shape[1]

        sums = cp.zeros((n_groups, n_genes), dtype=cp.float64) if need_sum else None
        counts = cp.zeros((n_groups, n_genes), dtype=cp.float64) if need_count else None
        sq_sums = (
            cp.zeros((n_groups, n_genes), dtype=cp.float64) if need_sqsum else None
        )
        mask = self._get_mask()

        _aggr_cuda.dense_aggr(
            self.data,
            out_sum=sums,
            out_count=counts,
            out_sqsum=sq_sums,
            cats=self.groupby,
            mask=mask,
            n_cells=self.data.shape[0],
            n_genes=n_genes,
            is_fortran=self.data.flags.f_contiguous,
        )
        return self._build_result(
            funcs, sums=sums, counts=counts, sq_sums=sq_sums, dof=dof
        )


def aggregate(
    adata: AnnData,
    by: str | Collection[str],
    func: AggType | Iterable[AggType],
    *,
    axis: Literal["obs", 0, "var", 1] | None = None,
    mask: NDArray[np.bool_] | str | None = None,
    dof: int = 1,
    layer: str | None = None,
    obsm: str | None = None,
    varm: str | None = None,
    return_sparse: bool = False,
    **kwargs,
) -> AnnData:
    """\
    Aggregate data matrix based on some categorical grouping.

    This function is useful for pseudobulking as well as plotting.

    Aggregation to perform is specified by `func`, which can be a single metric or a
    list of metrics. Each metric is computed over the group and results in a new layer
    in the output `AnnData` object.

    If none of `layer`, `obsm`, or `varm` are passed in, `X` will be used for aggregation data.
    If `func` only has length 1 or is just an `AggType`, then aggregation data is written to `X`.
    Otherwise, it is written to `layers` or `xxxm` as appropriate for the dimensions of the aggregation data.

    Parameters
    ----------
    adata
        :class:`~anndata.AnnData` to be aggregated.
    by
        Key of the column to be grouped-by.
    func
        How to aggregate.
    axis
        Axis on which to find group by column.
    mask
        Boolean mask (or key to column containing mask) to apply along the axis.
    dof
        Degrees of freedom for variance. Defaults to 1.
    layer
        If not None, key for aggregation data.
    obsm
        If not None, key for aggregation data.
    varm
        If not None, key for aggregation data.
    return_sparse
        Whether to return a sparse matrix. Only works for sparse input data.

    Returns
    -------
    Aggregated :class:`~anndata.AnnData`.

    Examples
    --------

    Calculating mean expression and number of nonzero entries per cluster:

    >>> import scanpy as sc, pandas as pd
    >>> import rapids_singlecell as rsc
    >>> pbmc = sc.datasets.pbmc3k_processed().raw.to_adata()
    >>> rsc.get.anndata_to_GPU(pbmc)
    >>> pbmc.shape
    (2638, 13714)
    >>> aggregated = rsc.get.aggregate(pbmc, by="louvain", func=["mean", "count_nonzero"])
    >>> aggregated
    AnnData object with n_obs × n_vars = 8 × 13714
        obs: 'louvain'
        var: 'n_cells'
        layers: 'mean', 'count_nonzero'

    We can group over multiple columns:

    >>> pbmc.obs["percent_mito_binned"] = pd.cut(pbmc.obs["percent_mito"], bins=5)
    >>> rsc.get.aggregate(pbmc, by=["louvain", "percent_mito_binned"], func=["mean", "count_nonzero"])
    AnnData object with n_obs × n_vars = 40 × 13714
        obs: 'louvain', 'percent_mito_binned'
        var: 'n_cells'
        layers: 'mean', 'count_nonzero'

    Note that this filters out any combination of groups that wasn't present in the original data.
    """
    if axis is None:
        axis = 1 if varm else 0
    axis, axis_name = _resolve_axis(axis)
    if mask is not None:
        mask = _check_mask(adata, mask, axis_name)
    data = adata.X
    if sum(p is not None for p in [varm, obsm, layer]) > 1:
        raise TypeError("Please only provide one (or none) of varm, obsm, or layer")

    if varm is not None:
        if axis != 1:
            raise ValueError("varm can only be used when axis is 1")
        data = adata.varm[varm]
    elif obsm is not None:
        if axis != 0:
            raise ValueError("obsm can only be used when axis is 0")
        data = adata.obsm[obsm]
    elif layer is not None:
        data = adata.layers[layer]
        if axis == 1:
            data = data.T
    elif axis == 1:
        # i.e., all of `varm`, `obsm`, `layers` are None so we use `X` which must be transposed
        data = data.T
    _check_gpu_X(data, allow_dask=True)
    dim_df = getattr(adata, axis_name)
    categorical, new_label_df = _combine_categories(dim_df, by)
    groupby = Aggregate(groupby=categorical, data=data, mask=mask)

    funcs = set([func] if isinstance(func, str) else func)
    if unknown := funcs - set(get_args(AggType)):
        raise ValueError(f"func {unknown} is not one of {get_args(AggType)}")

    if return_sparse and cp_sparse.issparse(data):
        result = groupby.count_mean_var_sparse_sparse(funcs, dof)
    else:
        split_every = kwargs.pop("split_every", 2)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {set(kwargs)}")
        result = groupby.count_mean_var(funcs, dof=dof, split_every=split_every)
    layers = {}

    if "sum" in funcs:
        layers["sum"] = result["sum"]
    if "mean" in funcs:
        layers["mean"] = result["mean"]
    if "count_nonzero" in funcs:
        layers["count_nonzero"] = result["count_nonzero"]
    if "var" in funcs:
        layers["var"] = result["var"]
    if "sq_sum" in funcs:
        layers["sq_sum"] = result["sq_sum"]

    result = AnnData(
        layers=layers,
        obs=new_label_df,
        var=getattr(adata, "var" if axis == 0 else "obs"),
    )
    if axis == 1:
        return result.T
    else:
        return result
