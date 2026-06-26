from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
from cupyx.scipy import sparse
from scanpy.get import _get_obs_rep

from rapids_singlecell._compat import DaskArray

from ._utils import _check_gpu_X

if TYPE_CHECKING:
    from anndata import AnnData

    from rapids_singlecell._utils import ArrayTypesDask


def calculate_qc_metrics(
    adata: AnnData,
    *,
    expr_type: str = "counts",
    var_type: str = "genes",
    qc_vars: str | list = None,
    log1p: bool = True,
    layer: str = None,
) -> None:
    """\
    Calculates basic qc Parameters :cite:p:`McCarthy2017`.

    Calculates number of genes per cell (n_genes) and number of counts per cell (n_counts).
    Loosely based on calculate_qc_metrics from scanpy :cite:p:`Wolf2018`. Updates :attr:`~anndata.AnnData.obs` and :attr:`~anndata.AnnData.var`  with columns with qc data.

    Parameters
    ----------
        adata
            AnnData object
        expr_type
            Name of kind of values in X.
        var_type
            The kind of thing the variables are.
        qc_vars
            Keys for boolean columns of :attr:`~anndata.AnnData.var` which identify variables you could want to control for (e.g. Mito).
            Run flag_gene_family first
        log1p
            Set to `False` to skip computing `log1p` transformed annotations.
        layer
            If provided, use :attr:`~anndata.AnnData.layers` for expression values instead of :attr:`~anndata.AnnData.X`.

    Returns
    -------
        adds the following columns in :attr:`~anndata.AnnData.obs` :
            `total_{var_type}_by_{expr_type}`
                E.g. 'total_genes_by_counts'. Number of genes with positive counts in a cell.
            `total_{expr_type}`
                E.g. 'total_counts'. Total number of counts for a cell.
            for `qc_var` in `qc_vars`
                `total_{expr_type}_{qc_var}`
                    number of counts per qc_var (e.g total counts mitochondrial genes)
                `pct_{expr_type}_{qc_var}`
                    Proportion of counts of qc_var (percent of counts mitochondrial genes)

        adds the following columns in :attr:`~anndata.AnnData.var` :
            `total_{expr_type}`
                E.g. 'total_counts'. Sum of counts for a gene.
            `n_genes_by_{expr_type}`
                E.g. 'n_cells_by_counts'. Number of cells this expression is measured in.
            `mean_{expr_type}`
                E.g. "mean_counts". Mean expression over all cells.
            `pct_dropout_by_{expr_type}`
                E.g. 'pct_dropout_by_counts'. Percentage of cells this feature does not appear in.

    """

    X = _get_obs_rep(adata, layer=layer)

    _check_gpu_X(X, allow_dask=True)

    sums_cells, sums_genes, genes_per_cell, cells_per_gene = _basic_qc(X)
    # .var
    adata.var[f"n_cells_by_{expr_type}"] = cp.asnumpy(cells_per_gene)
    adata.var[f"total_{expr_type}"] = cp.asnumpy(sums_genes)
    mean_array = sums_genes / adata.n_obs
    adata.var[f"mean_{expr_type}"] = cp.asnumpy(mean_array)
    adata.var[f"pct_dropout_by_{expr_type}"] = cp.asnumpy(
        (1 - cells_per_gene / adata.n_obs) * 100
    )
    if log1p:
        adata.var[f"log1p_total_{expr_type}"] = cp.asnumpy(cp.log1p(sums_genes))
        adata.var[f"log1p_mean_{expr_type}"] = cp.asnumpy(cp.log1p(mean_array))
    # .obs
    adata.obs[f"n_{var_type}_by_{expr_type}"] = cp.asnumpy(genes_per_cell)
    adata.obs[f"total_{expr_type}"] = cp.asnumpy(sums_cells)
    if log1p:
        adata.obs[f"log1p_n_{var_type}_by_{expr_type}"] = cp.asnumpy(
            cp.log1p(genes_per_cell)
        )
        adata.obs[f"log1p_total_{expr_type}"] = cp.asnumpy(cp.log1p(sums_cells))

    if qc_vars:
        if isinstance(qc_vars, str):
            qc_vars = [qc_vars]
        for qc_var in qc_vars:
            mask = cp.array(adata.var[qc_var], dtype=cp.bool_)
            sums_cells_sub = _geneset_qc(X, mask)

            adata.obs[f"total_{expr_type}_{qc_var}"] = cp.asnumpy(sums_cells_sub)
            adata.obs[f"pct_{expr_type}_{qc_var}"] = cp.asnumpy(
                sums_cells_sub / sums_cells * 100
            )
            if log1p:
                adata.obs[f"log1p_total_{expr_type}_{qc_var}"] = cp.asnumpy(
                    cp.log1p(sums_cells_sub)
                )


def _basic_qc(
    X: ArrayTypesDask,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    if isinstance(X, DaskArray):
        return _basic_qc_dask(X)

    sums_cells = cp.zeros(X.shape[0], dtype=X.dtype)
    sums_genes = cp.zeros(X.shape[1], dtype=X.dtype)
    genes_per_cell = cp.zeros(X.shape[0], dtype=cp.int32)
    cells_per_gene = cp.zeros(X.shape[1], dtype=cp.int32)
    if sparse.issparse(X):
        from rapids_singlecell._cuda import _qc_cuda as _qc

        if sparse.isspmatrix_csr(X):
            sparse_qc = _qc.sparse_qc_csr
            is_csr = True
        elif sparse.isspmatrix_csc(X):
            sparse_qc = _qc.sparse_qc_csc
            is_csr = False
        else:
            raise ValueError("Please use a csr or csc matrix")

        sparse_qc(
            X.indptr,
            X.indices,
            X.data,
            sums_cells=sums_cells,
            sums_genes=sums_genes,
            cell_ex=genes_per_cell,
            gene_ex=cells_per_gene,
            **({"n_cells": X.shape[0]} if is_csr else {"n_genes": X.shape[1]}),
            stream=cp.cuda.get_current_stream().ptr,
        )
    else:
        from rapids_singlecell._cuda import _qc_cuda as _qc

        if not X.flags.c_contiguous:
            X = cp.asarray(X, order="C")
        _qc.sparse_qc_dense(
            X,
            sums_cells=sums_cells,
            sums_genes=sums_genes,
            cell_ex=genes_per_cell,
            gene_ex=cells_per_gene,
            n_cells=X.shape[0],
            n_genes=X.shape[1],
            stream=cp.cuda.get_current_stream().ptr,
        )
    return sums_cells, sums_genes, genes_per_cell, cells_per_gene


def _basic_qc_dask(
    X: DaskArray,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    import dask

    if isinstance(X._meta, sparse.csr_matrix):
        from rapids_singlecell._cuda import _qc_dask_cuda as _qcd

        def __qc_calc_1(X_part):
            sums_cells = cp.zeros(X_part.shape[0], dtype=X_part.dtype)
            genes_per_cell = cp.zeros(X_part.shape[0], dtype=cp.int32)
            _qcd.sparse_qc_csr_cells(
                X_part.indptr,
                X_part.indices,
                X_part.data,
                sums_cells=sums_cells,
                cell_ex=genes_per_cell,
                n_cells=X_part.shape[0],
                stream=cp.cuda.get_current_stream().ptr,
            )
            return cp.stack([sums_cells, genes_per_cell.astype(X_part.dtype)], axis=1)

        def __qc_calc_2(X_part):
            sums_genes = cp.zeros(X_part.shape[1], dtype=X_part.dtype)
            cells_per_gene = cp.zeros(X_part.shape[1], dtype=cp.int32)
            _qcd.sparse_qc_csr_genes(
                X_part.indices,
                X_part.data,
                sums_genes=sums_genes,
                gene_ex=cells_per_gene,
                nnz=X_part.nnz,
                stream=cp.cuda.get_current_stream().ptr,
            )
            return cp.vstack([sums_genes, cells_per_gene.astype(X_part.dtype)])[
                None, ...
            ]

    elif isinstance(X._meta, cp.ndarray):
        from rapids_singlecell._cuda import _qc_dask_cuda as _qcd

        def __qc_calc_1(X_part):
            sums_cells = cp.zeros(X_part.shape[0], dtype=X_part.dtype)
            genes_per_cell = cp.zeros(X_part.shape[0], dtype=cp.int32)
            if not X_part.flags.c_contiguous:
                X_part = cp.asarray(X_part, order="C")
            _qcd.sparse_qc_dense_cells(
                X_part,
                sums_cells=sums_cells,
                cell_ex=genes_per_cell,
                n_cells=X_part.shape[0],
                n_genes=X_part.shape[1],
                stream=cp.cuda.get_current_stream().ptr,
            )
            return cp.stack([sums_cells, genes_per_cell.astype(X_part.dtype)], axis=1)

        def __qc_calc_2(X_part):
            sums_genes = cp.zeros((X_part.shape[1]), dtype=X_part.dtype)
            cells_per_gene = cp.zeros((X_part.shape[1]), dtype=cp.int32)
            if not X_part.flags.c_contiguous:
                X_part = cp.asarray(X_part, order="C")
            _qcd.sparse_qc_dense_genes(
                X_part,
                sums_genes=sums_genes,
                gene_ex=cells_per_gene,
                n_cells=X_part.shape[0],
                n_genes=X_part.shape[1],
                stream=cp.cuda.get_current_stream().ptr,
            )
            return cp.vstack([sums_genes, cells_per_gene.astype(X_part.dtype)])[
                None, ...
            ]
    else:
        raise ValueError(
            "Please use a cupy csr_matrix or cp.ndarray. csc_matrix are not supported with dask."
        )

    cell_results = X.map_blocks(
        __qc_calc_1,
        chunks=(X.chunks[0], (2,)),
        dtype=X.dtype,
        meta=cp.empty((0, 2), dtype=X.dtype),
    )
    sums_cells = cell_results[:, 0]
    genes_per_cell = cell_results[:, 1]

    n_blocks = X.blocks.size
    sums_genes, cells_per_gene = X.map_blocks(
        __qc_calc_2,
        new_axis=(1,),
        chunks=((1,) * n_blocks, (2,), (X.shape[1],)),
        dtype=X.dtype,
        meta=cp.array([]),
    ).sum(axis=0)

    sums_cells, genes_per_cell, sums_genes, cells_per_gene = dask.compute(
        sums_cells, genes_per_cell, sums_genes, cells_per_gene
    )

    return (
        sums_cells.ravel(),
        sums_genes.ravel(),
        genes_per_cell.ravel().astype(cp.int32),
        cells_per_gene.ravel().astype(cp.int32),
    )


def _geneset_qc(X: ArrayTypesDask, mask: cp.ndarray) -> cp.ndarray:
    if isinstance(X, DaskArray):
        return _geneset_qc_dask(X, mask)
    sums_cells_sub = cp.zeros(X.shape[0], dtype=X.dtype)
    from rapids_singlecell._cuda import _qc_cuda as _qc

    if sparse.issparse(X):
        if sparse.isspmatrix_csr(X):
            _qc.sparse_qc_csr_sub(
                X.indptr,
                X.indices,
                X.data,
                sums_cells=sums_cells_sub,
                mask=mask,
                n_cells=X.shape[0],
                stream=cp.cuda.get_current_stream().ptr,
            )
        elif sparse.isspmatrix_csc(X):
            _qc.sparse_qc_csc_sub(
                X.indptr,
                X.indices,
                X.data,
                sums_cells=sums_cells_sub,
                mask=mask,
                n_genes=X.shape[1],
                stream=cp.cuda.get_current_stream().ptr,
            )
        else:
            raise ValueError("Please use a csr or csc matrix")
    else:
        if not X.flags.c_contiguous:
            X = cp.asarray(X, order="C")
        _qc.sparse_qc_dense_sub(
            X,
            sums_cells=sums_cells_sub,
            mask=mask,
            n_cells=X.shape[0],
            n_genes=X.shape[1],
            stream=cp.cuda.get_current_stream().ptr,
        )
    return sums_cells_sub


def _geneset_qc_dask(X: DaskArray, mask: cp.ndarray) -> cp.ndarray:
    if isinstance(X._meta, sparse.csr_matrix):
        from rapids_singlecell._cuda import _qc_cuda as _qc

        def __qc_calc(X_part):
            sums_cells_sub = cp.zeros(X_part.shape[0], dtype=X_part.dtype)
            _qc.sparse_qc_csr_sub(
                X_part.indptr,
                X_part.indices,
                X_part.data,
                sums_cells=sums_cells_sub,
                mask=mask,
                n_cells=X_part.shape[0],
                stream=cp.cuda.get_current_stream().ptr,
            )
            return sums_cells_sub

    elif isinstance(X._meta, cp.ndarray):
        from rapids_singlecell._cuda import _qc_cuda as _qc

        def __qc_calc(X_part):
            sums_cells_sub = cp.zeros(X_part.shape[0], dtype=X_part.dtype)
            if not X_part.flags.c_contiguous:
                X_part = cp.asarray(X_part, order="C")
            _qc.sparse_qc_dense_sub(
                X_part,
                sums_cells=sums_cells_sub,
                mask=mask,
                n_cells=X_part.shape[0],
                n_genes=X_part.shape[1],
                stream=cp.cuda.get_current_stream().ptr,
            )
            return sums_cells_sub

    sums_cells_sub = X.map_blocks(
        __qc_calc, dtype=X.dtype, meta=cp.array([]), drop_axis=1
    ).compute()
    return sums_cells_sub.ravel()
