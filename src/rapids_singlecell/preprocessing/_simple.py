from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
from anndata import AnnData

from ._qc import _basic_qc
from ._utils import _check_gpu_X

if TYPE_CHECKING:
    import numpy as np

    from rapids_singlecell._utils import ArrayTypesDask


def flag_gene_family(
    adata: AnnData,
    *,
    gene_family_name: str,
    gene_family_prefix: str = None,
    gene_list: list = None,
) -> None:
    """
    Flags a gene or gene_family in .var with boolean. (e.g all mitochondrial genes).
    Please only choose gene_family prefix or gene_list

    Parameters
    ----------
        adata
            AnnData object

        gene_family_name
            name of columns in .var where you want to store informationa as a boolean

        gene_family_prefix
            prefix of the gene family (eg. mt- for all mitochondrial genes in mice)

        gene_list
            list of genes to flag in `.var`

    Returns
    -------
        adds the boolean column in `.var`

    """
    if gene_family_prefix is not None:
        adata.var[gene_family_name] = cp.asnumpy(
            adata.var.index.str.startswith(gene_family_prefix)
        ).ravel()
    if gene_list is not None:
        adata.var[gene_family_name] = cp.asnumpy(
            adata.var.index.isin(gene_list)
        ).ravel()


def filter_genes(
    data: AnnData | ArrayTypesDask,
    *,
    min_counts: int | None = None,
    min_cells: int | None = None,
    max_counts: int | None = None,
    max_cells: int | None = None,
    inplace: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """\
    Filter genes based on number of cells or counts.

    Keep genes that have at least `min_counts` counts or are expressed in at
    least `min_cells` cells or have at most `max_counts` counts or are expressed
    in at most `max_cells` cells.

    Only provide one of the optional parameters `min_counts`, `min_cells`,
    `max_counts`, `max_cells` per call.

    Parameters
    ----------
    data
        An annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    min_counts
        Minimum number of counts required for a gene to pass filtering.
    min_cells
        Minimum number of cells expressed required for a gene to pass filtering.
    max_counts
        Maximum number of counts required for a gene to pass filtering.
    max_cells
        Maximum number of cells expressed required for a gene to pass filtering.
    inplace
        Perform computation inplace or return result.
    verbose
        Print number of discarded genes

    Returns
    -------
    Depending on `inplace`, returns the following arrays or directly subsets
    and annotates the data matrix

    gene_subset
        Boolean index mask that does filtering. `True` means that the
        gene is kept. `False` means the gene is removed.
    number_per_gene
        Depending on what was thresholded (`counts` or `cells`), the array stores
        `n_counts` or `n_cells` per gene.
    """
    n_given_options = sum(
        option is not None for option in [min_cells, min_counts, max_cells, max_counts]
    )
    if n_given_options != 1:
        msg = (
            "Only provide one of the optional parameters `min_counts`, "
            "`min_cells`, `max_counts`, `max_cells` per call."
        )
        raise ValueError(msg)

    if isinstance(data, AnnData):
        X = data.X
    else:
        X = data  # proceed with processing the data matrix

    _check_gpu_X(X, allow_dask=True)
    _, sums_genes, _, n_cells_per_gene = _basic_qc(X=X)
    min_number = min_counts if min_cells is None else min_cells
    max_number = max_counts if max_cells is None else max_cells
    number_per_gene = (
        n_cells_per_gene
        if (min_cells is not None or max_cells is not None)
        else sums_genes
    )
    if min_number is not None:
        gene_subset = number_per_gene >= min_number
    if max_number is not None:
        gene_subset = number_per_gene <= max_number

    if verbose:
        s = cp.sum(~gene_subset)
        if s > 0:
            msg = f"filtered out {s} genes that are detected "
            if min_cells is not None or min_counts is not None:
                msg += "in less than "
                msg += (
                    f"{min_cells} cells"
                    if min_counts is None
                    else f"{min_counts} counts"
                )
            if max_cells is not None or max_counts is not None:
                msg += "in more than "
                msg += (
                    f"{max_cells} cells"
                    if max_counts is None
                    else f"{max_counts} counts"
                )
            print(msg)

    if isinstance(data, AnnData) and inplace:
        data.var["n_counts"] = sums_genes.get()

        data.var["n_cells"] = n_cells_per_gene.get()
        data._inplace_subset_var(gene_subset.get())
    else:
        return gene_subset.get(), number_per_gene.get()


def filter_cells(
    data: AnnData | ArrayTypesDask,
    *,
    min_counts: int | None = None,
    min_genes: int | None = None,
    max_counts: int | None = None,
    max_genes: int | None = None,
    inplace: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """\
    Filter cell outliers based on counts and numbers of genes expressed.

    For instance, only keep cells with at least `min_counts` counts or
    `min_genes` genes expressed. This is to filter measurement outliers,
    i.e. “unreliable” observations.

    Only provide one of the optional parameters `min_counts`, `min_genes`,
    `max_counts`, `max_genes` per call.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    min_counts
        Minimum number of counts required for a cell to pass filtering.
    min_genes
        Minimum number of genes expressed required for a cell to pass filtering.
    max_counts
        Maximum number of counts required for a cell to pass filtering.
    max_genes
        Maximum number of genes expressed required for a cell to pass filtering.
    inplace
        Perform computation inplace or return result.
    verbose
        Print number of discarded cells

    Returns
    -------
    Depending on `inplace`, returns the following arrays or directly subsets
    and annotates the data matrix:

    cells_subset
        Boolean index mask that does filtering. `True` means that the
        cell is kept. `False` means the cell is removed.
    number_per_cell
        Depending on what was thresholded (`counts` or `genes`),
        the array stores `n_counts` or `n_cells` per gene.

    Examples
    --------
    >>> import scanpy as sc
    >>> adata = sc.datasets.krumsiek11()
    UserWarning: Observation names are not unique. To make them unique, call `.obs_names_make_unique`.
        utils.warn_names_duplicates("obs")
    >>> adata.obs_names_make_unique()
    >>> adata.n_obs
    640
    >>> adata.var_names.tolist()  # doctest: +NORMALIZE_WHITESPACE
    ['Gata2', 'Gata1', 'Fog1', 'EKLF', 'Fli1', 'SCL',
     'Cebpa', 'Pu.1', 'cJun', 'EgrNab', 'Gfi1']
    >>> # add some true zeros
    >>> adata.X[adata.X < 0.3] = 0
    >>> # simply compute the number of genes per cell
    >>> sc.pp.filter_cells(adata, min_genes=0)
    >>> adata.n_obs
    640
    >>> int(adata.obs['n_genes'].min())
    1
    >>> # filter manually
    >>> adata_copy = adata[adata.obs['n_genes'] >= 3]
    >>> adata_copy.n_obs
    554
    >>> int(adata_copy.obs['n_genes'].min())
    3
    >>> # actually do some filtering
    >>> sc.pp.filter_cells(adata, min_genes=3)
    >>> adata.n_obs
    554
    >>> int(adata.obs['n_genes'].min())
    3
    """
    n_given_options = sum(
        option is not None for option in [min_genes, min_counts, max_genes, max_counts]
    )
    if n_given_options != 1:
        msg = (
            "Only provide one of the optional parameters `min_counts`, "
            "`min_genes`, `max_counts`, `max_genes` per call."
        )
        raise ValueError(msg)
    if isinstance(data, AnnData):
        X = data.X
    else:
        X = data

    _check_gpu_X(X, allow_dask=True)
    sums_cells, _, n_genes_per_cell, _ = _basic_qc(X=X)
    min_number = min_counts if min_genes is None else min_genes
    max_number = max_counts if max_genes is None else max_genes
    number_per_cell = (
        n_genes_per_cell
        if (min_genes is not None or max_genes is not None)
        else sums_cells
    )

    if min_number is not None:
        cell_subset = number_per_cell >= min_number
    if max_number is not None:
        cell_subset = number_per_cell <= max_number

    if verbose:
        s = cp.sum(~cell_subset)
        if s > 0:
            msg = f"filtered out {s} cells that have "
            if min_genes is not None or min_counts is not None:
                msg += "less than "
                msg += (
                    f"{min_genes} genes expressed"
                    if min_counts is None
                    else f"{min_counts} counts"
                )
            if max_genes is not None or max_counts is not None:
                msg += "more than "
                msg += (
                    f"{max_genes} genes expressed"
                    if max_counts is None
                    else f"{max_counts} counts"
                )
            print(msg)

    if isinstance(data, AnnData) and inplace:
        data.obs["n_counts"] = sums_cells.get()
        data.obs["n_genes"] = n_genes_per_cell.get()
        data._inplace_subset_obs(cell_subset.get())
    else:
        return cell_subset.get(), number_per_cell.get()


def filter_highly_variable(adata: AnnData) -> None:
    """
    Filters the :class:`~anndata.AnnData` object for highly_variable genes. Run highly_varible_genes first.

    Returns
    -------
        updates :class:`~anndata.AnnData` object to only contain highly variable genes.

    """
    if "highly_variable" in adata.var.keys():
        adata._inplace_subset_var(adata.var["highly_variable"])
    else:
        print("Please calculate highly variable genes first")
