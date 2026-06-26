from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
)

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy import sparse as sparse_gpu
from scipy import sparse
from statsmodels.stats.multitest import multipletests

from rapids_singlecell.preprocessing._utils import _sparse_to_dense

from ._gearysc import _gearys_C_cupy
from ._moransi import _morans_I_cupy
from ._utils import _p_value_calc

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData


def _to_cupy(vals, *, use_sparse: bool, dtype):
    """Convert input data to CuPy array with specified dtype.

    Note: `use_sparse` is only relevant for sparse input data.
    Dense input is always returned as a dense CuPy array.
    """
    is_sparse = sparse.issparse(vals) or sparse_gpu.isspmatrix(vals)

    # Dense input - use_sparse is ignored
    if not is_sparse:
        return cp.array(vals, dtype=dtype, order="C")

    # Sparse input - respect use_sparse parameter
    if use_sparse:
        if sparse_gpu.isspmatrix(vals):
            return vals.tocsr().astype(dtype)
        return sparse_gpu.csr_matrix(vals.tocsr(), dtype=dtype)

    # Sparse input but use_sparse=False - convert to dense
    if not sparse_gpu.isspmatrix(vals):
        vals = sparse_gpu.csr_matrix(vals.tocsr(), dtype=dtype)
    return _sparse_to_dense(vals, order="C")


def spatial_autocorr(
    adata: AnnData,
    *,
    connectivity_key: str = "spatial_connectivities",
    genes: str | Sequence[str] | None = None,
    mode: Literal["moran", "geary"] = "moran",
    transformation: bool = True,
    n_perms: int | None = None,
    two_tailed: bool = False,
    corr_method: str | None = "fdr_bh",
    layer: str | None = None,
    use_raw: bool = False,
    use_sparse: bool = True,
    dtype: np.dtype | None = None,
    multi_gpu: bool | list[int] | str | None = None,
    copy: bool = False,
) -> pd.DataFrame | None:
    """
    Calculate spatial autocorrelation for genes in an AnnData object.

    This function computes spatial autocorrelation scores (Moran's I or Geary's C) for each gene in an AnnData object.
    The function also calculates p-values and corrected p-values for multiple testing.

    Note:
        The precision (float32 or float64) is determined by the input data dtype.
        For best numerical stability, especially with genes having low variance,
        use float64 input data.

    Parameters
    ----------
        adata
            Annotated data matrix.
        connectivity_key
            Key of the connectivity matrix in `adata.obsp`, by default "spatial_connectivities".
        genes
            Genes for which to compute the autocorrelation scores. If None, all genes or highly variable genes will be used.
        mode
            Spatial autocorrelation method to use, either "moran" or "geary", by default "moran".
        transformation
            If True, row-normalize the connectivity matrix, by default True.
        n_perms
            Number of permutations for calculating p-values, by default None.
        two_tailed
            If True, calculate two-tailed p-values, by default False.
        corr_method
            Multiple testing correction method to use, by default "fdr_bh".
        layer
            Layer in the AnnData object to use, by default None.
        use_raw
            If True, use the raw data in the AnnData object, by default False.
        use_sparse
            If True, use a sparse representation for the input matrix `vals` when it is a sparse matrix, by default True.
        dtype
            Data type for computation. If None, uses input data dtype.
            For best numerical stability, especially with genes having low variance, use `np.float64`.
        multi_gpu
            GPU selection for permutation tests:
            - None: Use all GPUs if available (default)
            - True: Use all available GPUs
            - False: Use only GPU 0
            - list[int]: Use specific GPU IDs (e.g., [0, 2])
            - str: Comma-separated GPU IDs (e.g., "0,2")
        copy
            If True, return the results as a DataFrame instead of storing them in `adata.uns`, by default False.

    Returns
    -------
            DataFrame containing the autocorrelation scores, p-values, and corrected p-values for each gene. \
            If `copy` is False, the results are stored in `adata.uns` and None is returned.
    """
    if genes is None:
        if "highly_variable" in adata.var:
            genes = adata[:, adata.var["highly_variable"]].var_names.values
        else:
            genes = adata.var_names.values
    if isinstance(genes, str):
        genes = [genes]
    if use_raw:
        if adata.raw is None:
            raise AttributeError(
                "No `.raw` attribute found. Try specifying `use_raw=False`."
            )
        genes = list(set(genes) & set(adata.raw.var_names))
        vals = adata.raw[:, genes].X
    else:
        if layer is not None:
            vals = adata[:, genes].layers[layer]
        else:
            vals = adata[:, genes].X

    # Determine dtype from parameter or input data
    compute_dtype = np.dtype(dtype) if dtype is not None else vals.dtype
    if compute_dtype not in (np.float32, np.float64):
        compute_dtype = np.float32

    # create Adj-Matrix
    adj_matrix = adata.obsp[connectivity_key]
    adj_matrix_cupy = sparse_gpu.csr_matrix(adj_matrix, dtype=compute_dtype)

    if transformation:  # row-normalize
        row_sums = adj_matrix_cupy.sum(axis=1).reshape(-1, 1)
        non_zero_rows = row_sums != 0
        row_sums[non_zero_rows] = 1.0 / row_sums[non_zero_rows]
        adj_matrix_cupy = adj_matrix_cupy.multiply(sparse_gpu.csr_matrix(row_sums))

    params = {"two_tailed": two_tailed}

    def _run_autocorr(data, adj_matrix_cupy, mode, n_perms, multi_gpu):
        """Run spatial autocorrelation computation."""
        if mode == "moran":
            return _morans_I_cupy(
                data, adj_matrix_cupy, n_permutations=n_perms, multi_gpu=multi_gpu
            )
        elif mode == "geary":
            return _gearys_C_cupy(
                data, adj_matrix_cupy, n_permutations=n_perms, multi_gpu=multi_gpu
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

    data = _to_cupy(vals, use_sparse=use_sparse, dtype=compute_dtype)

    # Run full computation
    score, score_perms = _run_autocorr(data, adj_matrix_cupy, mode, n_perms, multi_gpu)

    # Set mode-specific params
    if mode == "moran":
        params["stat"] = "I"
        params["expected"] = -1.0 / (adata.shape[0] - 1)
        params["ascending"] = False
        params["mode"] = "moranI"
    else:  # geary
        params["stat"] = "C"
        params["expected"] = 1.0
        params["ascending"] = True
        params["mode"] = "gearyC"

    g = sparse.csr_matrix(adj_matrix_cupy.get())
    score = score.get()
    if n_perms is not None:
        score_perms = score_perms.get()
    with np.errstate(divide="ignore"):
        pval_results = _p_value_calc(score, sims=score_perms, weights=g, params=params)

    df = pd.DataFrame({params["stat"]: score, **pval_results}, index=genes)

    if corr_method is not None:
        for pv in filter(lambda x: "pval" in x, df.columns):
            _, pvals_adj, _, _ = multipletests(
                df[pv].values, alpha=0.05, method=corr_method
            )
            df[f"{pv}_{corr_method}"] = pvals_adj

    df.sort_values(by=params["stat"], ascending=params["ascending"], inplace=True)
    if copy:
        return df
    adata.uns[params["mode"]] = df
