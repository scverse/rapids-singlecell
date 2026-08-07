from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cupy as cp
import pandas as pd
from anndata import AnnData
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse import isspmatrix_csc as cp_isspmatrix_csc
from cupyx.scipy.sparse import isspmatrix_csr as cp_isspmatrix_csr
from scanpy import logging as logg

from .kernels._sepal import (
    _get_get_nhood_idx_with_distance,
    _get_sepal_simulation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def sepal(
    adata: AnnData,
    max_neighs: Literal[4, 6],
    genes: str | Sequence[str] | None = None,
    n_iter: int = 30000,
    dt: float = 0.001,
    thresh: float = 1e-8,
    connectivity_key: str = "spatial_connectivities",
    spatial_key: str = "spatial",
    layer: str | None = None,
    copy: bool = False,
) -> pd.DataFrame | None:
    """
    GPU-accelerated sepal implementation with unlimited scalability.
    Handles datasets from thousands to millions of cells.

    Grid/block configuration follows established patterns:
    - threads_per_block = 256 (as in src/rapids_singlecell/preprocessing/_harmony/_helper.py)
    - 1D grid sizing with ceil division (as in src/rapids_singlecell/preprocessing/_harmony/_helper.py)
    - Shared memory allocation for entropy computation (similar to co-occurrence kernels)
    """
    # won't support SpatialData to avoid dependencies on spatialdata
    assert isinstance(adata, AnnData), "adata must be an AnnData object"

    # _assert_connectivity_key(adata, connectivity_key)
    assert connectivity_key in adata.obsp, (
        f"Connectivity key {connectivity_key} not found in adata.obsp"
    )
    # _assert_spatial_basis(adata, key=spatial_key) replacement
    assert spatial_key in adata.obsm, (
        f"Spatial key {spatial_key} not found in adata.obsm"
    )

    if max_neighs not in (4, 6):
        raise ValueError(
            f"Expected `max_neighs` to be either `4` or `6`, found `{max_neighs}`."
        )

    # Setup spatial coordinates as float32 (standard for spatial data)
    spatial = cp.asarray(adata.obsm[spatial_key], dtype=cp.float32)

    # replacement for _assert_non_empty_sequence
    if genes is None:
        genes = adata.var_names.values
        if "highly_variable" in adata.var.columns:
            genes = genes[adata.var["highly_variable"].values]
    if len(genes) == 0:
        raise ValueError("No genes found")

    # Graph and index computation
    g = adata.obsp[connectivity_key]
    if not cp_isspmatrix_csr(g):
        g = cp_csr_matrix(g)
    g.eliminate_zeros()

    degrees = cp.diff(g.indptr)
    max_n = degrees.max()
    if max_n != max_neighs:
        raise ValueError(
            f"Expected `max_neighs={max_neighs}`, found node with `{max_n}` neighbors."
        )

    sat, sat_idx, unsat, unsat_to_nearest_sat = _compute_idxs(
        g=g,
        degrees=degrees,
        spatial=spatial,
        sat_thresh=max_neighs,
    )

    # replacement for _extract_expression
    if layer is None:
        vals = adata[:, genes].X
    elif layer not in adata.layers:
        raise KeyError(f"Layer `{layer}` not found in `adata.layers`.")
    else:
        vals = adata[:, genes].layers[layer]
        if isinstance(vals, AnnData):
            vals = vals.X
    start = logg.info(
        f"Calculating sepal score for `{len(genes)}` genes using scalable GPU kernel"
    )

    if cp_isspmatrix_csr(vals) or cp_isspmatrix_csc(vals):
        vals = vals.toarray()

    # Use double precision for numerical stability in simulation
    vals = cp.ascontiguousarray(cp.asarray(vals, dtype=cp.float64))

    # Run scalable simulation - handles ANY dataset size!
    scores = _cuda_kernel_diffusion_gpu(
        vals=vals,
        sat=sat,
        sat_idx=sat_idx,
        unsat=unsat,
        unsat_to_nearest_sat=unsat_to_nearest_sat,
        max_neighs=max_neighs,
        n_iter=n_iter,
        dt=dt,
        thresh=thresh,
    )

    # Results processing
    score = cp.asnumpy(scores)

    key_added = "sepal_score"
    sepal_score = pd.DataFrame(score, index=genes, columns=[key_added])

    if sepal_score[key_added].isna().any():
        logg.warning(
            "Found `NaN` in sepal scores, consider increasing `n_iter` to a higher value"
        )
    sepal_score = sepal_score.sort_values(by=key_added, ascending=False)

    if copy:
        logg.info("Finish", time=start)
        return sepal_score

    # replacement for _save_data
    if not copy:
        adata.uns[key_added] = sepal_score
    return sepal_score


def _cuda_kernel_diffusion_gpu(
    vals: cp.ndarray,  # (n_cells, n_genes) - all gene expressions
    sat: cp.ndarray,  # (n_sat,) - saturated node indices
    sat_idx: cp.ndarray,  # (n_sat, max_neighs) - neighborhood indices for sat nodes
    unsat: cp.ndarray,  # (n_unsat,) - unsaturated node indices
    unsat_to_nearest_sat: cp.ndarray,  # (n_unsat,) - nearest sat for each unsat
    max_neighs: int,
    n_iter: int,
    dt: float,
    thresh: float,
) -> cp.ndarray:
    n_cells, n_genes = vals.shape
    n_sat = len(sat)
    n_unsat = len(unsat)

    # Grid/block configuration following established patterns:
    # threads_per_block = 256 (as in src/rapids_singlecell/preprocessing/_harmony/_helper.py)
    threads_per_block = 256
    blocks_per_grid = n_genes  # Process ALL genes in parallel!

    # Allocate arrays for ALL genes at once
    concentration_all = cp.ascontiguousarray(
        vals.T, dtype=cp.float64
    )  # (n_genes, n_cells)
    derivatives_all = cp.zeros((n_genes, n_cells), dtype=cp.float64)
    results_all = cp.full(n_genes, -999999.0, dtype=cp.float64)  # Results for ALL genes

    # Calculate shared memory (fixed size per block, independent of n_cells)
    min_blocks = 256  # Hardware-specific minimum
    blocks_per_grid = max(n_genes, min_blocks)
    shared_mem_size = threads_per_block * 2 * 8  # 2 double arrays per thread

    # Get specialized kernel using cuda_kernel_factory pattern
    sepal_simulation_kernel = _get_sepal_simulation(derivatives_all.dtype)

    # **SINGLE KERNEL LAUNCH FOR ALL GENES**
    sepal_simulation_kernel(
        (blocks_per_grid,),  # Grid: one block per gene
        (threads_per_block,),  # Block: 256 threads
        (
            concentration_all,  # (n_genes, n_cells) - all genes
            derivatives_all,  # (n_genes, n_cells) - all derivatives
            sat,
            sat_idx,
            unsat,
            unsat_to_nearest_sat,
            results_all,  # (n_genes,) - results for all genes
            n_cells,  # n_cells (can be 1M+)
            n_genes,  # Number of genes to process
            n_sat,
            n_unsat,
            max_neighs,
            n_iter,
            cp.float64(dt),
            cp.float64(thresh),
        ),
        shared_mem=shared_mem_size,
    )

    # Convert results
    final_scores = cp.where(results_all < 0.0, cp.nan, dt * results_all)

    return final_scores  # Shape: (n_genes,)


def _compute_idxs(
    g: cp_csr_matrix,
    degrees: cp.ndarray,
    spatial: cp.ndarray,
    sat_thresh: int,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Compute saturated/unsaturated indices on GPU with unified distance computation.

    Grid/block configuration follows established patterns:
    - threads_per_block = 256 (as in src/rapids_singlecell/preprocessing/_harmony/_helper.py)
    - 1D grid sizing with ceil division (as in src/rapids_singlecell/preprocessing/_harmony/_helper.py)
    """

    # Get saturated and unsaturated nodes
    unsat_mask = degrees < sat_thresh
    sat_mask = degrees == sat_thresh

    unsat = cp.asarray(cp.where(unsat_mask)[0], dtype=cp.int32)
    sat = cp.asarray(cp.where(sat_mask)[0], dtype=cp.int32)

    # Extract saturated neighborhoods with vectorized CuPy
    nearest_sat = cp.full(len(unsat), -1, dtype=cp.int32)
    sat_idx = g.indices[g.indptr[sat][:, None] + cp.arange(sat_thresh)]

    # Single kernel handles both graph neighbors and distance fallback
    if len(unsat) > 0:
        # Grid/block configuration following established patterns:
        # threads_per_block = 256 (as in src/rapids_singlecell/preprocessing/_harmony/_helper.py)
        threads_per_block = 256
        blocks = (len(unsat) + threads_per_block - 1) // threads_per_block

        # Get specialized kernel using cuda_kernel_factory pattern
        get_nhood_kernel = _get_get_nhood_idx_with_distance(spatial.dtype)

        get_nhood_kernel(
            (blocks,),
            (threads_per_block,),
            (
                unsat,  # unsaturated nodes (read only int32)
                spatial,  # spatial coordinates [n_nodes, 2] (read only float64)
                sat,  # saturated node list (read only int32)
                g.indptr,  # CSR indptr (read only int32)
                g.indices,  # CSR indices (read only int32)
                sat_mask,  # boolean mask for saturated nodes (read only bool)
                nearest_sat,  # output int32
                len(unsat),  # number of unsaturated nodes read only int32
                len(sat),  # number of saturated nodes read only int32
            ),
        )

    return sat, sat_idx, unsat, nearest_sat
