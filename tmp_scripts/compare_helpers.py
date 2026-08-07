from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import os
import time
from pathlib import Path

import anndata as ad
import cupy as cp
import numpy as np

import rapids_singlecell as rsc

# Add utils to path for GPU version

HOME = Path(os.path.expanduser("~"))


def compare_indices(adata_cpu):
    """
    Compare the saturated/unsaturated indices between CPU and GPU versions.
    """
    print("🔍 Comparing CPU vs GPU index computation...")

    # Import the helper functions
    from utils._sepal import _compute_idxs as _compute_idxs_cpu

    from rapids_singlecell.squidpy_gpu._sepal import _compute_idxs as _compute_idxs_gpu

    # Get connectivity and spatial data
    adata_gpu = ad.read_h5ad(HOME / "data/visium_hne_adata.h5ad")
    rsc.get.anndata_to_GPU(adata_gpu, convert_all=True)
    g = adata_cpu.obsp["spatial_connectivities"]
    g_gpu = adata_gpu.obsp["spatial_connectivities"]
    degrees = cp.diff(g_gpu.indptr)
    spatial_cpu = adata_cpu.obsm["spatial"].astype(np.float64)
    spatial_gpu = adata_gpu.obsm["spatial"].astype(cp.float32)

    g_gpu = rsc.get.X_to_GPU(g_gpu)
    spatial_gpu = rsc.get.X_to_GPU(spatial_gpu)

    # Compute indices with both methods
    start = time.time()
    degrees = cp.diff(g_gpu.indptr)
    sat_gpu, sat_idx_gpu, unsat_gpu, unsat_idx_gpu = _compute_idxs_gpu(
        g_gpu, degrees, spatial_gpu, 6
    )
    end = time.time()
    print("GPU indices computed in ", end - start, "seconds")

    start = time.time()
    sat_cpu, sat_idx_cpu, unsat_cpu, unsat_idx_cpu = _compute_idxs_cpu(
        g, spatial_cpu, 6, "l1"
    )
    end = time.time()
    print("CPU indices computed in ", end - start, "seconds")
    # Convert GPU results to CPU for comparison
    sat_gpu_cpu = sat_gpu.get()
    sat_idx_gpu_cpu = sat_idx_gpu.get()
    unsat_gpu_cpu = unsat_gpu.get()
    unsat_idx_gpu_cpu = unsat_idx_gpu.get()

    print(f"Saturated nodes - CPU: {len(sat_cpu)}, GPU: {len(sat_gpu_cpu)}")
    print(f"Saturated nodes identical: {np.array_equal(sat_cpu, sat_gpu_cpu)}")

    print(f"Unsaturated nodes - CPU: {len(unsat_cpu)}, GPU: {len(unsat_gpu_cpu)}")
    print(f"Unsaturated nodes identical: {np.array_equal(unsat_cpu, unsat_gpu_cpu)}")

    print(
        f"Saturated indices identical: {np.array_equal(sat_idx_cpu, sat_idx_gpu_cpu)}"
    )

    # Check unsat_idx differences (these might differ due to tie-breaking)
    unsat_idx_diff = np.sum(unsat_idx_cpu != unsat_idx_gpu_cpu)
    print(
        f"Unsaturated index differences: {unsat_idx_diff}/{len(unsat_idx_cpu)} ({100 * unsat_idx_diff / len(unsat_idx_cpu):.1f}%)"
    )

    return {
        "sat_identical": np.array_equal(sat_cpu, sat_gpu_cpu),
        "unsat_identical": np.array_equal(unsat_cpu, unsat_gpu_cpu),
        "sat_idx_identical": np.array_equal(sat_idx_cpu, sat_idx_gpu_cpu),
        "unsat_idx_diff_count": unsat_idx_diff,
        "unsat_idx_diff_percent": 100 * unsat_idx_diff / len(unsat_idx_cpu),
    }


if __name__ == "__main__":
    # Run comparison
    adata_cpu = ad.read_h5ad(HOME / "data/visium_hne_adata.h5ad")
    res = compare_indices(adata_cpu)
    print(res)
