from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import os
from pathlib import Path

import anndata as ad
from utils.sepal_cpu import sepal
from utils.sepal_gpu import sepal_gpu

import rapids_singlecell as rsc

HOME = Path(os.path.expanduser("~"))


def debug_sepal_comparison():
    # Load data
    adata = ad.read_h5ad(HOME / "data/visium_hne_adata.h5ad")

    # Test with single gene first
    test_genes = ["Npbwr1"]  # Just one gene

    print("=== CPU vs GPU Comparison ===")

    # CPU version
    cpu_result = sepal(
        adata, max_neighs=6, genes=test_genes, n_iter=4000, copy=True, debug=True
    )
    print(f"CPU result: {cpu_result.iloc[0, 0]}")

    # GPU version
    adata = ad.read_h5ad(HOME / "data/visium_hne_adata.h5ad")
    rsc.get.anndata_to_GPU(adata, convert_all=True)
    adata.obsp["spatial_connectivities"] = rsc.get.X_to_GPU(
        adata.obsp["spatial_connectivities"]
    )
    adata.obsm["spatial"] = rsc.get.X_to_GPU(adata.obsm["spatial"])

    gpu_result = sepal_gpu(
        adata, max_neighs=6, genes=test_genes, n_iter=4000, copy=True, debug=True
    )
    print(f"GPU result: {gpu_result.iloc[0, 0]}")

    print(f"Difference: {abs(cpu_result.iloc[0, 0] - gpu_result.iloc[0, 0])}")


if __name__ == "__main__":
    debug_sepal_comparison()
