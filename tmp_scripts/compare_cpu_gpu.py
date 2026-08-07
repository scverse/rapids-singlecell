from __future__ import annotations

import time
import warnings
from pathlib import Path

import anndata as ad
import pandas as pd
from scipy.stats import spearmanr
from squidpy.gr import sepal as sepal_cpu

import rapids_singlecell as rsc
from rapids_singlecell.squidpy_gpu import sepal as sepal_gpu

warnings.filterwarnings("ignore")

HOME = Path.home()


def main():
    # Load data
    print("Loading data...")
    adata = ad.read_h5ad(HOME / "data/visium_hne_adata.h5ad")

    # Test on first 10 genes
    n_genes = 15
    genes = adata.var_names.values[:n_genes].tolist()

    print(f"\nTesting on first {n_genes} genes:")
    print(", ".join(genes))

    # Run CPU version
    print("\n" + "=" * 80)
    print("Running CPU version...")
    print("=" * 80)
    adata_cpu = adata.copy()
    start_time = time.time()
    result_cpu = sepal_cpu(
        adata_cpu, max_neighs=6, genes=genes, n_iter=30000, copy=True
    )
    cpu_time = time.time() - start_time
    print(f"CPU Time: {cpu_time:.2f} seconds")

    # Run GPU version
    print("\n" + "=" * 80)
    print("Running GPU version...")
    print("=" * 80)
    adata_gpu = adata.copy()
    rsc.get.anndata_to_GPU(adata_gpu, convert_all=True)
    adata_gpu.obsp["spatial_connectivities"] = rsc.get.X_to_GPU(
        adata_gpu.obsp["spatial_connectivities"]
    )
    adata_gpu.obsm["spatial"] = rsc.get.X_to_GPU(adata_gpu.obsm["spatial"])

    start_time = time.time()
    result_gpu = sepal_gpu(
        adata_gpu, max_neighs=6, genes=genes, n_iter=30000, copy=True
    )
    gpu_time = time.time() - start_time
    print(f"GPU Time: {gpu_time:.2f} seconds")

    # Prepare comparison
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    # Merge results and calculate ranks
    comparison = pd.DataFrame(
        {
            "Gene": result_cpu.index,
            "CPU_Score": result_cpu["sepal_score"].values,
            "GPU_Score": result_gpu["sepal_score"].values,
        }
    )

    # Calculate ranks (1 = highest score)
    comparison["CPU_Rank"] = (
        comparison["CPU_Score"].rank(ascending=False, method="min").astype(int)
    )
    comparison["GPU_Rank"] = (
        comparison["GPU_Score"].rank(ascending=False, method="min").astype(int)
    )
    comparison["Rank_Diff"] = abs(comparison["CPU_Rank"] - comparison["GPU_Rank"])

    # Calculate absolute and relative differences in scores
    comparison["Score_Diff"] = abs(comparison["CPU_Score"] - comparison["GPU_Score"])
    comparison["Rel_Diff_%"] = (
        comparison["Score_Diff"] / comparison["CPU_Score"].abs()
    ) * 100

    # Calculate correlations
    # Spearman correlation (rank-based)
    spearman_corr, spearman_pval = spearmanr(
        comparison["CPU_Score"], comparison["GPU_Score"]
    )

    # Also calculate Spearman on explicit ranks for clarity
    spearman_rank_corr, spearman_rank_pval = spearmanr(
        comparison["CPU_Rank"], comparison["GPU_Rank"]
    )

    # Pearson correlation (on scores)
    from scipy.stats import pearsonr

    pearson_corr, pearson_pval = pearsonr(
        comparison["CPU_Score"], comparison["GPU_Score"]
    )

    # Display overall metrics
    print(f"\n{'CORRELATION METRICS':^80}")
    print("=" * 80)
    print(
        f"Spearman Rank Correlation: {spearman_corr:.6f} (p-value: {spearman_pval:.2e})"
    )
    print(
        f"  (on explicit ranks):     {spearman_rank_corr:.6f} (p-value: {spearman_rank_pval:.2e})"
    )
    print(
        f"Pearson Correlation:       {pearson_corr:.6f} (p-value: {pearson_pval:.2e})"
    )
    print(f"\nSpeedup: {cpu_time / gpu_time:.2f}x")

    # Side-by-side comparison with ranks
    print("\n" + "=" * 80)
    print(f"{'SIDE-BY-SIDE COMPARISON (Sorted by CPU Rank)':^80}")
    print("=" * 80)

    # Format the table for display
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_rows", None)

    # Create display dataframe
    display_df = comparison[
        [
            "Gene",
            "CPU_Score",
            "CPU_Rank",
            "GPU_Score",
            "GPU_Rank",
            "Rank_Diff",
            "Score_Diff",
            "Rel_Diff_%",
        ]
    ].copy()
    display_df = display_df.sort_values("CPU_Rank")

    # Format for better readability
    print(
        display_df.to_string(
            index=False,
            formatters={
                "CPU_Score": "{:.3f}".format,
                "GPU_Score": "{:.3f}".format,
                "Score_Diff": "{:.3f}".format,
                "Rel_Diff_%": "{:.2f}".format,
            },
        )
    )

    # Summary statistics
    print("\n" + "=" * 80)
    print(f"{'SUMMARY STATISTICS':^80}")
    print("=" * 80)

    print("\nScore Differences:")
    print(f"  Mean Absolute Difference: {comparison['Score_Diff'].mean():.6f}")
    print(f"  Max Absolute Difference:  {comparison['Score_Diff'].max():.6f}")
    print(f"  Mean Relative Difference: {comparison['Rel_Diff_%'].mean():.2f}%")
    print(f"  Max Relative Difference:  {comparison['Rel_Diff_%'].max():.2f}%")

    print("\nRank Differences:")
    print(f"  Mean Rank Difference:     {comparison['Rank_Diff'].mean():.2f}")
    print(f"  Max Rank Difference:      {comparison['Rank_Diff'].max()}")
    print(
        f"  Perfect Rank Matches:     {(comparison['Rank_Diff'] == 0).sum()}/{len(comparison)}"
    )
    print(
        f"  Within 1 Rank:            {(comparison['Rank_Diff'] <= 1).sum()}/{len(comparison)}"
    )
    print(
        f"  Within 2 Ranks:           {(comparison['Rank_Diff'] <= 2).sum()}/{len(comparison)}"
    )

    # Top genes comparison
    print("\n" + "=" * 80)
    print(f"{'TOP 5 GENES COMPARISON':^80}")
    print("=" * 80)

    cpu_top5 = comparison.nsmallest(5, "CPU_Rank")[
        ["Gene", "CPU_Score", "CPU_Rank"]
    ].reset_index(drop=True)
    gpu_top5 = comparison.nsmallest(5, "GPU_Rank")[
        ["Gene", "GPU_Score", "GPU_Rank"]
    ].reset_index(drop=True)

    print("\nTop 5 by CPU:")
    print(cpu_top5.to_string(index=False, formatters={"CPU_Score": "{:.3f}".format}))

    print("\nTop 5 by GPU:")
    print(gpu_top5.to_string(index=False, formatters={"GPU_Score": "{:.3f}".format}))

    # Check overlap in top genes
    cpu_top5_genes = set(cpu_top5["Gene"])
    gpu_top5_genes = set(gpu_top5["Gene"])
    overlap = cpu_top5_genes & gpu_top5_genes
    print(f"\nTop 5 Overlap: {len(overlap)}/5 genes")
    if overlap:
        print(f"Common genes: {', '.join(sorted(overlap))}")

    # Save results
    output_file = HOME / "rapids_singlecell/tmp_scripts/comparison_results.csv"
    comparison_sorted = comparison.sort_values("CPU_Rank")
    comparison_sorted.to_csv(output_file, index=False)
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
