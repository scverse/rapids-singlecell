---
name: rapids-singlecell
description: "Read before writing any rapids-singlecell code (`import rapids_singlecell as rsc`) or running scanpy-, squidpy-, decoupler- or pertpy-style analysis on a GPU. Covers QC, doublets, normalization, HVGs, PCA, integration, clustering, UMAP, markers, annotation, pseudobulk DE, pathway scoring, CRISPR screens, spatial statistics and niches, out-of-core or multi-GPU runs, and GPU setup. rsc differs from scanpy in where data live and in defaults."
---

# rapids-singlecell

`rsc.pp`, `rsc.tl` and `rsc.get` mirror scanpy on the GPU.
`rsc.gr`, `rsc.ptg` and `rsc.dcg` port parts of squidpy, pertpy and decoupler.
Compute with rapids-singlecell (rsc), plot with `sc.pl`/`sq.pl`/`dc.pl`, and leave rsc only for the gaps listed below.

## Before you start

Run once in a shell:

```bash
rapids-singlecell-check-kernel                        # GPU, RMM and CUDA kernels; stop if NOT READY
python -m rapids_singlecell_skills.api map --options  # every public rsc symbol with its enum choices
```

The map reflects the installed version.
Narrow tasks can map one namespace with `--namespace pp`.
Do not fetch the docs website or read rsc source to discover the API.
For one detail, run `python -m rapids_singlecell_skills.api describe rsc.pp.neighbors --parameter algorithm`.
If either command fails, read [references/setup.md](references/setup.md).

## Setup code

```python
import rmm

rmm.reinitialize(pool_allocator=True)  # before any GPU array, once per process
import cupy as cp
from rmm.allocators.cupy import rmm_cupy_allocator

cp.cuda.set_allocator(rmm_cupy_allocator)
import rapids_singlecell as rsc
import scanpy as sc
```

- The pool is the fast route when data fit in VRAM.
  Importing rsc alone installs RMM without a pool.
- If data do not fit, use `rmm.reinitialize(managed_memory=True)` instead.
  For Dask, multi-GPU or data far beyond VRAM, read [references/dask.md](references/dask.md).

## Where data live

- Only `X`, and layers you move yourself, live on the GPU.
  rsc writes `obs`, `var`, `obsm`, `obsp` and `uns` results to the host.
- `rsc.pp` QC, filter, normalization and HVG functions raise `_check_gpu_X` on host input.
  `rsc.get.anndata_to_GPU(adata)` moves `X` only, and `layer=` or `convert_all=True` also moves layers.
- Keep raw counts as a host layer, not `.raw` or a GPU copy.
  `rsc.get.aggregate` and `rsc.tl.rank_genes_groups` read host data directly.
- `sc.pl` works with a GPU `X` when coloring by `obs` or `obsm`.
  Call `rsc.get.anndata_to_CPU(adata)` once before plotting gene expression.
- Check `type(adata.X)` instead of converting defensively or hand-rolling converters.

## Standard workflow

Inspect first: shape, whether `X` holds integer counts, and which `obs` columns hold the biological sample, batch and condition.
The code's `sample` and `condition` stand for the dataset's own columns.
Convert dense counts to CSR (`scipy.sparse.csr_matrix`) before moving them to the GPU.
Cross-tabulate sample against condition, because a batch nested in condition cannot be corrected without erasing the condition.
For spatial data, read [references/spatial.md](references/spatial.md) first, because its QC and normalization replace QC, doublet scoring and normalization below.

```python
import pandas as pd
from scipy.stats import median_abs_deviation


def qc_outliers(obs: pd.DataFrame, by: str) -> pd.Series:
    def mads(col: str) -> pd.Series:
        g = obs.groupby(by, observed=True)[col]
        return (obs[col] - g.transform("median")) / g.transform(median_abs_deviation)

    return (
        (mads("log1p_total_counts").abs() > 5)
        | (mads("log1p_n_genes_by_counts").abs() > 5)
        | ((mads("pct_counts_mt") > 3) & (obs["pct_counts_mt"] > 8))
    )
```

```python
adata.layers["counts"] = adata.X.copy()  # host copy of raw counts
rsc.get.anndata_to_GPU(adata)
adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt"])
adata = adata[~qc_outliers(adata.obs, by="sample").to_numpy()].copy()
rsc.pp.scrublet(adata, batch_key="sample", verbose=False)  # raw counts, per sample
adata = adata[~adata.obs["predicted_doublet"].to_numpy()].copy()
rsc.pp.filter_genes(adata, min_cells=20)
rsc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
rsc.pp.normalize_total(adata)  # median depth
rsc.pp.log1p(adata)
rsc.pp.pca(adata, n_comps=50, mask_var="highly_variable")
rsc.pp.harmony_integrate(adata, key="sample")  # only if batches share biology
rsc.pp.neighbors(adata, use_rep="X_pca_harmony")  # "X_pca" without integration
rsc.tl.leiden(adata, resolution=[0.25, 0.5, 1.0, 2.0])  # one graph, keys leiden_<res>
rsc.tl.umap(adata)
rsc.tl.rank_genes_groups(adata, "leiden_1.0", method="wilcoxon", pts=True)
```

## Defaults that override old tutorials

- QC thresholds are per-sample MADs, shown on the distributions before filtering.
  Flag rather than drop high-mito cells in tumors or metabolically active tissue.
- Correct ambient RNA (CellBender, SoupX) only when contamination is evident, and keep the uncorrected counts.
- Score doublets per sample and remove them before clustering.
- Normalize to median depth, not `target_sum=1e4` or `1e6`.
  Do not `scale` (it densifies `X`) or `regress_out` before PCA.
- Select about 2000 `seurat_v3` HVGs on raw counts, where `batch_key` is optional.
- Integrate only when the uncorrected embedding separates a technical batch that shares biology across batches.
  Start with Harmony and move to scVI/scANVI for complex or cross-system batches.
- Verify integration kept biology with known markers or scib-metrics, not UMAP inspection.
- Leiden partitions change with the seed.
  Sweep resolutions with 3 to 5 `rng` values and keep resolutions whose labels agree across seeds (ARI) and whose clusters have distinct markers.
- GPU PCA on sparse input is not bitwise reproducible across reruns and can shift cluster boundaries.
  Save final labels in the AnnData rather than expecting a rerun to reproduce them.
- Marker p-values after clustering are only for ranking because the same data defined the clusters.
  Filter markers by `pts` detection fractions and effect size.
- Annotate clusters from positive and negative markers (score marker sets with `rsc.dcg.ulm` and compare cluster means) or reference transfer (CellTypist, scANVI).
  Use `unknown` for weak or tied evidence and treat LLM-proposed labels as hypotheses.
- UMAP is display only, so never infer distances, relatedness or trajectories from it.
- Comparing conditions needs biological replicates and pseudobulk, so read [references/conditions.md](references/conditions.md) first.
- Use foundation-model embeddings only on request and next to a PCA, Harmony or scVI baseline, which they do not beat in most cases.

## Fast paths

- `rsc.pp.neighbors` defaults to exact `brute`, which is fine up to about 500k cells.
  Above that use `algorithm="nn_descent"` (5x faster at 2M cells, recall 0.99), not `ivfflat` (recall 0.72 by default).
- Use `rank_genes_groups(method="wilcoxon_binned")` for Dask input or tens of millions of cells.
- Do not rerun steps on CPU to validate them or benchmark rsc against the CPU implementation unless asked.

## Outside rsc

| Need | Use |
|---|---|
| Ambient RNA | CellBender, SoupX |
| scVI, scANVI, sysVI | scvi-tools |
| Annotation | CellTypist, scANVI |
| Integration metrics | scib-metrics |
| Pseudobulk DE | PyDESeq2 on `rsc.get.aggregate` output |
| Pathway resources and plots | decoupler `dc.op`, `dc.pl`, scoring with `rsc.dcg` |
| Cell-cell communication across samples | LIANA+ |
| Neighborhood enrichment | squidpy |
| Condition & perturbation analysis | pertpy |

Never replace a missing rsc capability with a silent CPU reimplementation.
For CRISPR screens and perturbation distances, read [references/perturbation.md](references/perturbation.md).

## Deliverables

Use a notebook for multi-step workflows or when asked.
Otherwise a small script artifact is fine.

- Check outputs against data scale.
  Hundreds of clusters, clusters near cell count, or figures of hundreds of megapixels mean stop and fix.
- Follow requested choices unless the data or design make them invalid.
  In that case, show the evidence and propose an alternative.

In a notebook:

- One stage per cell, under about 25 lines.
  Plot right after the computation it shows, then interpret in a short markdown cell.
- Never paste standalone scripts into notebook cells.
- Record package versions at the end of the notebook with `session_info2`.
- While developing, test new stages against a checkpoint AnnData saved after preprocessing instead of re-executing the whole notebook.
- Execute top to bottom in a fresh kernel with `jupyter nbconvert --to notebook --execute --inplace` and read every output.
- Deliver the executed notebook, the final AnnData, and a short markdown report of findings, evidence and limitations.
