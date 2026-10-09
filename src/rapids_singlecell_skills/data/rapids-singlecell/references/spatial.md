# Spatial analysis

## Load imaging output (Xenium, Atera)

```python
import spatialdata_io as sio

sdata = sio.xenium(
    outs_dir,
    cells_table=True,
    cells_boundaries=True,
    morphology_focus=True,  # only for image overlays
    transcripts=False,
    cells_labels=False,
    nucleus_labels=False,
    nucleus_boundaries=False,
    morphology_mip=False,
    aligned_images=False,
)
adata = sdata.tables["table"].copy()
```

- Transcripts and label images dominate load time and memory, so load them only when the analysis uses them.
- `obsm["spatial"]` is in microns, while images and the `global` coordinate system are in pixels.
  Convert with `spatialdata.transformations.get_transformation` before cropping.
- To color boundaries by table columns with spatialdata-plot, set `table.obs["region"] = pd.Categorical(["cell_boundaries"] * table.n_obs)` and then call `sdata.set_table_annotates_spatialelement("table", region="cell_boundaries", region_key="region", instance_key="cell_id")`.

## Segmentation QC

- Inspect `cell_area`, `nucleus_area`, `nucleus_count`, the control probe and codeword counts, and `segmentation_method`, both as histograms and in space.
- A spatial cluster of failures is a staining or tissue problem, not biology.
- Filter permissively (no nucleus, high control fraction, the extreme low tail of counts for the panel size) and plot removed cells in space.
  Per-sample MADs on counts can delete whole low-count regions.
- Skip scrublet, because droplet doublet models do not describe segmentation errors.
- Before DE or ligand-receptor analysis, look for cells co-expressing mutually exclusive lineage markers.
  DE that tracks neighboring cell types is a suspected segmentation artifact.

## Normalize

- Targeted panels of up to a few thousand genes: divide counts by cell area scaled to the median area, then `rsc.pp.log1p`.
  Total counts there reflect panel composition and tissue region.
- Whole-transcriptome panels such as Atera: `rsc.pp.normalize_total` is acceptable.
  Check that total counts do not simply track a tissue region.

```python
import cupyx.scipy.sparse as cpsp

area = adata.obs["cell_area"].to_numpy()
adata.X = cpsp.diags(cp.asarray(np.median(area) / area, dtype=adata.X.dtype)) @ adata.X
rsc.pp.log1p(adata)
```

## Physical graph

```python
rsc.gr.spatial_neighbors_delaunay(adata, library_key="sample", percentile=99)
```

- `rsc.gr.spatial_neighbors_knn` and, for Visium spot grids, `rsc.gr.spatial_neighbors_grid` take the same `library_key`.
- Build one graph per sample (`library_key`), prune long Delaunay edges with `percentile`, and check the degree distribution and isolated cells.
- The physical graph lives in `obsp["spatial_connectivities"]`, separate from the expression graph in `obsp["connectivities"]`.
  `rsc.gr` functions move it to the GPU themselves.

## Niches and domains

```python
niche = ad.AnnData(X=adata.obsm["X_pca"], obs=adata.obs[[]])
niche.obsp["spatial_connectivities"] = adata.obsp["spatial_connectivities"]
labels = {}
for k in range(5, 16):
    for seed in range(3):
        rsc.gr.calculate_niche_cellcharter(niche, n_components=k, random_state=seed)
        labels[k, seed] = niche.obs["cellcharter_niche"].copy()
```

- `calculate_niche_cellcharter` aggregates `X` over neighbor shells.
  Aggregating a PCA copy takes about a second at 170k cells, while the full-expression path can exceed cuSPARSE limits on whole-transcriptome panels.
- `use_rep` skips the spatial aggregation and only clusters the given embedding, so it does not find niches.
- The per-sample graph already keeps aggregation within samples.
  `library_key` would fit separate, non-comparable niches per sample.
- Pick `n_components` where labels agree across seeds (ARI), not a fixed guess.
- Compare niches with non-spatial Leiden clusters, because niches that only reproduce cell types add nothing.
- Characterize niches by cell-type composition and pathway scores (`rsc.dcg.ulm` with `dc.op.progeny` or `dc.op.hallmark`) rather than hand-picked five-gene programs.

## Spatial statistics

- `rsc.gr.spatial_autocorr(adata, mode="moran", copy=True)` scores HVGs by default and `genes=` any others.
  Rank genes by Moran's I, not p-values, and interpret within cell type at single-cell resolution.
- `rsc.gr.sepal(adata, max_neighs=6)` scores spatially variable genes on Visium hexagonal (`6`) or square-grid (`4`) spot or bin graphs from `rsc.gr.spatial_neighbors_grid`; single-cell graphs are not regular grids.
- `rsc.gr.co_occurrence(adata, cluster_key="cell_type", interval=...)` runs on the GPU.
- `rsc.gr.ripley(adata, cluster_key="cell_type", mode="L")` computes Ripley's F, G or L on the GPU; simulations differ from squidpy for the same `rng`.
  Neighborhood enrichment has no rsc port, so use `sq.gr.nhood_enrichment` with `backend="threading"` when `n_jobs > 1`.
- `rsc.gr.nhood_enrichment(adata, cluster_key="cell_type", library_key="sample")` runs all permutations on the GPU.
  `library_key` permutes labels only within each sample; cells without a label are dropped with their edges.
- Read enrichment between cell types as co-compartmentalization and summarize it per sample.
- `rsc.gr.ligrec` reads host `X` or `.raw`, so run it after `rsc.get.anndata_to_CPU`.
- The default OmniPath network contains intracellular pairs (for example KIT to PIK3R1).
  Keep secreted or membrane ligands with surface receptors, or pass a curated table through `interactions=`.
- Ligand-receptor hits are hypotheses that need proximity and downstream-target support.
  Compare conditions with LIANA+ at the sample level.

## Plots

- Use `sq.pl.spatial_scatter(adata, shape=None, ...)` for cell-level maps and `spatialdata-plot` with `sdata.query.bounding_box` for image and boundary crops.
- Load a cropped image into memory before rendering several panels from it, because each render otherwise re-reads the full-resolution TIFF.
- Keep `figsize` x `dpi` bounded (100 to 150 dpi), set `rasterized=True` where supported, and never downsample the cells being analyzed.
  If render time or file size explodes, simplify and rerender.
- h5ad cannot store `/` in category names that become keys, or tuples in `uns`, so rename or convert them before writing.
