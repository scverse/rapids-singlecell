# Perturbation analysis

Read this file for CRISPR screens, perturbation signatures, or distance-based perturbation comparisons.

## Order the workflow

- Assign guides with `rsc.ptg.GuideAssignment`, compute the signature with `perturbation_signature`, then run `mixscape` or `mixscale`; `lda` requires `mixscape` first. Each stage raises on a missing predecessor, so confirm `adata.layers["X_pert"]` exists before classifying.
- Run `perturbation_signature` on unscaled log-normalized data. Scaling beforehand corrupts the residual silently — no error is raised — so verify what `X` holds instead of reusing a scaled preprocessing object.
- Set `split_by` to the biological replicate so reference cells come from comparable samples, and keep inference tied to that unit.
- RSC caps `n_neighbors` to the controls available in each split where pertpy raises instead. Note the divergence whenever results are compared against pertpy.

## Respect the pertpy boundary

- `rsc.ptg` ports four APIs: `Distance`, `GuideAssignment`, `Mixscape`, and `Mixscale`. `MeanVar` is deprecated; do not use it in new analyses.
- The rest of pertpy has no RSC equivalent — Augur, Milo, Sccoda, Scgen, Cinemaot, Dialogue, the embedding-space classes, the differential-expression wrappers, and the `Distance` significance tests. Compose these under the pertpy skill at an attributed boundary rather than reimplementing them on GPU.
- `Distance` is the only public entry to its metrics: select one with `metric=` and pass metric-specific options through as keyword arguments. Describe the live signature for the supported set rather than assuming a metric name. Every entry point takes `multi_gpu` to fan out across devices.
- Choose the entry point from the question: `pairwise` for all-versus-all, `onesided_distances` for every group against one reference, and `contrast_distances` for an explicit list built with `create_contrasts`, whose `split_by` stratifies each perturbation-versus-control comparison within a cell type, batch, or timepoint. That stratified shape is the usual screen design and the one the other two entry points cannot express.
- Report effect sizes with `bootstrap` uncertainty on `pairwise` or `onesided_distances`. `contrast_distances` takes no `bootstrap` argument, so resample explicitly when a stratified contrast needs an interval. When replicates exist, permute over them rather than over cells; a cell-level permutation treats a single sample's cells as independent and overstates significance.

## Contrast a screen

Two steps, and the call sites differ: `create_contrasts` is a staticmethod on the class, while `contrast_distances` needs a configured instance.

```python
contrasts = rsc.ptg.Distance.create_contrasts(   # staticmethod: call on the class
    adata,
    groupby="target_gene",
    selected_group="Non_target",   # a sequence compares against several references
    split_by="cell_type",          # one contrast per perturbation *within* each cell type
)
contrasts = contrasts[contrasts["target_gene"].isin(hits)]   # filter before computing
result = rsc.ptg.Distance(metric="edistance").contrast_distances(adata, contrasts)
```

`create_contrasts` returns a plain DataFrame — one row per contrast, with the reference in a `reference` column — so inspect and subset it before computing rather than discarding rows afterwards. Combinations whose reference is absent from a split are dropped for you.
