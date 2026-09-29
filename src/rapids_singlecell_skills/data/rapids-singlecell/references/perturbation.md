# Perturbation analysis

`rsc.ptg` ports pertpy's `Distance`, `GuideAssignment`, `Mixscape` and `Mixscale`; use pertpy itself for everything else (Augur, Milo, scCODA, E-test, DE wrappers, embedding-space methods).

## Guides and perturbation signatures

- Compare at least two `rsc.ptg.GuideAssignment` strategies (threshold and mixture model) and report unassigned and multi-guide fractions; choose by on-target knockdown against non-targeting controls.
- Order: `GuideAssignment`, then `perturbation_signature`, then `mixscape` or `mixscale`; `lda` requires `mixscape`, and each stage raises when its predecessor is missing.
- Run `perturbation_signature` on unscaled log-normalized data; scaled input corrupts the residual silently.
- Set `split_by` to the biological replicate so control neighbors come from the same sample.
- rsc caps `n_neighbors` at the controls available per split where pertpy raises; note this when comparing with pertpy.
- Prefer `Mixscale` for CRISPRi and other graded knockdowns; `Mixscape` forces a binary KO/NP call.

## Effects and distances

- Rank perturbations by a transcriptome-wide distance to controls, not by DEG counts; most perturbations shift only a few dozen genes.
- Keep only perturbations that differ from controls (pertpy E-test) before embedding or clustering perturbations.
- Compare signatures against the average perturbed cell as well as against controls; programs shared by most perturbations (stress, cell cycle) are systematic, not perturbation-specific.
- For DE, pseudobulk per perturbation and replicate against non-targeting controls, and check calibration with non-targeting-versus-non-targeting tests.
- Any prediction or foundation-model claim needs mean, additive and linear baselines next to it.

## Distance API

- `Distance(metric=...)` is the only entry to the metrics; describe it for the supported set and pass metric options as keyword arguments.
- `pairwise` compares all groups, `onesided_distances` compares every group with one reference, and `contrast_distances` runs an explicit list from `create_contrasts`.
- `pairwise`, `onesided_distances` and `contrast_distances` take `multi_gpu`.
- `bootstrap=True` gives uncertainty on `pairwise` and `onesided_distances`; `contrast_distances` has no `bootstrap`, so resample yourself, over replicates rather than cells.

```python
contrasts = rsc.ptg.Distance.create_contrasts(  # staticmethod: call on the class
    adata,
    groupby="target_gene",
    selected_group="Non_target",
    split_by="cell_type",  # one contrast per perturbation within each cell type
)
contrasts = contrasts[contrasts["target_gene"].isin(hits)]  # filter before computing
result = rsc.ptg.Distance(metric="edistance").contrast_distances(adata, contrasts)
```

`create_contrasts` returns one row per contrast with the reference in a `reference` column, and drops splits that lack the reference.
