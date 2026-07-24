# Spatial analysis

Read this file for spatial graphs, niches, or large spatial plots.

## Build the physical graph

- When live RSC cannot construct the physical spatial graph, use Squidpy at an
  attributed host-side boundary. Build each sample or library independently, retain
  coordinate units and parameters, and prevent cross-sample edges.
- Keep physical and expression-derived graphs under distinct keys. Verify observation
  alignment, degree and component summaries, isolated cells, and edge distances before
  downstream RSC computation.

## Evaluate niches before interpretation

- `calculate_niche(flavor="neighborhood")` clusters cell-type-composition profiles
  with Leiden. Inspect the full niche-size distribution, `not_a_niche` fraction,
  spatial coherence, and stability across plausible graph scales, `distance`,
  `n_neighbors`, and `resolutions`; lower resolutions coarsen Leiden.
- `min_niche_size` only relabels small communities as `not_a_niche`; it does not
  merge or refit them. Fixed-k clustering is a separate method with no supported RSC
  route—`rsc.tl.kmeans` is deprecated. If fixed k is essential, document the
  composition representation, justify k and stability, and use an attributed,
  live-verified ecosystem method.
- Treat `flavor="cellcharter"` as complementary expression/PCA/GMM sensitivity, not
  validation of composition niches; `gmm_init="kmeans"` only initializes its GMM.
- Never remove `unknown` solely by name or encode it as missing. Filter independently
  validated artifacts and rebuild the graph; otherwise retain uncertain cells and
  report label-treatment sensitivity and coverage.

## Bound large spatial plots

- Estimate point and panel counts plus pixels from `figsize × dpi`. Bound figure size
  and DPI, use `rasterized=True` when supported, and downsample only contextual or grey
  background points—not analytical data.
- Treat roughly 100–150 DPI as an exploratory heuristic. If render time, pixel count,
  or file size is disproportionate, stop, simplify, disclose visual sampling, and
  rerender.
