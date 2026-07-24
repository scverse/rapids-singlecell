# Spatial analysis

Read this file for spatial graphs, niches, or large spatial plots.

## Build the physical graph

- When live RSC cannot construct the physical spatial graph, use Squidpy at an
  attributed host-side boundary. Build each sample or library independently, retain
  coordinate units and parameters, and prevent cross-sample edges.
- Keep physical and expression-derived graphs under distinct keys. Verify observation
  alignment, degree and component summaries, isolated cells, and edge distances before
  downstream RSC computation.

## Bound large spatial plots

- Estimate point and panel counts plus pixels from `figsize × dpi`. Bound figure size
  and DPI, use `rasterized=True` when supported, and downsample only contextual or grey
  background points—not analytical data.
- Treat roughly 100–150 DPI as an exploratory heuristic. If render time, pixel count,
  or file size is disproportionate, stop, simplify, disclose visual sampling, and
  rerender.
