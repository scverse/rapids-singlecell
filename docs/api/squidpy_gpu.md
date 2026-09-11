# squidpy-GPU: `gr`

{mod}`squidpy.gr` is a tool for the analysis of spatial molecular data {cite}`Palla2022`.
{mod}`rapids_singlecell.gr` accelerates some of these functions.

## Squidpy backend

With Squidpy versions that support computational backends, RAPIDS-singlecell is
available as the `rapids-singlecell` backend with the aliases `cuda`, `rapids`,
and `rapids_singlecell`.

```python
import squidpy as sq

sq.settings.backend = "cuda"
```

The backend exposes RAPIDS-singlecell's {mod}`rapids_singlecell.gr` functions for Squidpy's backend dispatcher, including the flavor-specific {func}`~rapids_singlecell.gr.calculate_niche_neighborhood`, {func}`~rapids_singlecell.gr.calculate_niche_utag` and {func}`~rapids_singlecell.gr.calculate_niche_cellcharter`.

Like their Squidpy counterparts, {func}`~rapids_singlecell.gr.spatial_autocorr`, {func}`~rapids_singlecell.gr.co_occurrence`, {func}`~rapids_singlecell.gr.ligrec` and the `calculate_niche_*` functions accept a {class}`~spatialdata.SpatialData` together with `table_key`. `flavor="spatialleiden"` of {func}`squidpy.gr.calculate_niche` has no GPU implementation and raises `NotImplementedError` on this backend; `calculate_niche_spatialleiden` is not exposed, so Squidpy runs it on the CPU.

Niche calls use RAPIDS-singlecell's native defaults and validation. The deprecated {func}`~rapids_singlecell.gr.calculate_niche` supplies `n_neighbors=15` and `resolutions=(0.5,)` when omitted and emits its native deprecation warning; prefer the flavor-specific functions. The backend translates Squidpy's `data`, `copy`, and `rng` arguments to the native names.

```{eval-rst}
.. module:: rapids_singlecell.gr
.. currentmodule:: rapids_singlecell

.. autosummary::
    :toctree: generated

    gr.spatial_autocorr
    gr.co_occurrence
    gr.ligrec
    gr.calculate_niche
    gr.calculate_niche_neighborhood
    gr.calculate_niche_utag
    gr.calculate_niche_cellcharter
```
