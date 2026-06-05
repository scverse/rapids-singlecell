# squidpy-GPU: `gr`

{mod}`squidpy.gr` is a tool for the analysis of spatial molecular data {cite}`Palla2022`.
{mod}`rapids_singlecell.gr` accelerates some of these functions.

## Squidpy backend

With Squidpy versions that support computational backends, RAPIDS-singlecell is available as the `rapids_singlecell` backend with the aliases `cuda`, `rapids-singlecell`, and `rsc`.

```python
import squidpy as sq

sq.settings.backend = "cuda"
```

The backend exposes RAPIDS-singlecell's {mod}`rapids_singlecell.gr` functions for Squidpy's backend dispatcher.

```{eval-rst}
.. module:: rapids_singlecell.gr
.. currentmodule:: rapids_singlecell

.. autosummary::
    :toctree: generated

    gr.spatial_autocorr
    gr.co_occurrence
    gr.ligrec
    gr.calculate_niche
```
