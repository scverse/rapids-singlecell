# pertpy-GPU: `ptg`

[pertpy](https://pertpy.readthedocs.io) provides tools for perturbation analysis {cite}`Heumos2025`.
{mod}`rapids_singlecell.ptg` accelerates some of these methods.

```{eval-rst}
.. module:: rapids_singlecell.ptg
.. currentmodule:: rapids_singlecell.ptg
```

## Distance

```{eval-rst}
.. autosummary::
    :toctree: generated

    Distance
```

```{eval-rst}
.. autoclass:: Distance
    :no-index:

    .. rubric:: Methods

    .. autosummary::

        ~Distance.pairwise
        ~Distance.onesided_distances
        ~Distance.contrast_distances
        ~Distance.create_contrasts
        ~Distance.bootstrap

    .. automethod:: __call__
        :no-index:
    .. automethod:: pairwise
        :no-index:
    .. automethod:: onesided_distances
        :no-index:
    .. automethod:: contrast_distances
        :no-index:
    .. automethod:: create_contrasts
        :no-index:
    .. automethod:: bootstrap
        :no-index:
```
