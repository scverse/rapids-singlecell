# pertpy-GPU: `ptg`

{doc}`pertpy <pertpy:index>` provides tools for perturbation analysis {cite}`Heumos2025`.
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

<!-- panache-ignore-format-start -->
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
<!-- panache-ignore-format-end -->

## GuideAssignment

```{eval-rst}
.. autosummary::
   :toctree: generated

   GuideAssignment
```

<!-- panache-ignore-format-start -->
```{eval-rst}
.. autoclass:: GuideAssignment
    :no-index:

    .. rubric:: Methods

    .. autosummary::

        ~GuideAssignment.assign_by_threshold
        ~GuideAssignment.assign_to_max_guide
        ~GuideAssignment.assign_mixture_model

    .. automethod:: assign_by_threshold
        :no-index:
    .. automethod:: assign_to_max_guide
        :no-index:
    .. automethod:: assign_mixture_model
        :no-index:
```
<!-- panache-ignore-format-end -->

## Mixscape

Mixscape and Mixscale can score Dask-backed, log-normalized expression directly.
Use `test_method="t-test"` for marker discovery and pass `signature_kwargs` to compute
the signature for each target’s marker genes. This example uses the existing embedding in `adata.obsm["X_pca"]`.

```python
import rapids_singlecell as rsc

opts = dict(pert_key="gene_target", control="NT", test_method="t-test", signature_kwargs={"use_rep": "X_pca"})
rsc.ptg.Mixscape().mixscape(adata, **opts)
rsc.ptg.Mixscale().mixscale(adata, **opts)
```

The t-test aggregates gene statistics by perturbation across chunks. Size the scoring GPU for
the largest dense panel of selected markers across a target and its controls, plus fitting workspace
and the control-neighbor graph when using nearest neighbors.

```{eval-rst}
.. autosummary::
   :toctree: generated

   Mixscape
```

<!-- panache-ignore-format-start -->
```{eval-rst}
.. autoclass:: Mixscape
    :no-index:

    .. rubric:: Methods

    .. autosummary::

        ~Mixscape.perturbation_signature
        ~Mixscape.mixscape
        ~Mixscape.lda

    .. automethod:: perturbation_signature
        :no-index:
    .. automethod:: mixscape
        :no-index:
    .. automethod:: lda
        :no-index:
```
<!-- panache-ignore-format-end -->

## Mixscale

```{eval-rst}
.. autosummary::
   :toctree: generated

   Mixscale
```

<!-- panache-ignore-format-start -->
```{eval-rst}
.. autoclass:: Mixscale
    :no-index:

    .. rubric:: Methods

    .. autosummary::

        ~Mixscale.perturbation_signature
        ~Mixscale.mixscale

    .. automethod:: perturbation_signature
        :no-index:
    .. automethod:: mixscale
        :no-index:
```
<!-- panache-ignore-format-end -->
