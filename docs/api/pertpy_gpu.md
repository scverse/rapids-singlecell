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
