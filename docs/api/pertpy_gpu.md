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

## DistanceTest

<!-- panache-ignore-format-start -->
```{eval-rst}
.. autosummary::
   :toctree: generated

   DistanceTest

.. autoclass:: DistanceTest
    :no-index:
    :members: __call__, test_xy, test_precomputed, contrast_distances
```
<!-- panache-ignore-format-end -->

A CuPy generator samples group labels on the GPU. Create a fresh generator
with the same seed to repeat the test. An integer seed uses NumPy’s shuffle
sequence.

```python
import cupy as cp
import rapids_singlecell as rsc

test = rsc.ptg.DistanceTest("edistance", n_perms=1000, random_state=cp.random.default_rng(0))
results = test(adata, groupby="perturbation", contrast="control")
```

To test each target against controls from the same cell class, pass the
contrast table to the test. The result includes the observed metric, p-values
and significance flags. Multiple-testing correction covers the distinct
contrasts in the supplied table.

```python
contrasts = rsc.ptg.Distance.create_contrasts(
    adata, groupby="target_gene", selected_group="Non_target", split_by="class_name"
)
results = test.contrast_distances(adata, contrasts)
```

With the default Euclidean ground distance, the test uses the same
off-diagonal within-group averages as `Distance.contrast_distances` for both
the observed and shuffled distances. The permutation test computes these in
float64. Grouped `Distance` calls also offer float32 and float16 kernels.

Leaving out `split_by` pools cells across the atlas. For example, each row
below compares all cells for one target gene with all `Non_target` cells.
A global contrast can reflect both within-cell-type changes and differences
in cell-type proportions.

```python
genes = adata.obs["target_gene"].dropna().unique()
genes = [g for g in genes if g not in {"Non_target", "Safe_target"}]
global_contrasts = rsc.ptg.Distance.create_contrasts(adata, "target_gene", "Non_target", groups=genes)
test = rsc.ptg.DistanceTest(
    "edistance", n_perms=1000, random_state=cp.random.default_rng(0), max_distance_memory=64 << 20
)
results = test.contrast_distances(adata, global_contrasts)
```

`max_distance_memory` sets a byte budget for cached cell distances or a
distance tile. By default, the test uses up to half of available GPU memory
for distances. Features, permutation labels and library workspace use
additional memory. Large balanced comparisons accumulate permutation scores
one tile at a time. For imbalanced dense Euclidean comparisons, the test can
stream the reference once, keep its row sums and reuse them across consecutive
targets. These paths use every cell and permutation. Exact distance
computation still takes quadratic work in the number of cells in a large
reference, even when its distances are streamed.

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
