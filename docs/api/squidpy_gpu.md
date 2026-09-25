# squidpy-GPU: `gr`

{mod}`squidpy.gr` is a tool for the analysis of spatial molecular data {cite}`Palla2022`.
{mod}`rapids_singlecell.gr` accelerates some of these functions.

```{eval-rst}
.. module:: rapids_singlecell.gr
.. module:: rapids_singlecell.gr.neighbors
.. currentmodule:: rapids_singlecell

.. autosummary::
    :toctree: generated

    gr.spatial_neighbors_knn
    gr.spatial_neighbors_radius
    gr.spatial_neighbors_delaunay
    gr.spatial_neighbors_grid
    gr.spatial_neighbors_from_builder
    gr.SpatialNeighborsResult
    gr.neighbors.GraphBuilder
    gr.neighbors.GraphBuilderCSR
    gr.neighbors.KNNBuilder
    gr.neighbors.RadiusBuilder
    gr.neighbors.DelaunayBuilder
    gr.neighbors.GridBuilder
    gr.neighbors.DistanceIntervalPostprocessor
    gr.neighbors.PercentilePostprocessor
    gr.neighbors.TransformPostprocessor
    gr.spatial_autocorr
    gr.co_occurrence
    gr.ligrec
    gr.calculate_niche
    gr.calculate_niche_neighborhood
    gr.calculate_niche_utag
    gr.calculate_niche_cellcharter
```
