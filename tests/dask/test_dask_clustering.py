from __future__ import annotations

import pytest
from scanpy.datasets import pbmc3k_processed
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import rapids_singlecell as rsc


@pytest.mark.parametrize("clustering_function", [rsc.tl.leiden, rsc.tl.louvain])
def test_dask_clustering(dist_client, clustering_function):
    adata = pbmc3k_processed()
    clustering_function(adata, use_dask=True, key_added="test_dask")
    clustering_function(adata, key_added="test_no_dask")

    assert adjusted_rand_score(adata.obs["test_dask"], adata.obs["test_no_dask"]) > 0.9
    assert (
        normalized_mutual_info_score(adata.obs["test_dask"], adata.obs["test_no_dask"])
        > 0.9
    )


@pytest.mark.parametrize("clustering_function", [rsc.tl.leiden, rsc.tl.louvain])
@pytest.mark.parametrize("resolution", [0.1, [0.5, 1.0]])
def test_dask_clustering_resolution(dist_client, clustering_function, resolution):
    adata = pbmc3k_processed()
    clustering_function(
        adata, use_dask=True, key_added="test_dask", resolution=resolution
    )
    if isinstance(resolution, list):
        for r in resolution:
            assert f"test_dask_{r}" in adata.obs.columns
    else:
        assert "test_dask" in adata.obs.columns
