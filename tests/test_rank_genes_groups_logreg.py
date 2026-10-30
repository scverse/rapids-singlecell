from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc

import rapids_singlecell as rsc


def test_rank_genes_groups_with_renamed_categories():
    adata = sc.datasets.blobs(n_variables=4, n_centers=3, n_observations=200)
    assert np.allclose(adata.X[1], [9.214668, -2.6487126, 4.2020774, 0.51076424])
    adata.obs.blobs = adata.obs.blobs.astype("category")
    # for method in ['logreg', 't-test']:

    rsc.tl.rank_genes_groups_logreg(adata, "blobs")
    assert adata.uns["rank_genes_groups"]["names"].dtype.names == ("0", "1", "2")
    assert adata.uns["rank_genes_groups"]["names"][0].tolist() == ("1", "3", "0")

    adata.rename_categories("blobs", ["Zero", "One", "Two"])
    assert adata.uns["rank_genes_groups"]["names"][0].tolist() == ("1", "3", "0")

    rsc.tl.rank_genes_groups_logreg(adata, "blobs")
    assert adata.uns["rank_genes_groups"]["names"][0].tolist() == ("1", "3", "0")
    assert adata.uns["rank_genes_groups"]["names"].dtype.names == ("Zero", "One", "Two")


def test_rank_genes_groups_with_renamed_categories_use_rep():
    adata = sc.datasets.blobs(n_variables=4, n_centers=3, n_observations=200)
    assert np.allclose(adata.X[1], [9.214668, -2.6487126, 4.2020774, 0.51076424])
    adata.obs.blobs = adata.obs.blobs.astype("category")

    adata.layers["to_test"] = adata.X.copy()
    adata.X = adata.X[::-1, :]

    rsc.tl.rank_genes_groups_logreg(adata, "blobs", layer="to_test", use_raw=False)
    assert adata.uns["rank_genes_groups"]["names"].dtype.names == ("0", "1", "2")
    assert adata.uns["rank_genes_groups"]["names"][0].tolist() == ("1", "3", "0")

    rsc.tl.rank_genes_groups_logreg(adata, "blobs")
    assert not adata.uns["rank_genes_groups"]["names"][0].tolist() == ("3", "1", "0")


def test_rank_genes_groups_with_unsorted_groups():
    adata = sc.datasets.blobs(n_variables=10, n_centers=5, n_observations=200)
    adata._sanitize()
    adata.rename_categories("blobs", ["Zero", "One", "Two", "Three", "Four"])
    bdata = adata.copy()
    rsc.tl.rank_genes_groups_logreg(adata, "blobs", groups=["Zero", "One", "Three"])
    rsc.tl.rank_genes_groups_logreg(bdata, "blobs", groups=["One", "Three", "Zero"])
    array_ad = pd.DataFrame(adata.uns["rank_genes_groups"]["scores"]["Three"]).to_numpy()
    array_bd = pd.DataFrame(bdata.uns["rank_genes_groups"]["scores"]["Three"]).to_numpy()
    np.testing.assert_equal(array_ad, array_bd)
