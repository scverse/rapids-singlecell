from __future__ import annotations

import numpy as np

import rapids_singlecell.decoupler_gpu as dc


class TestDask:
    """Tests for Dask execution."""

    def test_mlm_dask_vs_dense(self, adata, net):
        """Test that Dask and dense produce similar results for MLM."""
        import dask.array as da

        # Dense baseline
        dense = adata.copy()
        dc.mlm(dense, net, tmin=0)

        # Dask
        dask_adata = adata.copy()
        chunks = (10, adata.shape[1])
        dask_adata.X = da.from_array(np.asarray(adata.X), chunks=chunks)
        dc.mlm(dask_adata, net, tmin=0)

        np.testing.assert_allclose(
            dense.obsm["score_mlm"].values,
            dask_adata.obsm["score_mlm"].values,
            rtol=2e-4,
        )
        np.testing.assert_allclose(
            dense.obsm["padj_mlm"].values,
            dask_adata.obsm["padj_mlm"].values,
            rtol=2e-4,
        )

    def test_ulm_dask_vs_dense(self, adata, net):
        """Test that Dask and dense produce similar results for ULM."""
        import dask.array as da

        # Dense baseline
        dense = adata.copy()
        dc.ulm(dense, net, tmin=0)

        # Dask
        dask_adata = adata.copy()
        chunks = (10, adata.shape[1])
        dask_adata.X = da.from_array(np.asarray(adata.X), chunks=chunks)
        dc.ulm(dask_adata, net, tmin=0)

        np.testing.assert_allclose(
            dense.obsm["score_ulm"].values,
            dask_adata.obsm["score_ulm"].values,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            dense.obsm["padj_ulm"].values,
            dask_adata.obsm["padj_ulm"].values,
            rtol=1e-4,
        )

    def test_aucell_dask_vs_dense(self, adata, net):
        """Test that Dask and dense produce identical results for AUCell."""
        import dask.array as da

        # Dense baseline
        dense = adata.copy()
        dc.aucell(dense, net, tmin=0)

        # Dask
        dask_adata = adata.copy()
        chunks = (10, adata.shape[1])
        dask_adata.X = da.from_array(np.asarray(adata.X), chunks=chunks)
        dc.aucell(dask_adata, net, tmin=0)

        np.testing.assert_allclose(
            dense.obsm["score_aucell"].values,
            dask_adata.obsm["score_aucell"].values,
            rtol=1e-5,
        )

    def test_zscore_dask_vs_dense(self, adata, net):
        """Test that Dask and dense produce similar results for zscore."""
        import dask.array as da

        # Dense baseline
        dense = adata.copy()
        dc.zscore(dense, net, tmin=0)

        # Dask
        dask_adata = adata.copy()
        chunks = (10, adata.shape[1])
        dask_adata.X = da.from_array(np.asarray(adata.X), chunks=chunks)
        dc.zscore(dask_adata, net, tmin=0)

        np.testing.assert_allclose(
            dense.obsm["score_zscore"].values,
            dask_adata.obsm["score_zscore"].values,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            dense.obsm["padj_zscore"].values,
            dask_adata.obsm["padj_zscore"].values,
            rtol=1e-4,
        )

    def test_waggr_dask_vs_dense(self, adata, net):
        """Test that Dask and dense produce similar results for waggr."""
        import dask.array as da

        # Dense baseline
        dense = adata.copy()
        dc.waggr(dense, net, tmin=0, times=0)

        # Dask
        dask_adata = adata.copy()
        chunks = (10, adata.shape[1])
        dask_adata.X = da.from_array(np.asarray(adata.X), chunks=chunks)
        dc.waggr(dask_adata, net, tmin=0, times=0)

        np.testing.assert_allclose(
            dense.obsm["score_waggr"].values,
            dask_adata.obsm["score_waggr"].values,
            rtol=1e-4,
        )


class TestBatchOrdering:
    """Tests to verify batch ordering is preserved correctly."""

    def test_batch_order_preserved_dask(self, adata, net):
        """Test that observation order is preserved with Dask."""
        import dask.array as da

        dask_adata = adata.copy()
        chunks = (5, adata.shape[1])  # Small chunks
        dask_adata.X = da.from_array(np.asarray(adata.X), chunks=chunks)

        dc.mlm(dask_adata, net, tmin=0)

        # Check that index matches original
        np.testing.assert_array_equal(
            dask_adata.obsm["score_mlm"].index.values,
            adata.obs_names.values,
        )
