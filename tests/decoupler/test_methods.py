from __future__ import annotations

import cupy as cp
import cupyx.scipy.sparse as csps
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps

import rapids_singlecell.decoupler_gpu as dc


class TestMLM:
    """Tests for mlm public API."""

    def test_mlm_adata_inplace(self, adata, net):
        """Test mlm with AnnData input stores results in obsm."""
        result = dc.mlm(adata, net, tmin=0)
        assert result is None
        assert "score_mlm" in adata.obsm
        assert "padj_mlm" in adata.obsm
        assert adata.obsm["score_mlm"].shape[0] == adata.n_obs

    def test_mlm_dataframe(self, adata, net):
        """Test mlm with DataFrame input returns tuple."""
        df = adata.to_df()
        es, pv = dc.mlm(df, net, tmin=0)
        assert isinstance(es, pd.DataFrame)
        assert isinstance(pv, pd.DataFrame)
        assert es.shape[0] == df.shape[0]

    def test_mlm_sparse(self, adata, net):
        """Test mlm with sparse AnnData."""
        adata_sparse = adata.copy()
        adata_sparse.X = sps.csr_matrix(adata_sparse.X)
        result = dc.mlm(adata_sparse, net, tmin=0)
        assert result is None
        assert "score_mlm" in adata_sparse.obsm

    def test_mlm_layer(self, adata, net):
        """Test mlm using a layer."""
        dc.mlm(adata, net, layer="counts", tmin=0)
        assert "score_mlm" in adata.obsm

    def test_mlm_pre_load(self, adata, net):
        """Test mlm with pre_load=True."""
        dc.mlm(adata, net, tmin=0, pre_load=True)
        assert "score_mlm" in adata.obsm


class TestULM:
    """Tests for ulm public API."""

    def test_ulm_adata_inplace(self, adata, net):
        """Test ulm with AnnData input stores results in obsm."""
        result = dc.ulm(adata, net, tmin=0)
        assert result is None
        assert "score_ulm" in adata.obsm
        assert "padj_ulm" in adata.obsm
        assert adata.obsm["score_ulm"].shape[0] == adata.n_obs

    def test_ulm_dataframe(self, adata, net):
        """Test ulm with DataFrame input returns tuple."""
        df = adata.to_df()
        es, pv = dc.ulm(df, net, tmin=0)
        assert isinstance(es, pd.DataFrame)
        assert isinstance(pv, pd.DataFrame)
        assert es.shape[0] == df.shape[0]

    def test_ulm_sparse(self, adata, net):
        """Test ulm with sparse AnnData."""
        adata_sparse = adata.copy()
        adata_sparse.X = sps.csr_matrix(adata_sparse.X)
        result = dc.ulm(adata_sparse, net, tmin=0)
        assert result is None
        assert "score_ulm" in adata_sparse.obsm


class TestAUCell:
    """Tests for aucell public API."""

    def test_aucell_adata_inplace(self, adata, net):
        """Test aucell with AnnData input stores results in obsm."""
        result = dc.aucell(adata, net, tmin=0)
        assert result is None
        assert "score_aucell" in adata.obsm
        assert adata.obsm["score_aucell"].shape[0] == adata.n_obs

    def test_aucell_dataframe(self, adata, net):
        """Test aucell with DataFrame input returns tuple."""
        df = adata.to_df()
        es, pv_result = dc.aucell(df, net, tmin=0)
        assert isinstance(es, pd.DataFrame)
        assert pv_result is None
        assert es.shape[0] == df.shape[0]

    def test_aucell_sparse(self, adata, net):
        """Test aucell with sparse AnnData."""
        adata_sparse = adata.copy()
        adata_sparse.X = sps.csr_matrix(adata_sparse.X)
        result = dc.aucell(adata_sparse, net, tmin=0)
        assert result is None
        assert "score_aucell" in adata_sparse.obsm

    def test_aucell_n_up(self, adata, net):
        """Test aucell with custom n_up parameter."""
        dc.aucell(adata, net, tmin=0, n_up=5)
        assert "score_aucell" in adata.obsm

    def test_aucell_scores_bounded(self, adata, net):
        """Test that aucell scores are between 0 and 1."""
        dc.aucell(adata, net, tmin=0)
        scores = adata.obsm["score_aucell"].values
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)


class TestZscore:
    """Tests for zscore public API."""

    @pytest.mark.parametrize("flavor", ["KSEA", "RoKAI"])
    def test_zscore_adata_inplace(self, adata, net, flavor):
        """Test zscore with AnnData input stores results in obsm."""
        result = dc.zscore(adata, net, tmin=0, flavor=flavor)
        assert result is None
        assert "score_zscore" in adata.obsm
        assert "padj_zscore" in adata.obsm

    def test_zscore_dataframe(self, adata, net):
        """Test zscore with DataFrame input returns tuple."""
        df = adata.to_df()
        es, pv = dc.zscore(df, net, tmin=0)
        assert isinstance(es, pd.DataFrame)
        assert isinstance(pv, pd.DataFrame)
        assert es.shape[0] == df.shape[0]


class TestWaggr:
    """Tests for waggr public API."""

    @pytest.mark.parametrize("fun", ["wmean", "wsum"])
    def test_waggr_adata_inplace(self, adata, net, fun):
        """Test waggr with AnnData input stores results in obsm."""
        result = dc.waggr(adata, net, tmin=0, fun=fun, times=0)
        assert result is None
        assert "score_waggr" in adata.obsm

    def test_waggr_dataframe(self, adata, net):
        """Test waggr with DataFrame input returns tuple."""
        df = adata.to_df()
        es, pv = dc.waggr(df, net, tmin=0, times=0)
        assert isinstance(es, pd.DataFrame)
        assert isinstance(pv, pd.DataFrame)
        assert es.shape[0] == df.shape[0]


class TestInputTypes:
    """Tests for various input types across all methods."""

    @pytest.mark.parametrize(
        "method,has_pvals",
        [
            ("mlm", True),
            ("ulm", True),
            ("aucell", False),
            ("zscore", True),
        ],
    )
    def test_cupy_sparse_input(self, adata, net, method, has_pvals):
        """Test methods with CuPy sparse matrix input."""
        adata_cupy = adata.copy()
        adata_cupy.X = csps.csr_matrix(cp.array(adata.X))
        func = getattr(dc, method)
        func(adata_cupy, net, tmin=0)
        assert f"score_{method}" in adata_cupy.obsm
        if has_pvals:
            assert f"padj_{method}" in adata_cupy.obsm

    @pytest.mark.parametrize("method", ["mlm", "ulm", "aucell", "zscore"])
    def test_list_input(self, adata, net, method):
        """Test methods with list input [matrix, samples, features]."""
        data = [adata.X, adata.obs_names.tolist(), adata.var_names.tolist()]
        func = getattr(dc, method)
        es, pv = func(data, net, tmin=0)
        assert isinstance(es, pd.DataFrame)
        assert es.shape[0] == adata.n_obs


class TestBatching:
    """Tests for batch processing."""

    def test_mlm_small_batch(self, adata, net):
        """Test mlm with small batch size."""
        dc.mlm(adata, net, tmin=0, bsize=5)
        assert "score_mlm" in adata.obsm

    def test_ulm_small_batch(self, adata, net):
        """Test ulm with small batch size."""
        dc.ulm(adata, net, tmin=0, bsize=5)
        assert "score_ulm" in adata.obsm

    def test_aucell_small_batch(self, adata, net):
        """Test aucell with small batch size."""
        dc.aucell(adata, net, tmin=0, bsize=5)
        assert "score_aucell" in adata.obsm


class TestEmptyObservations:
    """Tests for handling empty observations."""

    def test_mlm_removes_empty_obs(self, adata, net):
        """Test that empty observations are removed and AnnData is repaired."""
        adata_empty = adata.copy()
        adata_empty.X[0, :] = 0.0
        result = dc.mlm(adata_empty, net, tmin=0, empty=True)
        # When obs are removed, a repaired AnnData is returned
        if result is not None:
            assert result.n_obs < adata.n_obs
        else:
            assert adata_empty.obsm["score_mlm"].shape[0] == adata_empty.n_obs


class TestRaw:
    """Tests for using raw attribute."""

    def test_mlm_raw(self, adata, net):
        """Test mlm using raw attribute."""
        adata_raw = adata.copy()
        adata_raw.raw = adata_raw.copy()
        adata_raw.X = adata_raw.X * 2  # Modify X
        dc.mlm(adata_raw, net, tmin=0, raw=True)
        assert "score_mlm" in adata_raw.obsm


class TestReturnFunction:
    """Tests for _return helper function."""

    def test_return_dataframe_input(self, adata, net):
        """Test that DataFrame input returns tuple."""
        df = adata.to_df()
        result = dc.mlm(df, net, tmin=0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_return_adata_inplace(self, adata, net):
        """Test that AnnData input modifies in place and returns None."""
        result = dc.mlm(adata, net, tmin=0)
        assert result is None
        assert "score_mlm" in adata.obsm


class TestMethodMeta:
    """Tests for MethodMeta functionality."""

    def test_mlm_meta(self):
        """Test that mlm has correct metadata."""
        meta = dc.mlm.meta()
        assert isinstance(meta, pd.DataFrame)
        assert meta["name"].values[0] == "mlm"
        assert meta["test"].values[0]

    def test_aucell_meta(self):
        """Test that aucell has correct metadata."""
        meta = dc.aucell.meta()
        assert isinstance(meta, pd.DataFrame)
        assert meta["name"].values[0] == "aucell"
        assert not meta["test"].values[0]
