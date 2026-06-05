from __future__ import annotations

import numpy as np
import pytest
import scanpy as sc

import rapids_singlecell as rsc
from testing.rapids_singlecell._helper import ARRAY_TYPES_MEM


@pytest.fixture
def adata_blobs():
    """Create a reproducible blobs dataset for testing."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=10, n_centers=3, n_observations=300)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")
    # Shift to non-negative range (simulating log1p data)
    adata.X = np.abs(adata.X).astype(np.float32)
    return adata


def _compare_gene_rankings(
    binned_result, exact_result, *, n_top: int = 5, field: str = "names"
):
    """Check that binned and exact results share most top genes."""
    for group in binned_result[field].dtype.names:
        binned_genes = set(binned_result[field][group][:n_top])
        exact_genes = set(exact_result[field][group][:n_top])
        overlap = len(binned_genes & exact_genes)
        # Allow some disagreement due to approximation, but expect good overlap
        assert overlap >= n_top - 2, (
            f"Group {group}: only {overlap}/{n_top} top genes overlap. "
            f"Binned: {binned_genes}, Exact: {exact_genes}"
        )


@pytest.mark.parametrize("array_type", ARRAY_TYPES_MEM)
class TestWilcoxonBinned:
    """Tests for wilcoxon_binned across all array types."""

    def test_basic_output_structure(self, adata_blobs, array_type):
        """Test that output has the standard structure."""
        adata = adata_blobs.copy()
        adata.X = array_type(adata.X)

        rsc.tl.rank_genes_groups(
            adata, "blobs", method="wilcoxon_binned", use_raw=False
        )

        result = adata.uns["rank_genes_groups"]
        assert "names" in result
        assert "scores" in result
        assert "pvals" in result
        assert "pvals_adj" in result
        assert "logfoldchanges" in result
        assert result["params"]["method"] == "wilcoxon_binned"

    def test_all_groups_present(self, adata_blobs, array_type):
        """Test that all groups appear in results."""
        adata = adata_blobs.copy()
        adata.X = array_type(adata.X)

        rsc.tl.rank_genes_groups(
            adata, "blobs", method="wilcoxon_binned", use_raw=False
        )

        result = adata.uns["rank_genes_groups"]
        expected_groups = ("0", "1", "2")
        assert result["names"].dtype.names == expected_groups
        assert result["scores"].dtype.names == expected_groups

    def test_pvals_in_valid_range(self, adata_blobs, array_type):
        """P-values should be in [0, 1]."""
        adata = adata_blobs.copy()
        adata.X = array_type(adata.X)

        rsc.tl.rank_genes_groups(
            adata, "blobs", method="wilcoxon_binned", use_raw=False
        )

        result = adata.uns["rank_genes_groups"]
        for group in result["pvals"].dtype.names:
            pvals = np.asarray(result["pvals"][group], dtype=float)
            assert np.all(pvals >= 0), f"Negative p-values in group {group}"
            assert np.all(pvals <= 1), f"P-values > 1 in group {group}"


@pytest.mark.parametrize("array_type", ARRAY_TYPES_MEM)
class TestWilcoxonBinnedMatchesExact:
    """Test that binned results agree with exact Wilcoxon."""

    def test_matches_exact_wilcoxon_rankings(self, adata_blobs, array_type):
        """Top-ranked genes should mostly agree with exact Wilcoxon."""
        adata_binned = adata_blobs.copy()
        adata_exact = adata_blobs.copy()

        adata_binned.X = array_type(adata_binned.X)

        rsc.tl.rank_genes_groups(
            adata_binned, "blobs", method="wilcoxon_binned", use_raw=False
        )
        rsc.tl.rank_genes_groups(adata_exact, "blobs", method="wilcoxon", use_raw=False)

        _compare_gene_rankings(
            adata_binned.uns["rank_genes_groups"],
            adata_exact.uns["rank_genes_groups"],
            n_top=5,
        )

    def test_scores_correlate_with_exact(self, adata_blobs, array_type):
        """Z-scores should be highly correlated with exact Wilcoxon."""
        adata_binned = adata_blobs.copy()
        adata_exact = adata_blobs.copy()

        adata_binned.X = array_type(adata_binned.X)

        rsc.tl.rank_genes_groups(
            adata_binned, "blobs", method="wilcoxon_binned", use_raw=False
        )
        rsc.tl.rank_genes_groups(adata_exact, "blobs", method="wilcoxon", use_raw=False)

        for group in adata_binned.uns["rank_genes_groups"]["scores"].dtype.names:
            binned_scores = np.asarray(
                adata_binned.uns["rank_genes_groups"]["scores"][group], dtype=float
            )
            exact_scores = np.asarray(
                adata_exact.uns["rank_genes_groups"]["scores"][group], dtype=float
            )
            corr = np.corrcoef(binned_scores, exact_scores)[0, 1]
            assert corr > 0.95, (
                f"Group {group}: correlation {corr:.4f} between binned and exact scores"
            )


class TestWilcoxonBinnedParameters:
    """Test parameter handling."""

    def test_n_genes_parameter(self, adata_blobs):
        adata = adata_blobs.copy()
        rsc.get.anndata_to_GPU(adata)

        rsc.tl.rank_genes_groups(
            adata, "blobs", method="wilcoxon_binned", use_raw=False, n_genes=3
        )

        result = adata.uns["rank_genes_groups"]
        for group in result["names"].dtype.names:
            assert len(result["names"][group]) == 3

    def test_bonferroni_correction(self, adata_blobs):
        adata = adata_blobs.copy()
        rsc.get.anndata_to_GPU(adata)

        rsc.tl.rank_genes_groups(
            adata,
            "blobs",
            method="wilcoxon_binned",
            use_raw=False,
            corr_method="bonferroni",
        )

        result = adata.uns["rank_genes_groups"]
        for group in result["pvals_adj"].dtype.names:
            pvals_adj = np.asarray(result["pvals_adj"][group], dtype=float)
            assert np.all(pvals_adj <= 1.0)

    def test_custom_n_bins(self, adata_blobs):
        adata_low = adata_blobs.copy()
        adata_high = adata_blobs.copy()
        rsc.get.anndata_to_GPU(adata_low)
        rsc.get.anndata_to_GPU(adata_high)

        rsc.tl.rank_genes_groups(
            adata_low, "blobs", method="wilcoxon_binned", use_raw=False, n_bins=100
        )
        rsc.tl.rank_genes_groups(
            adata_high, "blobs", method="wilcoxon_binned", use_raw=False, n_bins=5000
        )

        for adata in [adata_low, adata_high]:
            result = adata.uns["rank_genes_groups"]
            assert "scores" in result
            for group in result["pvals"].dtype.names:
                pvals = np.asarray(result["pvals"][group], dtype=float)
                assert np.all(pvals >= 0)
                assert np.all(pvals <= 1)

    def test_chunk_size(self, adata_blobs):
        """Test with different chunk_size values produce same results."""
        adata_small = adata_blobs.copy()
        adata_large = adata_blobs.copy()
        rsc.get.anndata_to_GPU(adata_small)
        rsc.get.anndata_to_GPU(adata_large)

        rsc.tl.rank_genes_groups(
            adata_small,
            "blobs",
            method="wilcoxon_binned",
            use_raw=False,
            chunk_size=3,
        )
        rsc.tl.rank_genes_groups(
            adata_large,
            "blobs",
            method="wilcoxon_binned",
            use_raw=False,
            chunk_size=1000,
        )

        for group in adata_small.uns["rank_genes_groups"]["scores"].dtype.names:
            scores_small = np.asarray(
                adata_small.uns["rank_genes_groups"]["scores"][group], dtype=float
            )
            scores_large = np.asarray(
                adata_large.uns["rank_genes_groups"]["scores"][group], dtype=float
            )
            np.testing.assert_allclose(scores_small, scores_large, rtol=1e-10)


class TestWilcoxonBinnedEdgeCases:
    """Edge case tests."""

    @pytest.mark.parametrize(
        ("reference", "groups"),
        [
            pytest.param("1", "all", id="ref_all_groups"),
            pytest.param("1", ["0", "1", "2"], id="ref_group_subset"),
        ],
    )
    def test_reference_matches_exact(self, adata_blobs, reference, groups):
        """wilcoxon_binned with reference should match exact wilcoxon rankings."""
        adata_binned = adata_blobs.copy()
        rsc.get.anndata_to_GPU(adata_binned)
        adata_exact = adata_binned.copy()

        rsc.tl.rank_genes_groups(
            adata_binned,
            "blobs",
            method="wilcoxon_binned",
            reference=reference,
            groups=groups,
            use_raw=False,
        )
        rsc.tl.rank_genes_groups(
            adata_exact,
            "blobs",
            method="wilcoxon",
            reference=reference,
            groups=groups,
            use_raw=False,
        )

        result_b = adata_binned.uns["rank_genes_groups"]
        result_e = adata_exact.uns["rank_genes_groups"]

        assert reference not in result_b["names"].dtype.names
        assert set(result_b["names"].dtype.names) == set(result_e["names"].dtype.names)

        for group in result_b["names"].dtype.names:
            names_b = list(result_b["names"][group])
            names_e = list(result_e["names"][group])
            assert names_b == names_e, f"Ranking mismatch for group {group}"

    @pytest.mark.parametrize(
        ("reference", "groups"),
        [
            pytest.param("rest", ["0"], id="rest_single_group"),
            pytest.param("rest", ["0", "2"], id="rest_group_subset"),
            pytest.param("1", ["0"], id="ref_single_group"),
            pytest.param("1", ["0", "1", "2"], id="ref_group_subset"),
        ],
    )
    def test_group_subset_matches_all_groups(self, adata_blobs, reference, groups):
        """Scores with group subset should match all-groups run."""
        adata_all = adata_blobs.copy()
        adata_sub = adata_blobs.copy()
        rsc.get.anndata_to_GPU(adata_all)
        rsc.get.anndata_to_GPU(adata_sub)

        rsc.tl.rank_genes_groups(
            adata_all,
            "blobs",
            method="wilcoxon_binned",
            reference=reference,
            use_raw=False,
        )
        rsc.tl.rank_genes_groups(
            adata_sub,
            "blobs",
            method="wilcoxon_binned",
            reference=reference,
            groups=groups,
            use_raw=False,
        )

        result_all = adata_all.uns["rank_genes_groups"]
        result_sub = adata_sub.uns["rank_genes_groups"]

        # Subset groups should be a subset of all-groups result
        assert set(result_sub["names"].dtype.names) <= set(
            result_all["names"].dtype.names
        )

        for group in result_sub["names"].dtype.names:
            assert tuple(result_all["names"][group]) == tuple(
                result_sub["names"][group]
            ), f"names mismatch for group {group}"
            for field in ("scores", "logfoldchanges", "pvals", "pvals_adj"):
                np.testing.assert_allclose(
                    np.asarray(result_all[field][group], dtype=float),
                    np.asarray(result_sub[field][group], dtype=float),
                    rtol=1e-10,
                    atol=1e-12,
                    err_msg=f"{field} mismatch for group {group}",
                )

    @pytest.mark.parametrize("reference", ["rest", "1"])
    def test_unsorted_groups(self, adata_blobs, reference):
        """Test that group order doesn't affect results."""
        adata = adata_blobs.copy()
        bdata = adata_blobs.copy()
        rsc.get.anndata_to_GPU(adata)
        rsc.get.anndata_to_GPU(bdata)

        groups = ["0", "1", "2"] if reference != "rest" else ["0", "2"]
        groups_reversed = list(reversed(groups))

        rsc.tl.rank_genes_groups(
            adata,
            "blobs",
            method="wilcoxon_binned",
            groups=groups,
            reference=reference,
            use_raw=False,
        )
        rsc.tl.rank_genes_groups(
            bdata,
            "blobs",
            method="wilcoxon_binned",
            groups=groups_reversed,
            reference=reference,
            use_raw=False,
        )

        expected_groups = {g for g in groups if g != reference}
        assert (
            set(adata.uns["rank_genes_groups"]["names"].dtype.names) == expected_groups
        )
        assert (
            set(bdata.uns["rank_genes_groups"]["names"].dtype.names) == expected_groups
        )

        test_group = next(iter(expected_groups))
        for field in ("scores", "logfoldchanges", "pvals", "pvals_adj"):
            np.testing.assert_allclose(
                np.asarray(
                    adata.uns["rank_genes_groups"][field][test_group], dtype=float
                ),
                np.asarray(
                    bdata.uns["rank_genes_groups"][field][test_group], dtype=float
                ),
                rtol=1e-5,
                atol=1e-6,
                equal_nan=True,
            )

        assert tuple(adata.uns["rank_genes_groups"]["names"][test_group]) == tuple(
            bdata.uns["rank_genes_groups"]["names"][test_group]
        )

    def test_mask_var(self, adata_blobs):
        """Test with mask_var to select subset of genes."""
        adata = adata_blobs.copy()
        rsc.get.anndata_to_GPU(adata)

        mask = np.zeros(adata.n_vars, dtype=bool)
        mask[:5] = True

        rsc.tl.rank_genes_groups(
            adata,
            "blobs",
            method="wilcoxon_binned",
            use_raw=False,
            mask_var=mask,
        )

        result = adata.uns["rank_genes_groups"]
        for group in result["names"].dtype.names:
            n_genes = len(result["names"][group])
            assert n_genes == 5

    def test_layer_parameter(self, adata_blobs):
        """Test that layer parameter is respected."""
        adata = adata_blobs.copy()
        adata.layers["test_layer"] = adata.X.copy() * 2
        rsc.get.anndata_to_GPU(adata)

        rsc.tl.rank_genes_groups(
            adata,
            "blobs",
            method="wilcoxon_binned",
            layer="test_layer",
            use_raw=False,
        )

        result = adata.uns["rank_genes_groups"]
        assert result["params"]["layer"] == "test_layer"
        assert "scores" in result

    def test_sparse_with_actual_zeros(self, adata_blobs):
        """Test sparse with significant zero fraction (like real scRNA-seq)."""
        from scipy.sparse import csr_matrix

        adata = adata_blobs.copy()
        # Zero out ~70% of entries
        rng = np.random.default_rng(123)
        mask = rng.random(adata.X.shape) < 0.7
        adata.X[mask] = 0.0
        adata.X = csr_matrix(adata.X)

        rsc.tl.rank_genes_groups(
            adata, "blobs", method="wilcoxon_binned", use_raw=False
        )

        result = adata.uns["rank_genes_groups"]
        for group in result["pvals"].dtype.names:
            pvals = np.asarray(result["pvals"][group], dtype=float)
            assert np.all(pvals >= 0)
            assert np.all(pvals <= 1)

    def test_sparse_negative_values_raises(self, adata_blobs):
        """Sparse input with negative values should raise ValueError."""
        import cupy as cp
        import cupyx.scipy.sparse as cpsp

        adata = adata_blobs.copy()
        rsc.get.anndata_to_GPU(adata)
        # Make sparse with negative values
        dense = cp.array(adata.X)
        dense[:, 0] = -1.0
        adata.X = cpsp.csr_matrix(dense)

        with pytest.raises(ValueError, match="Sparse input contains negative values"):
            rsc.tl.rank_genes_groups(
                adata, "blobs", method="wilcoxon_binned", use_raw=False
            )

    def test_log1p_warning(self, adata_blobs):
        """Warning should fire when adata.uns['log1p'] is missing."""
        adata = adata_blobs.copy()
        rsc.get.anndata_to_GPU(adata)
        # Ensure no log1p key
        adata.uns.pop("log1p", None)

        with pytest.warns(UserWarning, match="log1p"):
            rsc.tl.rank_genes_groups(
                adata, "blobs", method="wilcoxon_binned", use_raw=False
            )

    def test_constant_data(self, adata_blobs):
        """All-constant data (bin_width <= 0) should not crash."""
        adata = adata_blobs.copy()
        adata.X = np.ones_like(adata.X)
        rsc.get.anndata_to_GPU(adata)

        rsc.tl.rank_genes_groups(
            adata, "blobs", method="wilcoxon_binned", use_raw=False
        )

        result = adata.uns["rank_genes_groups"]
        for group in result["pvals"].dtype.names:
            pvals = np.asarray(result["pvals"][group], dtype=float)
            assert np.all(np.isfinite(pvals))

    def test_bin_range_auto_vs_log1p_similar(self, adata_blobs):
        """bin_range='auto' and 'log1p' should give similar results on log1p data."""
        adata_auto = adata_blobs.copy()
        adata_log1p = adata_blobs.copy()
        rsc.get.anndata_to_GPU(adata_auto)
        rsc.get.anndata_to_GPU(adata_log1p)

        rsc.tl.rank_genes_groups(
            adata_auto,
            "blobs",
            method="wilcoxon_binned",
            use_raw=False,
            bin_range="auto",
        )
        rsc.tl.rank_genes_groups(
            adata_log1p,
            "blobs",
            method="wilcoxon_binned",
            use_raw=False,
            bin_range="log1p",
        )

        for group in adata_auto.uns["rank_genes_groups"]["scores"].dtype.names:
            scores_auto = np.asarray(
                adata_auto.uns["rank_genes_groups"]["scores"][group], dtype=float
            )
            scores_log1p = np.asarray(
                adata_log1p.uns["rank_genes_groups"]["scores"][group], dtype=float
            )
            corr = np.corrcoef(scores_auto, scores_log1p)[0, 1]
            assert corr > 0.99, f"Group {group}: auto vs log1p correlation {corr:.4f}"


def test_top_genes_match_scipy(adata_blobs):
    """Top DE genes by p-value should match scipy mannwhitneyu ranking."""
    from scipy.stats import mannwhitneyu

    adata = adata_blobs.copy()
    rsc.get.anndata_to_GPU(adata)
    rsc.tl.rank_genes_groups(
        adata, "blobs", method="wilcoxon_binned", use_raw=False, tie_correct=True
    )

    X = adata_blobs.X.astype(np.float64)
    groups = adata_blobs.obs["blobs"]
    n_top = 5
    for group in adata.uns["rank_genes_groups"]["names"].dtype.names:
        # Sort binned results by p-value (direction-agnostic)
        pvals_binned = adata.uns["rank_genes_groups"]["pvals"][group]
        binned_order = np.argsort(pvals_binned)
        binned_top = set(
            adata.uns["rank_genes_groups"]["names"][group][binned_order[:n_top]]
        )

        mask_g = (groups == group).values
        pvals = [
            mannwhitneyu(X[mask_g, i], X[~mask_g, i], alternative="two-sided").pvalue
            for i in range(X.shape[1])
        ]
        scipy_top = set(adata_blobs.var_names[np.argsort(pvals)[:n_top]])
        overlap = len(binned_top & scipy_top)
        assert overlap >= n_top - 1, f"Group {group}: {overlap}/{n_top} overlap"
