from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import mannwhitneyu, rankdata, tiecorrect

import rapids_singlecell as rsc


@pytest.mark.parametrize("reference", ["rest", "1"])
@pytest.mark.parametrize("tie_correct", [True, False])
@pytest.mark.parametrize("sparse", [True, False])
def test_rank_genes_groups_wilcoxon_matches_scanpy(reference, tie_correct, sparse):
    """Test wilcoxon matches scanpy output across configurations."""
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=6, n_centers=3, n_observations=200)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")

    if sparse:
        adata_gpu.X = sp.csr_matrix(adata_gpu.X)

    adata_cpu = adata_gpu.copy()

    rsc.tl.rank_genes_groups(
        adata_gpu,
        "blobs",
        method="wilcoxon",
        use_raw=False,
        n_genes=3,
        reference=reference,
        corr_method="benjamini-hochberg",
        tie_correct=tie_correct,
    )
    sc.tl.rank_genes_groups(
        adata_cpu,
        "blobs",
        method="wilcoxon",
        use_raw=False,
        n_genes=3,
        reference=reference,
        tie_correct=tie_correct,
    )

    gpu_result = adata_gpu.uns["rank_genes_groups"]
    cpu_result = adata_cpu.uns["rank_genes_groups"]

    assert gpu_result["names"].dtype.names == cpu_result["names"].dtype.names
    for group in gpu_result["names"].dtype.names:
        assert list(gpu_result["names"][group]) == list(cpu_result["names"][group])

    for field in ("scores", "logfoldchanges", "pvals", "pvals_adj"):
        gpu_field = gpu_result[field]
        cpu_field = cpu_result[field]
        assert gpu_field.dtype.names == cpu_field.dtype.names
        for group in gpu_field.dtype.names:
            gpu_values = np.asarray(gpu_field[group], dtype=float)
            cpu_values = np.asarray(cpu_field[group], dtype=float)
            np.testing.assert_allclose(gpu_values, cpu_values, rtol=1e-13, atol=1e-15)

    params = gpu_result["params"]
    assert params["use_raw"] is False
    assert params["corr_method"] == "benjamini-hochberg"
    assert params["tie_correct"] is tie_correct
    assert params["layer"] is None
    assert params["reference"] == reference


@pytest.mark.parametrize("reference", ["rest", "1"])
def test_rank_genes_groups_wilcoxon_honors_layer_and_use_raw(reference):
    """Test that layer parameter is respected."""
    np.random.seed(42)
    base = sc.datasets.blobs(n_variables=5, n_centers=3, n_observations=150)
    base.obs["blobs"] = base.obs["blobs"].astype("category")
    base.layers["signal"] = base.X.copy()

    ref_adata = base.copy()
    rsc.tl.rank_genes_groups(
        ref_adata, "blobs", method="wilcoxon", use_raw=False, reference=reference
    )
    reference_names = ref_adata.uns["rank_genes_groups"]["names"].copy()

    rng = np.random.default_rng(0)
    perturbed_matrix = base.X.copy()
    perturbed_matrix[rng.integers(0, 2, perturbed_matrix.shape, dtype=bool)] = 0.0

    layered = base.copy()
    layered.X = perturbed_matrix
    rsc.tl.rank_genes_groups(
        layered,
        "blobs",
        method="wilcoxon",
        layer="signal",
        use_raw=False,
        reference=reference,
    )
    layered_names = layered.uns["rank_genes_groups"]["names"].copy()

    no_layer = base.copy()
    no_layer.X = perturbed_matrix
    rsc.tl.rank_genes_groups(
        no_layer, "blobs", method="wilcoxon", use_raw=False, reference=reference
    )
    no_layer_names = no_layer.uns["rank_genes_groups"]["names"].copy()

    assert layered_names.dtype.names == reference_names.dtype.names
    for group in reference_names.dtype.names:
        assert tuple(layered_names[group]) == tuple(reference_names[group])
    differences = [
        tuple(no_layer_names[group]) != tuple(reference_names[group])
        for group in reference_names.dtype.names
    ]
    assert any(differences)


@pytest.mark.parametrize("reference", ["rest", "1"])
def test_rank_genes_groups_wilcoxon_subset_and_bonferroni(reference):
    """Test group subsetting and bonferroni correction."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=5, n_centers=4, n_observations=150)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")

    groups = ["0", "1", "2"] if reference != "rest" else ["0", "2"]

    rsc.tl.rank_genes_groups(
        adata,
        "blobs",
        method="wilcoxon",
        groups=groups,
        reference=reference,
        use_raw=False,
        n_genes=2,
        corr_method="bonferroni",
    )

    result = adata.uns["rank_genes_groups"]
    expected_groups = tuple(g for g in groups if g != reference)
    assert result["scores"].dtype.names == expected_groups
    assert result["names"].dtype.names == expected_groups
    for group in result["names"].dtype.names:
        observed = np.asarray(result["names"][group])
        assert observed.size == 2
    for group in result["pvals_adj"].dtype.names:
        adjusted = np.asarray(result["pvals_adj"][group])
        assert np.all(adjusted <= 1.0)


@pytest.mark.parametrize(
    ("groups", "reference"),
    [
        (["0"], "rest"),
        (["0", "2"], "rest"),
        (["0"], "1"),
        (["0", "2"], "1"),
    ],
)
@pytest.mark.parametrize("tie_correct", [False, True])
@pytest.mark.parametrize("pre_load", [False, True])
def test_rank_genes_groups_wilcoxon_subset_matches_scanpy(
    groups, reference, tie_correct, pre_load
):
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=8, n_centers=5, n_observations=200)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")
    adata_cpu = adata_gpu.copy()

    rsc.tl.rank_genes_groups(
        adata_gpu,
        "blobs",
        method="wilcoxon",
        groups=groups,
        reference=reference,
        use_raw=False,
        tie_correct=tie_correct,
        pre_load=pre_load,
    )
    sc.tl.rank_genes_groups(
        adata_cpu,
        "blobs",
        method="wilcoxon",
        groups=groups,
        reference=reference,
        use_raw=False,
        tie_correct=tie_correct,
    )

    gpu_result = adata_gpu.uns["rank_genes_groups"]
    cpu_result = adata_cpu.uns["rank_genes_groups"]

    assert gpu_result["names"].dtype.names == cpu_result["names"].dtype.names
    for group in gpu_result["names"].dtype.names:
        gpu_names = list(gpu_result["names"][group])
        cpu_names = list(cpu_result["names"][group])
        for field in ("scores", "logfoldchanges", "pvals", "pvals_adj"):
            gpu_map = dict(
                zip(gpu_names, np.asarray(gpu_result[field][group], dtype=float))
            )
            cpu_map = dict(
                zip(cpu_names, np.asarray(cpu_result[field][group], dtype=float))
            )
            for gene, gpu_val in gpu_map.items():
                np.testing.assert_allclose(
                    gpu_val,
                    cpu_map[gene],
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"{field} mismatch for gene {gene} group {group}",
                )


@pytest.mark.parametrize(
    "reference_before,reference_after",
    [("rest", "rest"), ("1", "One")],
)
def test_rank_genes_groups_wilcoxon_with_renamed_categories(
    reference_before, reference_after
):
    """Test with renamed category labels."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=4, n_centers=3, n_observations=200)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")

    # First run with original category names
    rsc.tl.rank_genes_groups(
        adata, "blobs", method="wilcoxon", reference=reference_before
    )
    names = adata.uns["rank_genes_groups"]["names"]
    expected_groups = ("0", "1", "2") if reference_before == "rest" else ("0", "2")
    assert names.dtype.names == expected_groups
    first_run = tuple(names[0])

    adata.rename_categories("blobs", ["Zero", "One", "Two"])
    assert tuple(adata.uns["rank_genes_groups"]["names"][0]) == first_run

    # Second run with renamed category names
    rsc.tl.rank_genes_groups(
        adata, "blobs", method="wilcoxon", reference=reference_after
    )
    renamed_names = adata.uns["rank_genes_groups"]["names"]
    assert tuple(renamed_names[0]) == first_run
    expected_renamed = (
        ("Zero", "One", "Two") if reference_after == "rest" else ("Zero", "Two")
    )
    assert renamed_names.dtype.names == expected_renamed


@pytest.mark.parametrize("reference", ["rest", "1"])
def test_rank_genes_groups_wilcoxon_with_unsorted_groups(reference):
    """Test that group order doesn't affect results."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=6, n_centers=4, n_observations=180)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")
    bdata = adata.copy()

    groups = ["0", "1", "2", "3"] if reference != "rest" else ["0", "2", "3"]
    groups_reversed = list(reversed(groups))

    rsc.tl.rank_genes_groups(
        adata, "blobs", method="wilcoxon", groups=groups, reference=reference
    )
    rsc.tl.rank_genes_groups(
        bdata, "blobs", method="wilcoxon", groups=groups_reversed, reference=reference
    )

    expected_groups = {g for g in groups if g != reference}
    assert set(adata.uns["rank_genes_groups"]["names"].dtype.names) == expected_groups
    assert set(bdata.uns["rank_genes_groups"]["names"].dtype.names) == expected_groups

    # Pick a group that's not the reference for comparison
    test_group = "3" if reference != "3" else "0"
    for field in ("scores", "logfoldchanges", "pvals", "pvals_adj"):
        np.testing.assert_allclose(
            np.asarray(adata.uns["rank_genes_groups"][field][test_group], dtype=float),
            np.asarray(bdata.uns["rank_genes_groups"][field][test_group], dtype=float),
            rtol=1e-13,
            atol=1e-15,
            equal_nan=True,
        )

    assert tuple(adata.uns["rank_genes_groups"]["names"][test_group]) == tuple(
        bdata.uns["rank_genes_groups"]["names"][test_group]
    )


@pytest.mark.parametrize("reference", ["rest", "1"])
@pytest.mark.parametrize("pre_load", [True, False])
def test_rank_genes_groups_wilcoxon_pts(reference, pre_load):
    """Test that pts (fraction of cells expressing) is computed correctly."""
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=6, n_centers=3, n_observations=200)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")
    adata_cpu = adata_gpu.copy()

    # Run with pts=True
    rsc.tl.rank_genes_groups(
        adata_gpu,
        "blobs",
        method="wilcoxon",
        use_raw=False,
        pts=True,
        tie_correct=False,
        reference=reference,
        pre_load=pre_load,
    )
    sc.tl.rank_genes_groups(
        adata_cpu,
        "blobs",
        method="wilcoxon",
        use_raw=False,
        pts=True,
        tie_correct=False,
        reference=reference,
    )

    gpu_result = adata_gpu.uns["rank_genes_groups"]
    cpu_result = adata_cpu.uns["rank_genes_groups"]

    # Check pts DataFrame exists and has correct structure
    assert "pts" in gpu_result
    assert "pts" in cpu_result

    # Check pts values match scanpy
    gpu_pts = gpu_result["pts"]
    cpu_pts = cpu_result["pts"]
    assert list(gpu_pts.columns) == list(cpu_pts.columns)
    assert list(gpu_pts.index) == list(cpu_pts.index)

    for col in gpu_pts.columns:
        np.testing.assert_allclose(
            gpu_pts[col].values, cpu_pts[col].values, rtol=1e-13, atol=1e-15
        )

    # pts_rest only exists when reference='rest'
    if reference == "rest":
        assert "pts_rest" in gpu_result
        assert "pts_rest" in cpu_result

        gpu_pts_rest = gpu_result["pts_rest"]
        cpu_pts_rest = cpu_result["pts_rest"]

        for col in gpu_pts_rest.columns:
            np.testing.assert_allclose(
                gpu_pts_rest[col].values,
                cpu_pts_rest[col].values,
                rtol=1e-13,
                atol=1e-15,
            )


# ============================================================================
# Ground-truth validation against scipy.stats.mannwhitneyu
# ============================================================================


def _make_perturbation_adata(
    n_control: int = 200,
    n_treatment: int = 150,
    n_genes: int = 500,
    n_de_genes: int = 50,
    seed: int = 42,
):
    """Two-group perturbation AnnData with count-based log1p data (many ties)."""
    rng = np.random.default_rng(seed)
    n_cells = n_control + n_treatment

    gene_means = rng.gamma(shape=2.0, scale=5.0, size=n_genes)
    X = rng.poisson(lam=gene_means[None, :], size=(n_cells, n_genes)).astype(np.float32)
    for g in range(n_de_genes):
        X[n_control:, g] = rng.poisson(
            lam=gene_means[g] * 1.5, size=n_treatment
        ).astype(np.float32)

    obs = pd.DataFrame(
        {
            "group": pd.Categorical(
                ["control"] * n_control + ["treatment"] * n_treatment,
                categories=["control", "treatment"],
            ),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = sc.AnnData(X=X, obs=obs, var=var)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def _scipy_mannwhitneyu_pvals(adata, *, group, reference, groupby="group"):
    """Per-gene two-sided Mann-Whitney U p-values via scipy (ground truth)."""
    X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    X = X.astype(np.float64)
    mask_g = (adata.obs[groupby] == group).values
    mask_r = (adata.obs[groupby] == reference).values
    return np.array(
        [
            mannwhitneyu(X[mask_g, i], X[mask_r, i], alternative="two-sided").pvalue
            for i in range(X.shape[1])
        ]
    )


@pytest.fixture
def perturbation_adata():
    return _make_perturbation_adata()


class TestWilcoxonAgainstScipy:
    """Validate rsc wilcoxon p-values against scipy.stats.mannwhitneyu."""

    def test_with_continuity_matches_scipy(self, perturbation_adata):
        """use_continuity + tie_correct matches scipy mannwhitneyu to machine eps."""
        adata = perturbation_adata.copy()
        rsc.tl.rank_genes_groups(
            adata,
            "group",
            groups=["treatment"],
            reference="control",
            method="wilcoxon",
            use_raw=False,
            tie_correct=True,
            use_continuity=True,
        )

        rsc_df = (
            sc.get.rank_genes_groups_df(adata, group="treatment")
            .sort_values("names")
            .reset_index(drop=True)
        )
        scipy_pvals = _scipy_mannwhitneyu_pvals(
            perturbation_adata, group="treatment", reference="control"
        )
        # Align scipy pvals to the same gene order
        gene_to_idx = {g: i for i, g in enumerate(perturbation_adata.var_names)}
        scipy_sorted = np.array([scipy_pvals[gene_to_idx[g]] for g in rsc_df["names"]])

        np.testing.assert_allclose(
            rsc_df["pvals"].values, scipy_sorted, rtol=1e-13, atol=1e-15
        )

    def test_without_continuity_close_to_scipy(self, perturbation_adata):
        """Without continuity correction the gap is only the 0.5 adjustment term."""
        adata = perturbation_adata.copy()
        rsc.tl.rank_genes_groups(
            adata,
            "group",
            groups=["treatment"],
            reference="control",
            method="wilcoxon",
            use_raw=False,
            tie_correct=True,
            use_continuity=False,
        )

        rsc_df = (
            sc.get.rank_genes_groups_df(adata, group="treatment")
            .sort_values("names")
            .reset_index(drop=True)
        )
        scipy_pvals = _scipy_mannwhitneyu_pvals(
            perturbation_adata, group="treatment", reference="control"
        )
        gene_to_idx = {g: i for i, g in enumerate(perturbation_adata.var_names)}
        scipy_sorted = np.array([scipy_pvals[gene_to_idx[g]] for g in rsc_df["names"]])

        np.testing.assert_allclose(
            rsc_df["pvals"].values, scipy_sorted, rtol=1e-2, atol=1e-15
        )

    @pytest.mark.parametrize("sparse", [True, False])
    def test_sparse_matches_dense(self, perturbation_adata, sparse):
        """Sparse and dense wilcoxon give identical results."""
        adata_dense = perturbation_adata.copy()
        adata_sparse = perturbation_adata.copy()
        adata_sparse.X = sp.csr_matrix(adata_sparse.X)

        kw = {
            "groupby": "group",
            "groups": ["treatment"],
            "reference": "control",
            "method": "wilcoxon",
            "use_raw": False,
            "tie_correct": True,
        }
        rsc.tl.rank_genes_groups(adata_dense, **kw)
        rsc.tl.rank_genes_groups(adata_sparse, **kw)

        dense_df = (
            sc.get.rank_genes_groups_df(adata_dense, group="treatment")
            .sort_values("names")
            .reset_index(drop=True)
        )
        sparse_df = (
            sc.get.rank_genes_groups_df(adata_sparse, group="treatment")
            .sort_values("names")
            .reset_index(drop=True)
        )
        np.testing.assert_array_equal(
            dense_df["scores"].values, sparse_df["scores"].values
        )
        np.testing.assert_array_equal(
            dense_df["pvals"].values, sparse_df["pvals"].values
        )


# ============================================================================
# Tests for ranking and tie correction kernels (edge cases from scipy)
# ============================================================================


class TestRankingKernel:
    """Tests for _average_ranks based on scipy.stats.rankdata edge cases."""

    @pytest.fixture
    def average_ranks(self):
        """Import the ranking function."""
        from rapids_singlecell.tools._rank_genes_groups._wilcoxon import (
            _average_ranks,
        )

        return _average_ranks

    @staticmethod
    def _to_gpu(values):
        """Convert 1D values to GPU column matrix with F-order."""
        arr = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        return cp.asarray(arr, order="F")

    def test_basic_ranking(self, average_ranks):
        """Test basic average ranking on simple data."""
        values = [3.0, 1.0, 2.0]
        result = average_ranks(self._to_gpu(values))
        expected = rankdata(values, method="average")
        np.testing.assert_allclose(result.get().flatten(), expected)

    def test_all_ties(self, average_ranks):
        """All identical values should get the average rank."""
        values = [5.0, 5.0, 5.0, 5.0]
        result = average_ranks(self._to_gpu(values))
        expected = rankdata(values, method="average")
        np.testing.assert_allclose(result.get().flatten(), expected)

    def test_no_ties(self, average_ranks):
        """All unique values should get sequential ranks."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = average_ranks(self._to_gpu(values))
        expected = rankdata(values, method="average")
        np.testing.assert_allclose(result.get().flatten(), expected)

    def test_mixed_ties(self, average_ranks):
        """Mix of ties and unique values."""
        values = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0]
        result = average_ranks(self._to_gpu(values))
        expected = rankdata(values, method="average")
        np.testing.assert_allclose(result.get().flatten(), expected)

    def test_negative_values(self, average_ranks):
        """Test with negative values."""
        values = [-3.0, -1.0, -2.0, 0.0, 1.0]
        result = average_ranks(self._to_gpu(values))
        expected = rankdata(values, method="average")
        np.testing.assert_allclose(result.get().flatten(), expected)

    def test_single_element(self, average_ranks):
        """Single element should have rank 1."""
        values = [42.0]
        result = average_ranks(self._to_gpu(values))
        np.testing.assert_allclose(result.get().flatten(), [1.0])

    def test_two_elements_tied(self, average_ranks):
        """Two tied elements should both have rank 1.5."""
        values = [7.0, 7.0]
        result = average_ranks(self._to_gpu(values))
        np.testing.assert_allclose(result.get().flatten(), [1.5, 1.5])

    def test_multiple_columns(self, average_ranks):
        """Test ranking across multiple columns independently."""
        col0 = [3.0, 1.0, 2.0]
        col1 = [1.0, 1.0, 2.0]
        data = np.column_stack([col0, col1]).astype(np.float64)
        result = average_ranks(cp.asarray(data, order="F"))

        np.testing.assert_allclose(result.get()[:, 0], rankdata(col0, method="average"))
        np.testing.assert_allclose(result.get()[:, 1], rankdata(col1, method="average"))


class TestTieCorrectionKernel:
    """Tests for _tie_correction based on scipy.stats.tiecorrect edge cases."""

    @pytest.fixture
    def tie_correction(self):
        """Import the tie correction function and ranking function."""
        from rapids_singlecell.tools._rank_genes_groups._wilcoxon import (
            _average_ranks,
            _tie_correction,
        )

        return _tie_correction, _average_ranks

    @staticmethod
    def _to_gpu(values):
        """Convert 1D values to GPU column matrix with F-order."""
        arr = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        return cp.asarray(arr, order="F")

    def test_no_ties(self, tie_correction):
        """No ties should give correction factor 1.0."""
        _tie_correction, _average_ranks = tie_correction

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        _, sorted_vals = _average_ranks(self._to_gpu(values), return_sorted=True)
        result = _tie_correction(sorted_vals)

        expected = tiecorrect(rankdata(values))
        np.testing.assert_allclose(result.get()[0], expected, rtol=1e-10)

    def test_all_ties(self, tie_correction):
        """All tied values should give correction factor 0.0."""
        _tie_correction, _average_ranks = tie_correction

        values = [5.0, 5.0, 5.0, 5.0]
        _, sorted_vals = _average_ranks(self._to_gpu(values), return_sorted=True)
        result = _tie_correction(sorted_vals)

        expected = tiecorrect(rankdata(values))
        np.testing.assert_allclose(result.get()[0], expected, rtol=1e-10)

    def test_mixed_ties(self, tie_correction):
        """Mix of ties should give intermediate correction factor."""
        _tie_correction, _average_ranks = tie_correction

        values = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0]
        _, sorted_vals = _average_ranks(self._to_gpu(values), return_sorted=True)
        result = _tie_correction(sorted_vals)

        expected = tiecorrect(rankdata(values))
        np.testing.assert_allclose(result.get()[0], expected, rtol=1e-10)

    def test_two_elements_tied(self, tie_correction):
        """Two tied elements."""
        _tie_correction, _average_ranks = tie_correction

        values = [7.0, 7.0]
        _, sorted_vals = _average_ranks(self._to_gpu(values), return_sorted=True)
        result = _tie_correction(sorted_vals)

        expected = tiecorrect(rankdata(values))
        np.testing.assert_allclose(result.get()[0], expected, rtol=1e-10)

    def test_single_element(self, tie_correction):
        """Single element should give correction factor 1.0."""
        _tie_correction, _average_ranks = tie_correction

        values = [42.0]
        _, sorted_vals = _average_ranks(self._to_gpu(values), return_sorted=True)
        result = _tie_correction(sorted_vals)

        # Single element: n^3 - n = 0, so formula gives 1.0
        np.testing.assert_allclose(result.get()[0], 1.0, rtol=1e-10)

    def test_multiple_columns(self, tie_correction):
        """Test tie correction across multiple columns independently."""
        _tie_correction, _average_ranks = tie_correction

        col0 = [1.0, 2.0, 3.0]  # No ties
        col1 = [5.0, 5.0, 5.0]  # All ties
        data = np.column_stack([col0, col1]).astype(np.float64)
        _, sorted_vals = _average_ranks(cp.asarray(data, order="F"), return_sorted=True)
        result = _tie_correction(sorted_vals)

        np.testing.assert_allclose(
            result.get()[0], tiecorrect(rankdata(col0)), rtol=1e-10
        )
        np.testing.assert_allclose(
            result.get()[1], tiecorrect(rankdata(col1)), rtol=1e-10
        )

    def test_large_tie_groups(self, tie_correction):
        """Test with large tie groups."""
        _tie_correction, _average_ranks = tie_correction

        # 50 values of 1, 50 values of 2 (non-multiple of 32 to test warp handling)
        values = [1.0] * 50 + [2.0] * 50
        _, sorted_vals = _average_ranks(self._to_gpu(values), return_sorted=True)
        result = _tie_correction(sorted_vals)

        expected = tiecorrect(rankdata(values))
        np.testing.assert_allclose(result.get()[0], expected, rtol=1e-10)
