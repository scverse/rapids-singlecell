from __future__ import annotations

import numpy as np
import pytest
import scanpy as sc
import scipy.sparse as sp
from scanpy.datasets import pbmc68k_reduced
from scipy import stats

import rapids_singlecell as rsc


@pytest.mark.parametrize("reference", ["rest", "1"])
@pytest.mark.parametrize("method", ["t-test", "t-test_overestim_var"])
@pytest.mark.parametrize("sparse", [True, False])
def test_rank_genes_groups_ttest_matches_scanpy(reference, method, sparse):
    """Test t-test methods match scanpy output across configurations."""
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=6, n_centers=3, n_observations=200)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")

    if sparse:
        adata_gpu.X = sp.csr_matrix(adata_gpu.X)

    adata_cpu = adata_gpu.copy()

    rsc.tl.rank_genes_groups(
        adata_gpu,
        "blobs",
        method=method,
        use_raw=False,
        n_genes=3,
        reference=reference,
        corr_method="benjamini-hochberg",
    )
    sc.tl.rank_genes_groups(
        adata_cpu,
        "blobs",
        method=method,
        use_raw=False,
        n_genes=3,
        reference=reference,
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
            np.testing.assert_allclose(
                gpu_values, cpu_values, rtol=1e-13, atol=1e-15, equal_nan=True
            )

    params = gpu_result["params"]
    assert params["use_raw"] is False
    assert params["corr_method"] == "benjamini-hochberg"
    assert params["layer"] is None
    assert params["reference"] == reference
    assert params["method"] == method


@pytest.mark.parametrize("reference", ["rest", "1"])
@pytest.mark.parametrize("method", ["t-test", "t-test_overestim_var"])
def test_rank_genes_groups_ttest_honors_layer_and_use_raw(reference, method):
    """Test that layer parameter is respected."""
    np.random.seed(42)
    base = sc.datasets.blobs(n_variables=5, n_centers=3, n_observations=150)
    base.obs["blobs"] = base.obs["blobs"].astype("category")
    base.layers["signal"] = base.X.copy()

    ref_adata = base.copy()
    rsc.tl.rank_genes_groups(
        ref_adata, "blobs", method=method, use_raw=False, reference=reference
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
        method=method,
        layer="signal",
        use_raw=False,
        reference=reference,
    )
    layered_names = layered.uns["rank_genes_groups"]["names"].copy()

    no_layer = base.copy()
    no_layer.X = perturbed_matrix
    rsc.tl.rank_genes_groups(
        no_layer, "blobs", method=method, use_raw=False, reference=reference
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
@pytest.mark.parametrize("method", ["t-test", "t-test_overestim_var"])
def test_rank_genes_groups_ttest_subset_and_bonferroni(reference, method):
    """Test group subsetting and bonferroni correction."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=5, n_centers=4, n_observations=150)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")

    groups = ["0", "1", "2"] if reference != "rest" else ["0", "2"]

    rsc.tl.rank_genes_groups(
        adata,
        "blobs",
        method=method,
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
@pytest.mark.parametrize("method", ["t-test", "t-test_overestim_var"])
def test_rank_genes_groups_ttest_subset_matches_scanpy(method, groups, reference):
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=8, n_centers=5, n_observations=200)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")
    adata_cpu = adata_gpu.copy()

    rsc.tl.rank_genes_groups(
        adata_gpu,
        "blobs",
        method=method,
        groups=groups,
        reference=reference,
        use_raw=False,
    )
    sc.tl.rank_genes_groups(
        adata_cpu,
        "blobs",
        method=method,
        groups=groups,
        reference=reference,
        use_raw=False,
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
@pytest.mark.parametrize("method", ["t-test", "t-test_overestim_var"])
def test_rank_genes_groups_ttest_with_renamed_categories(
    reference_before, reference_after, method
):
    """Test with renamed category labels."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=4, n_centers=3, n_observations=200)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")

    # First run with original category names
    rsc.tl.rank_genes_groups(adata, "blobs", method=method, reference=reference_before)
    names = adata.uns["rank_genes_groups"]["names"]
    expected_groups = ("0", "1", "2") if reference_before == "rest" else ("0", "2")
    assert names.dtype.names == expected_groups
    first_run = tuple(names[0])

    adata.rename_categories("blobs", ["Zero", "One", "Two"])
    assert tuple(adata.uns["rank_genes_groups"]["names"][0]) == first_run

    # Second run with renamed category names
    rsc.tl.rank_genes_groups(adata, "blobs", method=method, reference=reference_after)
    renamed_names = adata.uns["rank_genes_groups"]["names"]
    assert tuple(renamed_names[0]) == first_run
    expected_renamed = (
        ("Zero", "One", "Two") if reference_after == "rest" else ("Zero", "Two")
    )
    assert renamed_names.dtype.names == expected_renamed


@pytest.mark.parametrize("reference", ["rest", "1"])
@pytest.mark.parametrize("method", ["t-test", "t-test_overestim_var"])
def test_rank_genes_groups_ttest_with_unsorted_groups(reference, method):
    """Test that group order doesn't affect results."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=6, n_centers=4, n_observations=180)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")
    bdata = adata.copy()

    groups = ["0", "1", "2", "3"] if reference != "rest" else ["0", "2", "3"]
    groups_reversed = list(reversed(groups))

    rsc.tl.rank_genes_groups(
        adata, "blobs", method=method, groups=groups, reference=reference
    )
    rsc.tl.rank_genes_groups(
        bdata, "blobs", method=method, groups=groups_reversed, reference=reference
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
@pytest.mark.parametrize("method", ["t-test", "t-test_overestim_var"])
def test_rank_genes_groups_ttest_pts(reference, method):
    """Test that pts (fraction of cells expressing) is computed correctly."""
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=6, n_centers=3, n_observations=200)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")
    adata_cpu = adata_gpu.copy()

    # Run with pts=True
    rsc.tl.rank_genes_groups(
        adata_gpu,
        "blobs",
        method=method,
        use_raw=False,
        pts=True,
        reference=reference,
    )
    sc.tl.rank_genes_groups(
        adata_cpu,
        "blobs",
        method=method,
        use_raw=False,
        pts=True,
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


def test_rank_genes_groups_ttest_direct_scipy():
    """Test t-test scores directly against scipy.stats.ttest_ind on two matrices.

    Creates a simple two-group dataset and compares rapids_singlecell t-test
    directly against scipy.stats.ttest_ind without intermediate statistics.
    """
    import anndata as ad

    np.random.seed(42)
    n_group1, n_group2, n_genes = 50, 60, 20

    # Create two groups with different distributions
    X_group1 = np.random.randn(n_group1, n_genes).astype(np.float64)
    X_group2 = np.random.randn(n_group2, n_genes).astype(np.float64) + 0.5  # shifted

    # Combine into AnnData
    X = np.vstack([X_group1, X_group2])
    obs = {"group": ["A"] * n_group1 + ["B"] * n_group2}
    adata = ad.AnnData(X=X, obs=obs)
    adata.obs["group"] = adata.obs["group"].astype("category")

    # Run rapids_singlecell t-test (group A vs B as reference)
    rsc.tl.rank_genes_groups(
        adata, "group", method="t-test", reference="B", use_raw=False
    )

    # Get rsc scores for group A
    rsc_names = list(adata.uns["rank_genes_groups"]["names"]["A"])
    rsc_scores = np.asarray(
        adata.uns["rank_genes_groups"]["scores"]["A"], dtype=np.float64
    )
    rsc_score_map = dict(zip(rsc_names, rsc_scores))

    # Compute scipy t-test directly on the two matrices
    scipy_scores, _ = stats.ttest_ind(X_group1, X_group2, equal_var=False)

    # Compare scores for each gene
    # Scores are stored as float32 in the structured array, so rtol ~ float32 eps
    for i in range(n_genes):
        gene = adata.var_names[i]
        np.testing.assert_allclose(
            rsc_score_map[gene],
            scipy_scores[i],
            rtol=1e-7,
            atol=1e-15,
            err_msg=f"Score mismatch for gene {gene}",
        )


def test_rank_genes_groups_ttest_matches_scipy():
    """Test that t-test scores match scipy computation directly.

    This test verifies that our variance clipping fix produces correct results
    by comparing against scipy.stats.ttest_ind_from_stats with properly computed
    (non-negative) variances. Uses real pbmc68k_reduced dataset at float64 precision.
    """
    adata = pbmc68k_reduced()
    # Convert to float64 for maximum precision in comparison
    adata.X = adata.X.astype(np.float64)

    # Run rapids_singlecell t-test
    rsc.tl.rank_genes_groups(adata, "bulk_labels", method="t-test", use_raw=False)

    # Compute scipy t-test directly for each group vs rest
    groups = adata.obs["bulk_labels"].cat.categories
    var_names = adata.var_names

    for group in groups:
        mask = (adata.obs["bulk_labels"] == group).values
        X_group = adata.X[mask].astype(np.float64)
        X_rest = adata.X[~mask].astype(np.float64)

        # Compute stats with numpy (guaranteed non-negative variance)
        mean_group = X_group.mean(axis=0)
        var_group = np.maximum(X_group.var(axis=0, ddof=1), 0)
        n_group = mask.sum()

        mean_rest = X_rest.mean(axis=0)
        var_rest = np.maximum(X_rest.var(axis=0, ddof=1), 0)
        n_rest = (~mask).sum()

        # Compute scipy t-test for all genes
        scipy_scores = np.zeros(len(var_names), dtype=np.float64)
        for i in range(len(var_names)):
            with np.errstate(invalid="ignore"):
                score, _ = stats.ttest_ind_from_stats(
                    mean1=mean_group[i],
                    std1=np.sqrt(var_group[i]),
                    nobs1=n_group,
                    mean2=mean_rest[i],
                    std2=np.sqrt(var_rest[i]),
                    nobs2=n_rest,
                    equal_var=False,
                )
            scipy_scores[i] = 0 if np.isnan(score) else score

        # Get rapids_singlecell scores (need to map by gene name since order differs)
        rsc_names = list(adata.uns["rank_genes_groups"]["names"][group])
        rsc_scores = np.asarray(
            adata.uns["rank_genes_groups"]["scores"][group], dtype=np.float64
        )
        rsc_score_map = dict(zip(rsc_names, rsc_scores))

        # Compare scores for each gene
        # Gap is from GPU vs CPU mean/var computation (~float32 intermediates)
        for i, gene in enumerate(var_names):
            np.testing.assert_allclose(
                rsc_score_map[gene],
                scipy_scores[i],
                rtol=1e-6,
                atol=1e-15,
                err_msg=f"Score mismatch for gene {gene} in group {group}",
            )


@pytest.mark.parametrize("method", ["t-test", "t-test_overestim_var"])
def test_rank_genes_groups_ttest_mask_var_array(method):
    """Test mask_var parameter with boolean array."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=10, n_centers=3, n_observations=150)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")

    # Create mask to select only first 5 genes
    mask = np.array([True] * 5 + [False] * 5)

    # Run with mask
    rsc.tl.rank_genes_groups(
        adata, "blobs", method=method, mask_var=mask, use_raw=False
    )

    result = adata.uns["rank_genes_groups"]

    # Check that only masked genes appear in results
    for group in result["names"].dtype.names:
        genes = list(result["names"][group])
        assert len(genes) == 5
        # All genes should be from the first 5 (Gene0-Gene4)
        for gene in genes:
            gene_idx = int(gene.replace("Gene", ""))
            assert gene_idx < 5, f"Gene {gene} should not be in results"


@pytest.mark.parametrize("method", ["t-test", "t-test_overestim_var"])
def test_rank_genes_groups_ttest_mask_var_string(method):
    """Test mask_var parameter with string key in adata.var."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=10, n_centers=3, n_observations=150)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")

    # Add mask column to adata.var
    adata.var["highly_variable"] = [True] * 6 + [False] * 4

    # Run with mask string key
    rsc.tl.rank_genes_groups(
        adata, "blobs", method=method, mask_var="highly_variable", use_raw=False
    )

    result = adata.uns["rank_genes_groups"]

    # Check that only masked genes appear in results
    for group in result["names"].dtype.names:
        genes = list(result["names"][group])
        assert len(genes) == 6
        for gene in genes:
            gene_idx = int(gene.replace("Gene", ""))
            assert gene_idx < 6, f"Gene {gene} should not be in results"


@pytest.mark.parametrize("method", ["t-test", "t-test_overestim_var"])
def test_rank_genes_groups_ttest_mask_var_matches_scanpy(method):
    """Test that mask_var results match scanpy."""
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=8, n_centers=3, n_observations=150)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")
    adata_cpu = adata_gpu.copy()

    mask = np.array([True, False, True, False, True, True, False, True])

    rsc.tl.rank_genes_groups(
        adata_gpu, "blobs", method=method, mask_var=mask, use_raw=False
    )
    sc.tl.rank_genes_groups(
        adata_cpu, "blobs", method=method, mask_var=mask, use_raw=False
    )

    gpu_result = adata_gpu.uns["rank_genes_groups"]
    cpu_result = adata_cpu.uns["rank_genes_groups"]

    # Compare names
    assert gpu_result["names"].dtype.names == cpu_result["names"].dtype.names
    for group in gpu_result["names"].dtype.names:
        assert list(gpu_result["names"][group]) == list(cpu_result["names"][group])

    # Compare scores
    for group in gpu_result["scores"].dtype.names:
        gpu_values = np.asarray(gpu_result["scores"][group], dtype=float)
        cpu_values = np.asarray(cpu_result["scores"][group], dtype=float)
        np.testing.assert_allclose(gpu_values, cpu_values, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("method", ["t-test", "t-test_overestim_var"])
def test_rank_genes_groups_ttest_rankby_abs(method):
    """Test rankby_abs parameter."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=6, n_centers=3, n_observations=200)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")
    adata_abs = adata.copy()

    # Run without rankby_abs
    rsc.tl.rank_genes_groups(adata, "blobs", method=method, use_raw=False)

    # Run with rankby_abs
    rsc.tl.rank_genes_groups(
        adata_abs, "blobs", method=method, rankby_abs=True, use_raw=False
    )

    # When rankby_abs=True, genes are ranked by absolute score
    # So a gene with score -5 should rank higher than a gene with score 2
    for group in adata.uns["rank_genes_groups"]["scores"].dtype.names:
        abs_scores = np.asarray(
            adata_abs.uns["rank_genes_groups"]["scores"][group], dtype=float
        )
        # Scores themselves are not absolute, but ordering is by absolute value
        # So absolute values should be monotonically decreasing
        assert np.all(np.diff(np.abs(abs_scores)) <= 1e-10)


@pytest.mark.parametrize("method", ["t-test", "t-test_overestim_var"])
def test_rank_genes_groups_ttest_key_added(method):
    """Test key_added parameter."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=6, n_centers=3, n_observations=200)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")

    custom_key = "my_custom_key"

    rsc.tl.rank_genes_groups(
        adata, "blobs", method=method, key_added=custom_key, use_raw=False
    )

    # Check that results are stored under custom key
    assert custom_key in adata.uns
    assert "rank_genes_groups" not in adata.uns

    # Check structure is correct
    result = adata.uns[custom_key]
    assert "names" in result
    assert "scores" in result
    assert "pvals" in result
    assert "pvals_adj" in result
    assert "logfoldchanges" in result
    assert "params" in result
