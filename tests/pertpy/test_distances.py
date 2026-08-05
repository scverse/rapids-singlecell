from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.spatial.distance import cdist

from rapids_singlecell.pertpy_gpu import Distance, MeanVar


@pytest.fixture
def small_adata() -> AnnData:
    rng = np.random.default_rng(0)
    n_groups = 3
    cells_per_group = 4
    n_features = 5
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding, obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)
    return adata


# ============================================================================
# Tests for Distance class API
# ============================================================================


def test_distance_class_initialization() -> None:
    """Test Distance class can be initialized with different metrics."""
    distance = Distance(metric="edistance")
    assert distance.metric == "edistance"
    assert distance.obsm_key == "X_pca"

    distance_custom = Distance(metric="edistance", obsm_key="X_custom")
    assert distance_custom.obsm_key == "X_custom"


def test_unused_categories_dropped() -> None:
    """A zero-cell (unused) category in groupby is dropped, not kept as a NaN
    group (shared subsetting on BaseMetric handles this for all metrics)."""
    rng = np.random.default_rng(0)
    cpg, nf = 15, 5
    X = np.vstack(
        [(rng.normal(size=(cpg, nf)) + i).astype(np.float32) for i in range(3)]
    )
    labels = [c for c in ["g0", "g1", "g2"] for _ in range(cpg)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(labels, categories=["g0", "g1", "g2", "gZ"])}
    )
    adata = AnnData(X.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(X, dtype=cp.float32)
    df = Distance(metric="edistance").pairwise(adata, groupby="group")
    assert list(df.index) == ["g0", "g1", "g2"]  # unused 'gZ' dropped
    assert np.all(np.isfinite(df.values))


def test_distance_class_invalid_metric() -> None:
    """Test Distance class raises error for unsupported metrics."""
    with pytest.raises(ValueError, match="Unknown metric"):
        Distance(metric="invalid_metric")


def test_distance_class_pairwise_with_bootstrap(small_adata: AnnData) -> None:
    """Test Distance.pairwise() with bootstrap."""
    distance = Distance(metric="edistance")
    distances, distances_var = distance.pairwise(
        small_adata,
        groupby="group",
        bootstrap=True,
        n_bootstrap=10,
        random_state=42,
    )

    assert isinstance(distances, pd.DataFrame)
    assert isinstance(distances_var, pd.DataFrame)
    assert distances.shape == distances_var.shape
    assert np.all(distances_var.values >= 0)


def test_distance_class_onesided_distances(small_adata: AnnData) -> None:
    """Test Distance.onesided_distances() method."""
    distance = Distance(metric="edistance")
    result = distance.onesided_distances(
        small_adata, groupby="group", selected_group="g0"
    )

    assert isinstance(result, pd.Series)
    assert len(result) == 3  # 3 groups total
    assert "g0" in result.index
    assert "g1" in result.index
    assert "g2" in result.index


def test_distance_class_onesided_matches_pairwise(small_adata: AnnData) -> None:
    """Test onesided_distances matches the corresponding row from pairwise."""
    distance = Distance(metric="edistance")

    # Get pairwise distances
    pairwise_df = distance.pairwise(small_adata, groupby="group")

    # Get onesided distances for each group and verify it matches pairwise row
    for group in ["g0", "g1", "g2"]:
        onesided = distance.onesided_distances(
            small_adata, groupby="group", selected_group=group
        )
        # Should match the row from pairwise matrix
        np.testing.assert_allclose(
            onesided.values, pairwise_df.loc[group].values, atol=1e-6
        )
        # Self-distance should be 0
        assert onesided.loc[group] == pytest.approx(0.0, abs=1e-6)


def test_distance_class_onesided_multiple_controls(small_adata: AnnData) -> None:
    """Test onesided_distances with multiple selected groups."""
    distance = Distance(metric="edistance")

    # Multiple controls in a single call
    result = distance.onesided_distances(
        small_adata, groupby="group", selected_group=["g0", "g1"]
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["g0", "g1"]
    assert len(result) == 3

    # Each column should match the corresponding row from pairwise
    pairwise_df = distance.pairwise(small_adata, groupby="group")
    for group in ["g0", "g1"]:
        np.testing.assert_allclose(
            result[group].values, pairwise_df.loc[group].values, atol=1e-7
        )


def test_distance_class_onesided_invalid_group(small_adata: AnnData) -> None:
    """Test Distance.onesided_distances() raises error for invalid group."""
    distance = Distance(metric="edistance")
    with pytest.raises(ValueError, match="not found"):
        distance.onesided_distances(
            small_adata, groupby="group", selected_group="invalid"
        )


def test_distance_class_onesided_bootstrap(small_adata: AnnData) -> None:
    """Test Distance.onesided_distances() with bootstrap returns tuple."""
    distance = Distance(metric="edistance")
    result = distance.onesided_distances(
        small_adata,
        groupby="group",
        selected_group="g0",
        bootstrap=True,
        n_bootstrap=10,
        random_state=42,
    )

    # Should return tuple of (distances, distances_var)
    assert isinstance(result, tuple)
    assert len(result) == 2
    distances, distances_var = result

    assert isinstance(distances, pd.Series)
    assert isinstance(distances_var, pd.Series)
    assert len(distances) == 3
    assert len(distances_var) == 3

    # Self-distance variance should be 0
    assert distances.loc["g0"] == pytest.approx(0.0, abs=1e-6)
    assert distances_var.loc["g0"] == pytest.approx(0.0, abs=1e-6)

    # Non-self variances should be positive
    assert distances_var.loc["g1"] > 0
    assert distances_var.loc["g2"] > 0


def test_distance_class_onesided_bootstrap_matches_pairwise(
    small_adata: AnnData,
) -> None:
    """Test onesided_distances with bootstrap matches pairwise bootstrap."""
    distance = Distance(metric="edistance")

    # Get pairwise with bootstrap
    pairwise_df, pairwise_var_df = distance.pairwise(
        small_adata,
        groupby="group",
        bootstrap=True,
        n_bootstrap=20,
        random_state=42,
    )

    # Get onesided with bootstrap
    onesided, onesided_var = distance.onesided_distances(
        small_adata,
        groupby="group",
        selected_group="g0",
        bootstrap=True,
        n_bootstrap=20,
        random_state=42,
    )

    # Should match the corresponding row from pairwise
    np.testing.assert_allclose(onesided.values, pairwise_df.loc["g0"].values, atol=1e-7)
    np.testing.assert_allclose(
        onesided_var.values, pairwise_var_df.loc["g0"].values, atol=1e-7
    )


def test_distance_class_repr() -> None:
    """Test Distance.__repr__() method."""
    distance = Distance(metric="edistance", obsm_key="X_custom")
    repr_str = repr(distance)
    assert "Distance" in repr_str
    assert "edistance" in repr_str
    assert "X_custom" in repr_str


# ============================================================================
# Correctness tests against reference implementation
# ============================================================================


def _compute_mean_euclidean_distance_cpu(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute mean Euclidean distance between all pairs in X and Y (CPU reference)."""
    if X is Y or np.array_equal(X, Y):
        # Within-group: compute upper triangle only
        dists = cdist(X, X, metric="euclidean")
        n = len(X)
        # Get upper triangle (excluding diagonal)
        triu_indices = np.triu_indices(n, k=1)
        return dists[triu_indices].mean()
    else:
        # Between-group: all pairs
        dists = cdist(X, Y, metric="euclidean")
        return dists.mean()


def _compute_energy_distance_cpu(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute energy distance between X and Y (CPU reference).

    E(X, Y) = 2 * mean_dist(X, Y) - mean_dist(X, X) - mean_dist(Y, Y)
    """
    d_xy = _compute_mean_euclidean_distance_cpu(X, Y)
    d_xx = _compute_mean_euclidean_distance_cpu(X, X)
    d_yy = _compute_mean_euclidean_distance_cpu(Y, Y)
    return 2 * d_xy - d_xx - d_yy


def test_edistance_correctness_vs_cpu(small_adata: AnnData) -> None:
    """Test that GPU energy distance matches CPU reference implementation."""
    distance = Distance(metric="edistance")
    result_df = distance.pairwise(small_adata, groupby="group")

    # Get CPU embedding
    cpu_embedding = small_adata.obsm["X_pca"].get()
    groups = small_adata.obs["group"].values

    # Compute reference energy distances for each pair
    group_names = ["g0", "g1", "g2"]
    for i, g1 in enumerate(group_names):
        for j, g2 in enumerate(group_names):
            if i <= j:
                X = cpu_embedding[groups == g1]
                Y = cpu_embedding[groups == g2]

                if g1 == g2:
                    expected = 0.0  # Self-distance is 0
                else:
                    expected = _compute_energy_distance_cpu(X, Y)

                actual = result_df.loc[g1, g2]
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"Mismatch for ({g1}, {g2}): GPU={actual}, CPU={expected}",
                )


def test_edistance_correctness_larger_dataset() -> None:
    """Test correctness with larger dataset and more groups."""
    rng = np.random.default_rng(42)
    n_groups = 5
    cells_per_group = 20
    n_features = 10
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")
    result_df = distance.pairwise(adata, groupby="group")

    # Check a few pairs against CPU reference
    for g1, g2 in [("g0", "g1"), ("g0", "g4"), ("g2", "g3")]:
        X = cpu_embedding[np.array(groups) == g1]
        Y = cpu_embedding[np.array(groups) == g2]
        expected = _compute_energy_distance_cpu(X, Y)
        actual = result_df.loc[g1, g2]
        np.testing.assert_allclose(
            actual, expected, rtol=1e-6, atol=1e-6, err_msg=f"Mismatch for ({g1}, {g2})"
        )


# ============================================================================
# Onesided distance correctness tests
# ============================================================================


def test_onesided_distances_correctness_vs_cpu(small_adata: AnnData) -> None:
    """Test that onesided_distances matches CPU reference implementation."""
    distance = Distance(metric="edistance")

    # Get CPU embedding
    cpu_embedding = small_adata.obsm["X_pca"].get()
    groups = small_adata.obs["group"].values

    # Test onesided from each group
    for selected_group in ["g0", "g1", "g2"]:
        onesided = distance.onesided_distances(
            small_adata, groupby="group", selected_group=selected_group
        )

        # Verify against CPU reference for each target group
        X = cpu_embedding[groups == selected_group]
        for target_group in ["g0", "g1", "g2"]:
            Y = cpu_embedding[groups == target_group]

            if selected_group == target_group:
                expected = 0.0
            else:
                expected = _compute_energy_distance_cpu(X, Y)

            actual = onesided.loc[target_group]
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"Onesided mismatch for ({selected_group}, {target_group})",
            )


# ============================================================================
# Contrast distance tests
# ============================================================================


@pytest.fixture
def contrast_adata() -> AnnData:
    """AnnData with treatment and celltype columns for contrast tests."""
    rng = np.random.default_rng(42)
    n = 10
    cpu_emb = rng.normal(size=(n * 4, 5)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "treatment": pd.Categorical(
                ["ctrl"] * n + ["drugA"] * n + ["ctrl"] * n + ["drugA"] * n
            ),
            "celltype": pd.Categorical(["T"] * n * 2 + ["B"] * n * 2),
        }
    )
    adata = AnnData(cpu_emb.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_emb, dtype=cp.float32)
    return adata


def test_contrast_distances_matches_compute_distance(
    contrast_adata: AnnData,
) -> None:
    """Test contrast_distances matches per-pair compute_distance reference."""
    from rapids_singlecell.pertpy_gpu._metrics._edistance import EDistanceMetric

    d = EDistanceMetric(obsm_key="X_pca")

    contrasts = Distance.create_contrasts(
        contrast_adata,
        groupby="treatment",
        selected_group="ctrl",
        split_by="celltype",
    )

    result = d.contrast_distances(contrast_adata, contrasts=contrasts)

    assert isinstance(result, pd.DataFrame)
    assert "edistance" in result.columns
    assert len(result) == len(contrasts)

    # Verify each contrast against compute_distance
    for _, row in result.iterrows():
        mask_target = (contrast_adata.obs["treatment"].values == row["treatment"]) & (
            contrast_adata.obs["celltype"].values == row["celltype"]
        )
        mask_ref = (contrast_adata.obs["treatment"].values == row["reference"]) & (
            contrast_adata.obs["celltype"].values == row["celltype"]
        )

        X = contrast_adata.obsm["X_pca"][mask_target]
        Y = contrast_adata.obsm["X_pca"][mask_ref]
        expected = d.compute_distance(X, Y)

        np.testing.assert_allclose(
            row["edistance"],
            expected,
            atol=1e-6,
            err_msg=f"Contrast {row['treatment']} vs {row['reference']} "
            f"in {row['celltype']} mismatch",
        )


def test_contrast_distances_shared_condition(contrast_adata: AnnData) -> None:
    """Test that contrasts sharing a condition (e.g. same control) work."""
    distance = Distance(metric="edistance")

    contrasts = Distance.create_contrasts(
        contrast_adata,
        groupby="treatment",
        selected_group="ctrl",
        split_by="celltype",
    )

    result = distance.contrast_distances(contrast_adata, contrasts=contrasts)

    assert isinstance(result, pd.DataFrame)
    assert "edistance" in result.columns
    # All distances should be finite
    assert np.all(np.isfinite(result["edistance"].values))


def test_contrast_distances_self_distance_zero(contrast_adata: AnnData) -> None:
    """Test that self-distance (same group vs itself) is zero."""
    distance = Distance(metric="edistance")

    # Manually create a contrast where target == reference
    contrasts = pd.DataFrame(
        {
            "treatment": ["ctrl"],
            "reference": ["ctrl"],
            "celltype": ["T"],
        }
    )

    result = distance.contrast_distances(contrast_adata, contrasts=contrasts)
    assert result["edistance"].iloc[0] == pytest.approx(0.0, abs=1e-7)


def test_contrast_distances_no_split(contrast_adata: AnnData) -> None:
    """Test contrast_distances without split_by columns."""
    distance = Distance(metric="edistance")

    contrasts = Distance.create_contrasts(
        contrast_adata,
        groupby="treatment",
        selected_group="ctrl",
    )

    result = distance.contrast_distances(contrast_adata, contrasts=contrasts)

    assert isinstance(result, pd.DataFrame)
    assert "edistance" in result.columns
    assert len(result) == 1  # only drugA vs ctrl
    assert np.all(np.isfinite(result["edistance"].values))


def test_contrast_distances_multiple_references() -> None:
    """Test create_contrasts with multiple reference groups."""
    rng = np.random.default_rng(42)
    n = 10
    cpu_emb = rng.normal(size=(n * 6, 5)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "treatment": pd.Categorical(
                ["ref1"] * n
                + ["ref2"] * n
                + ["drugA"] * n
                + ["drugB"] * n
                + ["ref1"] * n
                + ["drugA"] * n
            ),
            "celltype": pd.Categorical(["T"] * n * 4 + ["B"] * n * 2),
        }
    )
    adata = AnnData(cpu_emb.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_emb, dtype=cp.float32)

    from rapids_singlecell.pertpy_gpu._metrics._edistance import EDistanceMetric

    d = EDistanceMetric(obsm_key="X_pca")
    distance = Distance(metric="edistance")

    # Two references
    contrasts = Distance.create_contrasts(
        adata,
        groupby="treatment",
        selected_group=["ref1", "ref2"],
        split_by="celltype",
    )

    # References should not appear as targets
    assert "ref1" not in contrasts["treatment"].values
    assert "ref2" not in contrasts["treatment"].values

    # Both references should appear in the reference column
    assert "ref1" in contrasts["reference"].values
    assert "ref2" in contrasts["reference"].values

    result = distance.contrast_distances(adata, contrasts=contrasts)
    assert "edistance" in result.columns

    # Verify each row against compute_distance
    for _, row in result.iterrows():
        mask_target = (adata.obs["treatment"].values == row["treatment"]) & (
            adata.obs["celltype"].values == row["celltype"]
        )
        mask_ref = (adata.obs["treatment"].values == row["reference"]) & (
            adata.obs["celltype"].values == row["celltype"]
        )
        X = adata.obsm["X_pca"][mask_target]
        Y = adata.obsm["X_pca"][mask_ref]

        if len(X) == 0 or len(Y) == 0:
            continue
        expected = d.compute_distance(X, Y)
        np.testing.assert_allclose(row["edistance"], expected, rtol=1e-5, atol=1e-5)


def test_contrast_distances_multiple_references_no_split() -> None:
    """Test create_contrasts with multiple references and no split_by."""
    rng = np.random.default_rng(42)
    n = 15
    cpu_emb = rng.normal(size=(n * 4, 5)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "treatment": pd.Categorical(
                ["ref1"] * n + ["ref2"] * n + ["drugA"] * n + ["drugB"] * n
            ),
        }
    )
    adata = AnnData(cpu_emb.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_emb, dtype=cp.float32)

    distance = Distance(metric="edistance")

    contrasts = Distance.create_contrasts(
        adata,
        groupby="treatment",
        selected_group=["ref1", "ref2"],
    )

    # 2 targets x 2 references = 4 rows
    assert len(contrasts) == 4
    assert set(contrasts["treatment"].values) == {"drugA", "drugB"}
    assert set(contrasts["reference"].values) == {"ref1", "ref2"}

    result = distance.contrast_distances(adata, contrasts=contrasts)
    assert len(result) == 4
    assert np.all(np.isfinite(result["edistance"].values))


def test_contrast_distances_filtered(contrast_adata: AnnData) -> None:
    """Test that filtering a contrasts DataFrame before computing works."""
    from rapids_singlecell.pertpy_gpu._metrics._edistance import EDistanceMetric

    d = EDistanceMetric(obsm_key="X_pca")
    distance = Distance(metric="edistance")

    # Create full contrasts, then drop one celltype
    contrasts = Distance.create_contrasts(
        contrast_adata,
        groupby="treatment",
        selected_group="ctrl",
        split_by="celltype",
    )
    assert len(contrasts) == 2  # drugA-T, drugA-B

    # Keep only celltype == "T"
    filtered = contrasts[contrasts["celltype"] == "T"].reset_index(drop=True)
    assert len(filtered) == 1

    result = distance.contrast_distances(contrast_adata, contrasts=filtered)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result["celltype"].iloc[0] == "T"

    # Verify the distance matches compute_distance
    mask_target = (contrast_adata.obs["treatment"].values == "drugA") & (
        contrast_adata.obs["celltype"].values == "T"
    )
    mask_ref = (contrast_adata.obs["treatment"].values == "ctrl") & (
        contrast_adata.obs["celltype"].values == "T"
    )
    expected = d.compute_distance(
        contrast_adata.obsm["X_pca"][mask_target],
        contrast_adata.obsm["X_pca"][mask_ref],
    )
    np.testing.assert_allclose(result["edistance"].iloc[0], expected, atol=1e-6)

    # Also verify it differs from the full (unfiltered) result
    full_result = distance.contrast_distances(contrast_adata, contrasts=contrasts)
    assert len(full_result) == 2

    # The T-cell row should match between filtered and full
    full_t = full_result[full_result["celltype"] == "T"]["edistance"].iloc[0]
    np.testing.assert_allclose(result["edistance"].iloc[0], full_t, atol=1e-10)


def test_contrast_distances_two_split_by() -> None:
    """Test contrast_distances with two split_by columns."""
    rng = np.random.default_rng(42)
    n = 10
    cpu_emb = rng.normal(size=(n * 6, 5)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "treatment": pd.Categorical(
                ["ctrl"] * n
                + ["drugA"] * n
                + ["ctrl"] * n
                + ["drugA"] * n
                + ["ctrl"] * n
                + ["drugA"] * n
            ),
            "celltype": pd.Categorical(["T"] * n * 2 + ["B"] * n * 2 + ["T"] * n * 2),
            "batch": pd.Categorical(["b1"] * n * 4 + ["b2"] * n * 2),
        }
    )
    adata = AnnData(cpu_emb.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_emb, dtype=cp.float32)

    from rapids_singlecell.pertpy_gpu._metrics._edistance import EDistanceMetric

    d = EDistanceMetric(obsm_key="X_pca")
    distance = Distance(metric="edistance")

    contrasts = Distance.create_contrasts(
        adata,
        groupby="treatment",
        selected_group="ctrl",
        split_by=["celltype", "batch"],
    )

    assert "celltype" in contrasts.columns
    assert "batch" in contrasts.columns

    result = distance.contrast_distances(adata, contrasts=contrasts)

    assert isinstance(result, pd.DataFrame)
    assert "edistance" in result.columns

    # Verify each contrast against compute_distance
    for _, row in result.iterrows():
        mask_target = (
            (adata.obs["treatment"].values == row["treatment"])
            & (adata.obs["celltype"].values == row["celltype"])
            & (adata.obs["batch"].values == row["batch"])
        )
        mask_ref = (
            (adata.obs["treatment"].values == row["reference"])
            & (adata.obs["celltype"].values == row["celltype"])
            & (adata.obs["batch"].values == row["batch"])
        )

        X = adata.obsm["X_pca"][mask_target]
        Y = adata.obsm["X_pca"][mask_ref]
        expected = d.compute_distance(X, Y)

        np.testing.assert_allclose(
            row["edistance"],
            expected,
            rtol=1e-6,
            atol=1e-6,
        )


# ============================================================================
# Bootstrap correctness tests
# ============================================================================


def test_bootstrap_variance_is_positive(small_adata: AnnData) -> None:
    """Test that bootstrap variance is always non-negative."""
    distance = Distance(metric="edistance")
    _, distances_var_df = distance.pairwise(
        small_adata,
        groupby="group",
        bootstrap=True,
        n_bootstrap=20,
        random_state=42,
    )

    # All variances should be >= 0
    assert np.all(distances_var_df.values >= 0), (
        "Bootstrap variance should be non-negative"
    )

    # Off-diagonal variances should be > 0 (with bootstrap resampling)
    for i, g1 in enumerate(["g0", "g1", "g2"]):
        for j, g2 in enumerate(["g0", "g1", "g2"]):
            if i != j:
                assert distances_var_df.loc[g1, g2] > 0, (
                    f"Off-diagonal variance should be positive for ({g1}, {g2})"
                )


def test_bootstrap_mean_close_to_point_estimate() -> None:
    """Test that bootstrap mean is close to point estimate (no bootstrap)."""
    # Need larger dataset for meaningful bootstrap comparison
    rng = np.random.default_rng(42)
    n_groups = 3
    cells_per_group = 50  # Larger for stable bootstrap
    n_features = 10
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")

    # Point estimate
    point_df = distance.pairwise(adata, groupby="group")

    # Bootstrap estimate
    boot_df, _ = distance.pairwise(
        adata,
        groupby="group",
        bootstrap=True,
        n_bootstrap=100,
        random_state=42,
    )

    # Bootstrap mean should be reasonably close to point estimate
    np.testing.assert_allclose(
        boot_df.values,
        point_df.values,
        rtol=0.3,  # 30% tolerance for bootstrap variation
        atol=0.5,  # Absolute tolerance for small distances
        err_msg="Bootstrap mean should be close to point estimate",
    )


def test_bootstrap_reproducibility(small_adata: AnnData) -> None:
    """Test that bootstrap with same seed produces same results."""
    distance = Distance(metric="edistance")

    df1, var1 = distance.pairwise(
        small_adata,
        groupby="group",
        bootstrap=True,
        n_bootstrap=20,
        random_state=42,
    )

    df2, var2 = distance.pairwise(
        small_adata,
        groupby="group",
        bootstrap=True,
        n_bootstrap=20,
        random_state=42,
    )

    np.testing.assert_allclose(
        df1.values,
        df2.values,
        atol=1e-10,
        err_msg="Same seed should produce same bootstrap results",
    )

    np.testing.assert_allclose(
        var1.values,
        var2.values,
        atol=1e-10,
        err_msg="Same seed should produce same bootstrap variance",
    )


# ============================================================================
# Tests for Distance.__call__ API (pertpy-compatible)
# ============================================================================


def test_distance_call_api_basic(small_adata: AnnData) -> None:
    """Test Distance.__call__ computes distance directly from arrays."""
    distance = Distance(metric="edistance")

    # Extract arrays for two groups
    cpu_embedding = small_adata.obsm["X_pca"].get()
    groups = small_adata.obs["group"].values

    X = cpu_embedding[groups == "g0"]
    Y = cpu_embedding[groups == "g1"]

    # Compute via __call__
    d = distance(X, Y)

    assert isinstance(d, float)
    assert np.isfinite(d), "Distance should be finite"


def test_distance_call_api_vs_cpu_reference(small_adata: AnnData) -> None:
    """Test Distance.__call__ matches CPU reference implementation."""
    distance = Distance(metric="edistance")

    cpu_embedding = small_adata.obsm["X_pca"].get()
    groups = small_adata.obs["group"].values

    # Test for each pair
    for g1, g2 in [("g0", "g1"), ("g0", "g2"), ("g1", "g2")]:
        X = cpu_embedding[groups == g1]
        Y = cpu_embedding[groups == g2]

        # GPU result via __call__
        actual = distance(X, Y)

        # CPU reference
        expected = _compute_energy_distance_cpu(X, Y)

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"__call__ mismatch for ({g1}, {g2})",
        )


def test_distance_call_api_vs_pairwise(small_adata: AnnData) -> None:
    """Test Distance.__call__ matches pairwise results."""
    distance = Distance(metric="edistance")

    # Get pairwise results
    result_df = distance.pairwise(small_adata, groupby="group")

    cpu_embedding = small_adata.obsm["X_pca"].get()
    groups = small_adata.obs["group"].values

    # Test each pair
    for g1 in ["g0", "g1", "g2"]:
        for g2 in ["g0", "g1", "g2"]:
            if g1 == g2:
                continue

            X = cpu_embedding[groups == g1]
            Y = cpu_embedding[groups == g2]

            # Via __call__
            call_result = distance(X, Y)

            # Via pairwise
            pairwise_result = result_df.loc[g1, g2]

            np.testing.assert_allclose(
                call_result,
                pairwise_result,
                atol=1e-6,
                err_msg=f"__call__ vs pairwise mismatch for ({g1}, {g2})",
            )


def test_distance_call_api_with_cupy_arrays(small_adata: AnnData) -> None:
    """Test Distance.__call__ works with CuPy arrays."""
    distance = Distance(metric="edistance")

    # Use CuPy arrays directly
    embedding = small_adata.obsm["X_pca"]  # Already CuPy
    groups = small_adata.obs["group"].values

    X = embedding[cp.array(groups == "g0")]
    Y = embedding[cp.array(groups == "g1")]

    # Should work with CuPy arrays
    d = distance(X, Y)
    assert isinstance(d, float)


# ============================================================================
# Tests for Distance.bootstrap with arrays (pertpy-compatible)
# ============================================================================


def test_distance_bootstrap_arrays_basic(small_adata: AnnData) -> None:
    """Test Distance.bootstrap() with arrays (pertpy-compatible API)."""
    distance = Distance(metric="edistance")

    cpu_embedding = small_adata.obsm["X_pca"].get()
    groups = small_adata.obs["group"].values

    X = cpu_embedding[groups == "g0"]
    Y = cpu_embedding[groups == "g1"]

    result = distance.bootstrap(X, Y, n_bootstrap=20, random_state=42)

    assert isinstance(result, MeanVar)
    assert isinstance(result.mean, float)
    assert isinstance(result.variance, float)
    assert result.variance >= 0


def test_distance_bootstrap_arrays_reproducibility() -> None:
    """Test bootstrap with arrays is reproducible with same seed."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 10)).astype(np.float32)
    Y = rng.normal(size=(30, 10)).astype(np.float32)

    distance = Distance(metric="edistance")

    result1 = distance.bootstrap(X, Y, n_bootstrap=50, random_state=123)
    result2 = distance.bootstrap(X, Y, n_bootstrap=50, random_state=123)

    assert result1.mean == result2.mean
    assert result1.variance == result2.variance


def test_distance_bootstrap_arrays_vs_call() -> None:
    """Test bootstrap mean is close to direct __call__ result."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 10)).astype(np.float32)
    Y = rng.normal(loc=1.0, size=(50, 10)).astype(np.float32)  # Different distribution

    distance = Distance(metric="edistance")

    # Point estimate
    point_estimate = distance(X, Y)

    # Bootstrap estimate
    result = distance.bootstrap(X, Y, n_bootstrap=100, random_state=42)

    # Bootstrap mean should be reasonably close to point estimate
    np.testing.assert_allclose(
        result.mean,
        point_estimate,
        rtol=0.3,
        atol=0.5,
        err_msg="Bootstrap mean should be close to point estimate",
    )


# ============================================================================
# Tests for layer_key support
# ============================================================================


def test_distance_layer_key_basic() -> None:
    """Test Distance with layer_key (uses data with same shape as adata.X)."""
    rng = np.random.default_rng(42)
    n_groups = 3
    cells_per_group = 20
    n_features = 10  # This must match n_vars for layers
    total_cells = n_groups * cells_per_group

    # Create expression data with proper shape for both X and layer
    cpu_data = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    # Create AnnData with n_features as n_vars (required for layer to match)
    adata = AnnData(np.zeros((total_cells, n_features), dtype=np.float32), obs=obs)
    # Store in layer (must match shape of X: n_obs x n_vars)
    adata.layers["counts"] = cpu_data

    distance = Distance(metric="edistance", layer_key="counts")
    assert distance.layer_key == "counts"
    assert distance.obsm_key is None

    result_df = distance.pairwise(adata, groupby="group")

    # Verify against CPU reference
    for g1, g2 in [("g0", "g1"), ("g0", "g2")]:
        X = cpu_data[np.array(groups) == g1]
        Y = cpu_data[np.array(groups) == g2]
        expected = _compute_energy_distance_cpu(X, Y)
        actual = result_df.loc[g1, g2]
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"layer_key mismatch for ({g1}, {g2})",
        )


def test_distance_layer_key_mutual_exclusion() -> None:
    """Test that layer_key and obsm_key are mutually exclusive."""
    with pytest.raises(ValueError, match="Cannot use 'layer_key' and 'obsm_key'"):
        Distance(metric="edistance", layer_key="counts", obsm_key="X_pca")


def test_distance_layer_key_repr() -> None:
    """Test __repr__ includes layer_key."""
    distance = Distance(metric="edistance", layer_key="counts")
    repr_str = repr(distance)
    assert "layer_key" in repr_str
    assert "counts" in repr_str


def test_distance_default_obsm_key() -> None:
    """Test default obsm_key is X_pca when neither layer_key nor obsm_key is specified."""
    distance = Distance(metric="edistance")
    assert distance.obsm_key == "X_pca"
    assert distance.layer_key is None


# ============================================================================
# Dtype and kernel coverage tests (parametrized)
# ============================================================================


def test_float64_matches_float32_results() -> None:
    """Test that float64 and float32 produce similar results (within float32 precision)."""
    # Use small dataset to avoid GPU resource exhaustion with float64
    rng = np.random.default_rng(0)
    n_groups = 3
    cells_per_group = 4
    n_features = 5
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata_f32 = AnnData(cpu_embedding, obs=obs)
    adata_f32.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    adata_f64 = AnnData(cpu_embedding, obs=obs.copy())
    adata_f64.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float64)

    distance = Distance(metric="edistance")

    result_f32 = distance.pairwise(adata_f32, groupby="group")
    result_f64 = distance.pairwise(adata_f64, groupby="group")

    np.testing.assert_allclose(
        result_f32.values,
        result_f64.values,
        rtol=1e-6,
        atol=1e-6,
        err_msg="Float64 and float32 results should be similar",
    )


@pytest.mark.parametrize("n_features", [50, 400])
def test_bootstrap_different_feature_counts(n_features: int) -> None:
    """Test bootstrap works with different feature counts (50 and 400)."""
    rng = np.random.default_rng(42)
    n_groups = 3
    cells_per_group = 10
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")
    distances, variances = distance.pairwise(
        adata, groupby="group", bootstrap=True, n_bootstrap=10, random_state=42
    )

    assert isinstance(distances, pd.DataFrame)
    assert isinstance(variances, pd.DataFrame)
    assert np.all(variances.values >= 0)

    # Non-self distances should have positive variance
    for g1 in ["g0", "g1", "g2"]:
        for g2 in ["g0", "g1", "g2"]:
            if g1 != g2:
                assert variances.loc[g1, g2] > 0


# ============================================================================
# Combined dtype and kernel tests
# ============================================================================


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pairwise_correctness_parametrized(dtype) -> None:
    """Parametrized test for pairwise correctness across dtypes."""
    rng = np.random.default_rng(42)
    # Use smaller sizes for float64 to avoid GPU resource exhaustion
    n_groups = 3
    cells_per_group = 4 if dtype == np.float64 else 15
    n_features = 5 if dtype == np.float64 else 20
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(dtype)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy().astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=dtype)

    distance = Distance(metric="edistance")
    result_df = distance.pairwise(adata, groupby="group")

    # Check a few pairs
    rtol = 1e-10 if dtype == np.float64 else 1e-5
    atol = 1e-12 if dtype == np.float64 else 1e-6

    for g1, g2 in [("g0", "g1"), ("g0", "g2"), ("g1", "g2")]:
        X = cpu_embedding[np.array(groups) == g1]
        Y = cpu_embedding[np.array(groups) == g2]
        expected = _compute_energy_distance_cpu(X, Y)
        actual = result_df.loc[g1, g2]
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            err_msg=f"dtype={dtype.__name__} mismatch for ({g1}, {g2})",
        )


@pytest.mark.parametrize("n_features", [50, 400])
def test_correctness_float32_features(n_features) -> None:
    """Test correctness across feature counts with float32."""
    rng = np.random.default_rng(42)
    n_groups = 3
    cells_per_group = 10
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=np.float32)

    distance = Distance(metric="edistance")
    result_df = distance.pairwise(adata, groupby="group")

    X = cpu_embedding[np.array(groups) == "g0"]
    Y = cpu_embedding[np.array(groups) == "g1"]
    expected = _compute_energy_distance_cpu(X, Y)
    actual = result_df.loc["g0", "g1"]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-5,
        err_msg=f"n_features={n_features} mismatch",
    )


def test_correctness_float64_small() -> None:
    """Test correctness with float64 using small data to avoid GPU resource exhaustion."""
    rng = np.random.default_rng(42)
    n_groups = 3
    cells_per_group = 4
    n_features = 5
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float64)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy().astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=np.float64)

    distance = Distance(metric="edistance")
    result_df = distance.pairwise(adata, groupby="group")

    X = cpu_embedding[np.array(groups) == "g0"]
    Y = cpu_embedding[np.array(groups) == "g1"]
    expected = _compute_energy_distance_cpu(X, Y)
    actual = result_df.loc["g0", "g1"]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-10,
        atol=1e-12,
        err_msg="float64 mismatch",
    )


# ============================================================================
# Block size consistency test
# ============================================================================


def test_block_size_consistency() -> None:
    """Test that both block sizes (1024 and 256) produce identical results.

    This test runs on GPUs that support both block sizes (Ampere+, CC >= 8.0)
    and verifies both code paths produce the same results.
    """
    from rapids_singlecell._utils import _multi_gpu

    # Check if GPU supports both block sizes (Ampere+)
    device_attrs = _multi_gpu._get_device_attrs()
    if device_attrs["cc_major"] < 8:
        pytest.skip("GPU does not support both block sizes (requires CC >= 8.0)")

    rng = np.random.default_rng(42)
    n_groups = 3
    cells_per_group = 10
    n_features = 20
    total_cells = n_groups * cells_per_group

    # Use float64 since block size difference only matters for float64 on pre-Ampere
    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float64)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy().astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float64)

    # Get result with default block size (1024 for Ampere+)
    distance = Distance(metric="edistance")
    result = distance.pairwise(adata, groupby="group")

    # Temporarily override cc_major to force 256 block size
    device_id = cp.cuda.Device().id
    original_cc = device_attrs["cc_major"]
    try:
        _multi_gpu._DEVICE_ATTRS_CACHE[device_id]["cc_major"] = (
            7  # Force pre-Ampere path
        )
        distance_256 = Distance(metric="edistance")
        result_256 = distance_256.pairwise(adata, groupby="group")
    finally:
        _multi_gpu._DEVICE_ATTRS_CACHE[device_id]["cc_major"] = original_cc  # Restore

    # Results should be identical between 256 and 1024 block paths
    np.testing.assert_allclose(
        result.values,
        result_256.values,
        rtol=1e-7,
        atol=1e-7,
        err_msg="256-block and 1024-block paths should produce identical results",
    )
    # Diagonal should be zero (self-distance)
    np.testing.assert_allclose(
        np.diag(result.values),
        0,
        atol=1e-7,
        err_msg="Diagonal (self-distance) should be zero",
    )


# ============================================================================
# Distance axioms tests (semimetric properties)
# ============================================================================


@pytest.mark.parametrize(
    "n_groups,cells_per_group,n_features,use_distinct_distributions",
    [
        (3, 4, 5, False),  # Small dataset (uses small_adata-like setup)
        (5, 50, 20, True),  # Larger dataset with distinct distributions
    ],
)
def test_distance_axioms(
    n_groups: int,
    cells_per_group: int,
    n_features: int,
    use_distinct_distributions: bool,
) -> None:
    """Test distance axioms: definiteness, symmetry, and positivity.

    Note: Energy distance can be negative with very small samples from same
    distribution. With distinct distributions or larger samples, positivity holds.
    """
    rng = np.random.default_rng(42)
    total_cells = n_groups * cells_per_group

    if use_distinct_distributions:
        # Create groups with distinctly different distributions
        cpu_embeddings = []
        for i in range(n_groups):
            group_data = rng.normal(loc=i * 2.0, size=(cells_per_group, n_features))
            cpu_embeddings.append(group_data)
        cpu_embedding = np.vstack(cpu_embeddings).astype(np.float32)
    else:
        cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)

    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")
    result_df = distance.pairwise(adata, groupby="group")

    # Definiteness: d(x, x) = 0
    for group in result_df.index:
        assert result_df.loc[group, group] == pytest.approx(0.0, abs=1e-10), (
            f"Self-distance for {group} should be 0"
        )

    # Symmetry: d(x, y) = d(y, x)
    np.testing.assert_allclose(
        result_df.values,
        result_df.values.T,
        atol=1e-7,
        err_msg="Matrix should be symmetric",
    )

    # Positivity: d(x, y) >= 0 for distinct distributions
    if use_distinct_distributions:
        for g1 in result_df.index:
            for g2 in result_df.columns:
                if g1 != g2:
                    assert result_df.loc[g1, g2] >= 0, (
                        f"Distance ({g1}, {g2}) should be non-negative"
                    )


# ============================================================================
# Triangle inequality test
# ============================================================================


@pytest.fixture
def pertpy_adata() -> AnnData:
    """Load pertpy's example dataset for testing.

    This uses the same data pertpy uses in their tests.
    """
    pt = pytest.importorskip("pertpy")
    import scanpy as sc

    adata = pt.dt.distance_example()

    # Subsample like pertpy tests do (0.1% for most distances)
    sc.pp.subsample(adata, fraction=0.01, random_state=42)

    # Compute PCA if not present
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=5)

    # Convert to GPU
    adata.obsm["X_pca"] = cp.asarray(adata.obsm["X_pca"], dtype=cp.float32)

    return adata


def test_triangle_inequality_pertpy_data(pertpy_adata: AnnData) -> None:
    """Test triangle inequality using pertpy's example dataset.

    NOTE: The raw energy statistic does not mathematically guarantee the
    triangle inequality (only sqrt(energy_statistic) is a true metric).
    However, with real biological data, violations are rare. This test
    follows pertpy's approach of testing random triplets.

    We test 10 random triplets and allow up to 2 violations (20%), which
    accounts for the statistical nature of the raw energy statistic.
    """
    rng = np.random.default_rng(42)

    distance = Distance(metric="edistance")
    pairwise_df = distance.pairwise(pertpy_adata, groupby="perturbation")

    groups = list(pairwise_df.index)
    n_tests = 10
    max_violations = 2  # Allow some violations since it's not guaranteed

    violations = []
    for _ in range(n_tests):
        triplet = rng.choice(groups, size=3, replace=False)
        a, b, c = triplet

        d_ab = pairwise_df.loc[a, b]
        d_bc = pairwise_df.loc[b, c]
        d_ac = pairwise_df.loc[a, c]

        # Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        if d_ac > d_ab + d_bc + 1e-6:
            violations.append(
                f"d({a},{c})={d_ac:.2f} > d({a},{b})={d_ab:.2f} + d({b},{c})={d_bc:.2f}"
            )

    assert len(violations) <= max_violations, (
        f"Too many triangle inequality violations ({len(violations)}/{n_tests}): "
        f"{violations[:3]}..."  # Show first 3
    )


# ============================================================================
# Output format validation tests
# ============================================================================


def test_pairwise_output_format(small_adata: AnnData) -> None:
    """Test pairwise output format: DataFrame, symmetric, proper names."""
    distance = Distance(metric="edistance")
    result = distance.pairwise(small_adata, groupby="group")

    # Is DataFrame
    assert isinstance(result, pd.DataFrame), "pairwise should return DataFrame"

    # Index and columns match
    assert list(result.index) == list(result.columns), "Index and columns should match"

    # Symmetric
    np.testing.assert_allclose(
        result.values, result.values.T, atol=1e-10, err_msg="Matrix should be symmetric"
    )

    # Proper names
    assert result.index.name == "group", "Index name should match groupby key"
    assert result.columns.name == "group", "Columns name should match groupby key"

    # Contains all groups
    expected_groups = list(small_adata.obs["group"].cat.categories)
    assert list(result.index) == expected_groups
    assert list(result.columns) == expected_groups


def test_onesided_output_format(small_adata: AnnData) -> None:
    """Test onesided_distances output format: DataFrame with proper structure."""
    distance = Distance(metric="edistance")
    result = distance.onesided_distances(
        small_adata, groupby="group", selected_group="g0"
    )

    assert isinstance(result, pd.Series), (
        "onesided_distances with single control should return Series"
    )
    assert result.index.name == "group"


# ============================================================================
# Groups parameter filtering tests
# ============================================================================


def test_pairwise_groups_parameter_filters_output(small_adata: AnnData) -> None:
    """Test that groups parameter filters the pairwise output correctly."""
    distance = Distance(metric="edistance")

    # Request only subset of groups
    result = distance.pairwise(small_adata, groupby="group", groups=["g0", "g1"])

    assert list(result.index) == ["g0", "g1"]
    assert list(result.columns) == ["g0", "g1"]
    assert "g2" not in result.index
    assert "g2" not in result.columns


def test_pairwise_groups_parameter_values_correct(small_adata: AnnData) -> None:
    """Test that filtered pairwise values match full pairwise computation."""
    distance = Distance(metric="edistance")

    # Full pairwise
    full_result = distance.pairwise(small_adata, groupby="group")

    # Filtered pairwise
    filtered_result = distance.pairwise(
        small_adata, groupby="group", groups=["g0", "g2"]
    )

    # Values should match
    np.testing.assert_allclose(
        filtered_result.loc["g0", "g2"], full_result.loc["g0", "g2"], atol=1e-10
    )
    np.testing.assert_allclose(
        filtered_result.loc["g0", "g0"], full_result.loc["g0", "g0"], atol=1e-10
    )


def test_onesided_groups_parameter_filters_output(small_adata: AnnData) -> None:
    """Test that groups parameter filters onesided_distances output."""
    distance = Distance(metric="edistance")

    result = distance.onesided_distances(
        small_adata, groupby="group", selected_group="g0", groups=["g0", "g1"]
    )

    assert len(result) == 2
    assert "g0" in result.index
    assert "g1" in result.index
    assert "g2" not in result.index


def test_pairwise_groups_parameter_with_bootstrap(small_adata: AnnData) -> None:
    """Test groups parameter works with bootstrap."""
    distance = Distance(metric="edistance")

    distances, variances = distance.pairwise(
        small_adata,
        groupby="group",
        groups=["g0", "g1"],
        bootstrap=True,
        n_bootstrap=10,
        random_state=42,
    )

    assert list(distances.index) == ["g0", "g1"]
    assert list(variances.index) == ["g0", "g1"]


# ============================================================================
# Edge case tests
# ============================================================================


def test_two_groups_only() -> None:
    """Test with exactly two groups (minimum for pairwise comparison)."""
    rng = np.random.default_rng(42)
    n_groups = 2
    cells_per_group = 10
    n_features = 5
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")
    result_df = distance.pairwise(adata, groupby="group")

    assert result_df.shape == (2, 2)
    assert result_df.loc["g0", "g0"] == pytest.approx(0.0, abs=1e-10)
    assert result_df.loc["g1", "g1"] == pytest.approx(0.0, abs=1e-10)
    assert result_df.loc["g0", "g1"] >= 0


def test_many_groups() -> None:
    """Test with many groups (stress test)."""
    rng = np.random.default_rng(42)
    n_groups = 20
    cells_per_group = 5
    n_features = 10
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")
    result_df = distance.pairwise(adata, groupby="group")

    assert result_df.shape == (20, 20)
    # Check diagonal is all zeros
    for i in range(n_groups):
        assert result_df.iloc[i, i] == pytest.approx(0.0, abs=1e-10)


def test_unequal_group_sizes() -> None:
    """Test with unequal group sizes."""
    rng = np.random.default_rng(42)
    n_features = 5

    # Groups with sizes 5, 20, 50
    group_sizes = [5, 20, 50]
    total_cells = sum(group_sizes)

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = []
    for i, size in enumerate(group_sizes):
        groups.extend([f"g{i}"] * size)

    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(3)])}
    )

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")
    result_df = distance.pairwise(adata, groupby="group")

    # Should still produce valid symmetric matrix
    assert result_df.shape == (3, 3)
    np.testing.assert_allclose(result_df.values, result_df.values.T, atol=1e-10)

    # Verify against CPU reference
    for g1, g2 in [("g0", "g1"), ("g0", "g2"), ("g1", "g2")]:
        X = cpu_embedding[np.array(groups) == g1]
        Y = cpu_embedding[np.array(groups) == g2]
        expected = _compute_energy_distance_cpu(X, Y)
        actual = result_df.loc[g1, g2]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_distance_call_empty_array_error() -> None:
    """Test that __call__ raises error for empty arrays."""
    distance = Distance(metric="edistance")
    X = np.array([], dtype=np.float32).reshape(0, 5)
    Y = np.random.default_rng(42).normal(size=(10, 5)).astype(np.float32)

    with pytest.raises(ValueError, match="empty"):
        distance(X, Y)


def test_distance_call_empty_second_array_error() -> None:
    """Test that __call__ raises error when second array is empty."""
    distance = Distance(metric="edistance")
    X = np.random.default_rng(42).normal(size=(10, 5)).astype(np.float32)
    Y = np.array([], dtype=np.float32).reshape(0, 5)

    with pytest.raises(ValueError, match="empty"):
        distance(X, Y)


def test_bootstrap_empty_array_error() -> None:
    """Test that bootstrap raises error for empty arrays."""
    distance = Distance(metric="edistance")
    X = np.array([], dtype=np.float32).reshape(0, 5)
    Y = np.random.default_rng(42).normal(size=(10, 5)).astype(np.float32)

    with pytest.raises(ValueError, match="empty"):
        distance.bootstrap(X, Y, n_bootstrap=10)


def test_missing_obsm_key_error(small_adata: AnnData) -> None:
    """Test error when obsm_key doesn't exist in adata."""
    distance = Distance(metric="edistance", obsm_key="X_nonexistent")

    with pytest.raises(KeyError):
        distance.pairwise(small_adata, groupby="group")


def test_missing_layer_key_error(small_adata: AnnData) -> None:
    """Test error when layer_key doesn't exist in adata."""
    distance = Distance(metric="edistance", layer_key="nonexistent_layer")

    with pytest.raises(KeyError):
        distance.pairwise(small_adata, groupby="group")


def test_non_categorical_groupby_error(small_adata: AnnData) -> None:
    """Test error when groupby column is not categorical."""
    # Convert to non-categorical
    small_adata.obs["group_str"] = small_adata.obs["group"].astype(str)

    distance = Distance(metric="edistance")

    with pytest.raises((ValueError, TypeError)):
        distance.pairwise(small_adata, groupby="group_str")


def test_missing_groupby_column_error(small_adata: AnnData) -> None:
    """Test error when groupby column doesn't exist."""
    distance = Distance(metric="edistance")

    with pytest.raises(KeyError):
        distance.pairwise(small_adata, groupby="nonexistent_column")


def test_single_cell_per_group() -> None:
    """Test with single cell per group (edge case for within-group distance)."""
    rng = np.random.default_rng(42)
    n_groups = 3
    cells_per_group = 1  # Single cell per group
    n_features = 5
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")
    result_df = distance.pairwise(adata, groupby="group")

    # All results should be finite (no NaN or inf)
    assert np.all(np.isfinite(result_df.values)), "All distances should be finite"

    # Diagonal (within-group) distances should be 0 for single-cell groups
    for g in result_df.index:
        assert result_df.loc[g, g] == 0.0, f"Within-group distance for {g} should be 0"

    # Between-group distances should be positive (different cells)
    for g1 in result_df.index:
        for g2 in result_df.columns:
            if g1 != g2:
                assert result_df.loc[g1, g2] > 0, f"Distance ({g1}, {g2}) should be > 0"


def test_single_cell_mixed_groups() -> None:
    """Test with mix of single-cell and multi-cell groups."""
    rng = np.random.default_rng(42)
    n_features = 5

    # g0: 1 cell, g1: 1 cell, g2: 5 cells
    group_sizes = [1, 1, 5]
    total_cells = sum(group_sizes)

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    group_labels = []
    for i, size in enumerate(group_sizes):
        group_labels.extend([f"g{i}"] * size)

    obs = pd.DataFrame(
        {"group": pd.Categorical(group_labels, categories=["g0", "g1", "g2"])}
    )
    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")
    result_df = distance.pairwise(adata, groupby="group")

    # All results should be finite
    assert np.all(np.isfinite(result_df.values)), "All distances should be finite"

    # Single-cell groups (g0, g1) should have 0 diagonal
    assert result_df.loc["g0", "g0"] == 0.0, "g0 within-group should be 0"
    assert result_df.loc["g1", "g1"] == 0.0, "g1 within-group should be 0"

    # Multi-cell group (g2) can have non-zero diagonal (within-group variance)
    # Just check it's finite
    assert np.isfinite(result_df.loc["g2", "g2"]), "g2 within-group should be finite"

    # Test onesided_distances with single-cell selected group
    onesided = distance.onesided_distances(adata, groupby="group", selected_group="g0")
    assert np.all(np.isfinite(onesided.values)), "Onesided distances should be finite"
    assert onesided.loc["g0"] == 0.0, "Self-distance should be 0"


def test_high_dimensional_features() -> None:
    """Test with very high dimensional features (beyond typical PCA)."""
    rng = np.random.default_rng(42)
    n_groups = 3
    cells_per_group = 10
    n_features = 1000  # Very high dimensional
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")
    result_df = distance.pairwise(adata, groupby="group")

    # Verify correctness against CPU reference
    X = cpu_embedding[np.array(groups) == "g0"]
    Y = cpu_embedding[np.array(groups) == "g1"]
    expected = _compute_energy_distance_cpu(X, Y)
    actual = result_df.loc["g0", "g1"]

    np.testing.assert_allclose(
        actual, expected, rtol=1e-4, atol=1e-5, err_msg="High-dimensional mismatch"
    )


def test_similar_distributions_small_distance() -> None:
    """Test that similar distributions have small distance.

    Note: Energy distance measures distributional difference. Two groups
    sampled from the same underlying distribution will have distance
    close to 0 (but not exactly 0 due to sampling variation).

    Two groups with identical data points do NOT have 0 distance because
    the formula uses different pair counts for between-group vs within-group.
    """
    rng = np.random.default_rng(42)
    n_features = 10

    # Create two groups sampled from the same distribution
    # With enough samples, their distance should be small
    cells_per_group = 100
    g0_data = rng.normal(size=(cells_per_group, n_features)).astype(np.float32)
    g1_data = rng.normal(size=(cells_per_group, n_features)).astype(np.float32)
    cpu_embedding = np.vstack([g0_data, g1_data])

    groups = ["g0"] * cells_per_group + ["g1"] * cells_per_group
    obs = pd.DataFrame({"group": pd.Categorical(groups, categories=["g0", "g1"])})

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")
    result_df = distance.pairwise(adata, groupby="group")

    # Distance between samples from same distribution should be small
    # (close to 0, but not exactly due to sampling)
    assert abs(result_df.loc["g0", "g1"]) < 0.5, (
        f"Distance between similar distributions should be small, "
        f"got {result_df.loc['g0', 'g1']}"
    )


def test_different_distributions_larger_distance() -> None:
    """Test that different distributions have larger distance than similar ones."""
    rng = np.random.default_rng(42)
    n_features = 10
    cells_per_group = 50

    # g0 and g1: same distribution (N(0, 1))
    # g2: different distribution (N(5, 1))
    g0_data = rng.normal(loc=0, size=(cells_per_group, n_features))
    g1_data = rng.normal(loc=0, size=(cells_per_group, n_features))
    g2_data = rng.normal(loc=5, size=(cells_per_group, n_features))
    cpu_embedding = np.vstack([g0_data, g1_data, g2_data]).astype(np.float32)

    groups = (
        ["g0"] * cells_per_group + ["g1"] * cells_per_group + ["g2"] * cells_per_group
    )
    obs = pd.DataFrame({"group": pd.Categorical(groups, categories=["g0", "g1", "g2"])})

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")
    result_df = distance.pairwise(adata, groupby="group")

    # Distance between same distributions should be smaller than different
    d_same = abs(result_df.loc["g0", "g1"])
    d_diff = result_df.loc["g0", "g2"]

    assert d_diff > d_same, (
        f"Distance to different distribution ({d_diff:.4f}) should be larger "
        f"than distance between same distributions ({d_same:.4f})"
    )


# ============================================================================
# Multi-GPU tests
# ============================================================================


def _has_multiple_gpus() -> bool:
    """Check if system has multiple GPUs available."""
    return cp.cuda.runtime.getDeviceCount() >= 2


@pytest.mark.skipif(not _has_multiple_gpus(), reason="Requires 2+ GPUs")
def test_multi_gpu_pairwise_matches_single_gpu() -> None:
    """Test that multi-GPU pairwise produces same results as single-GPU."""
    rng = np.random.default_rng(42)
    n_groups = 5
    cells_per_group = 20
    n_features = 10
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")

    # Compute with forced single GPU
    single_result = distance.pairwise(adata, groupby="group", multi_gpu=False)

    # Compute with multi GPU (uses all available GPUs)
    multi_result = distance.pairwise(adata, groupby="group", multi_gpu=True)

    np.testing.assert_allclose(
        single_result.values,
        multi_result.values,
        rtol=1e-7,
        atol=1e-7,
        err_msg="Multi-GPU pairwise should match single-GPU",
    )


@pytest.mark.skipif(not _has_multiple_gpus(), reason="Requires 2+ GPUs")
def test_multi_gpu_onesided_matches_single_gpu() -> None:
    """Test that multi-GPU onesided_distances produces same results as single-GPU."""
    rng = np.random.default_rng(42)
    n_groups = 5
    cells_per_group = 20
    n_features = 10
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")

    # Compute with forced single GPU
    single_result = distance.onesided_distances(
        adata, groupby="group", selected_group="g0", multi_gpu=False
    )

    # Compute with multi GPU
    multi_result = distance.onesided_distances(
        adata, groupby="group", selected_group="g0", multi_gpu=True
    )

    np.testing.assert_allclose(
        single_result.values,
        multi_result.values,
        rtol=1e-7,
        atol=1e-7,
        err_msg="Multi-GPU onesided should match single-GPU",
    )


@pytest.mark.skipif(not _has_multiple_gpus(), reason="Requires 2+ GPUs")
def test_multi_gpu_bootstrap_matches_single_gpu() -> None:
    """Test that multi-GPU bootstrap produces same results as single-GPU.

    Since bootstrap indices are generated on CPU, results should be identical
    regardless of GPU count (same seed gives same bootstrap samples).
    """
    rng = np.random.default_rng(42)
    n_groups = 4
    cells_per_group = 15
    n_features = 10
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")

    # Compute with forced single GPU
    single_df, single_var = distance.pairwise(
        adata,
        groupby="group",
        bootstrap=True,
        n_bootstrap=20,
        random_state=42,
        multi_gpu=False,
    )

    # Compute with multi GPU
    multi_df, multi_var = distance.pairwise(
        adata,
        groupby="group",
        bootstrap=True,
        n_bootstrap=20,
        random_state=42,
        multi_gpu=True,
    )

    np.testing.assert_allclose(
        single_df.values,
        multi_df.values,
        rtol=1e-7,
        atol=1e-7,
        err_msg="Multi-GPU bootstrap mean should match single-GPU",
    )

    np.testing.assert_allclose(
        single_var.values,
        multi_var.values,
        rtol=1e-7,
        atol=1e-7,
        err_msg="Multi-GPU bootstrap variance should match single-GPU",
    )


@pytest.mark.skipif(not _has_multiple_gpus(), reason="Requires 2+ GPUs")
def test_multi_gpu_onesided_bootstrap_matches_single_gpu() -> None:
    """Test that multi-GPU onesided bootstrap produces same results as single-GPU."""
    rng = np.random.default_rng(42)
    n_groups = 4
    cells_per_group = 15
    n_features = 10
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")

    # Compute with forced single GPU
    single_dist, single_var = distance.onesided_distances(
        adata,
        groupby="group",
        selected_group="g0",
        bootstrap=True,
        n_bootstrap=20,
        random_state=42,
        multi_gpu=False,
    )

    # Compute with multi GPU
    multi_dist, multi_var = distance.onesided_distances(
        adata,
        groupby="group",
        selected_group="g0",
        bootstrap=True,
        n_bootstrap=20,
        random_state=42,
        multi_gpu=True,
    )

    np.testing.assert_allclose(
        single_dist.values,
        multi_dist.values,
        rtol=1e-7,
        atol=1e-7,
        err_msg="Multi-GPU onesided bootstrap mean should match single-GPU",
    )

    np.testing.assert_allclose(
        single_var.values,
        multi_var.values,
        rtol=1e-7,
        atol=1e-7,
        err_msg="Multi-GPU onesided bootstrap variance should match single-GPU",
    )


def test_single_gpu_fallback_unchanged(small_adata: AnnData) -> None:
    """Test that single-GPU fallback produces correct results."""
    distance = Distance(metric="edistance")

    # Force single GPU mode using multi_gpu=False
    result = distance.pairwise(small_adata, groupby="group", multi_gpu=False)

    # Verify correctness against CPU reference
    cpu_embedding = small_adata.obsm["X_pca"].get()
    groups = small_adata.obs["group"].values

    for g1, g2 in [("g0", "g1"), ("g0", "g2"), ("g1", "g2")]:
        X = cpu_embedding[groups == g1]
        Y = cpu_embedding[groups == g2]
        expected = _compute_energy_distance_cpu(X, Y)
        actual = result.loc[g1, g2]
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Single-GPU fallback mismatch for ({g1}, {g2})",
        )


def test_small_group_count_works() -> None:
    """Test that computation works correctly with small number of groups."""
    rng = np.random.default_rng(42)
    n_groups = 2  # Only 3 pairs
    cells_per_group = 20
    n_features = 10
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")
    result = distance.pairwise(adata, groupby="group")

    # Result should be correct
    assert result.shape == (2, 2)
    assert result.loc["g0", "g0"] == pytest.approx(0.0, abs=1e-10)
    assert result.loc["g1", "g1"] == pytest.approx(0.0, abs=1e-10)

    # Verify against CPU reference
    cpu_embedding_np = cpu_embedding
    X = cpu_embedding_np[:cells_per_group]
    Y = cpu_embedding_np[cells_per_group:]
    expected = _compute_energy_distance_cpu(X, Y)
    np.testing.assert_allclose(result.loc["g0", "g1"], expected, rtol=1e-6, atol=1e-6)


def test_multi_gpu_with_more_gpus_than_pairs() -> None:
    """Test multi-GPU works when there are more GPUs than pairs.

    Tests that using multi_gpu=True handles cases where some GPUs may get no work.
    """
    rng = np.random.default_rng(42)
    n_groups = 4  # 10 pairs
    cells_per_group = 20
    n_features = 10
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")

    # Use multi_gpu=True which will use all available GPUs
    # The function should handle cases where some GPUs get no work gracefully
    result = distance.pairwise(adata, groupby="group", multi_gpu=True)

    # Just verify we get a valid result
    assert result.shape == (4, 4)
    for i in range(4):
        assert result.iloc[i, i] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.skipif(not _has_multiple_gpus(), reason="Requires 2+ GPUs")
def test_multi_gpu_device_attrs_cache_per_device() -> None:
    """Test that device attributes are cached per device."""
    from rapids_singlecell._utils import _multi_gpu

    # Clear cache to ensure fresh test
    _multi_gpu._DEVICE_ATTRS_CACHE.clear()

    # Get attrs for device 0
    attrs_0 = _multi_gpu._get_device_attrs(device_id=0)
    assert 0 in _multi_gpu._DEVICE_ATTRS_CACHE

    # Get attrs for device 1
    attrs_1 = _multi_gpu._get_device_attrs(device_id=1)
    assert 1 in _multi_gpu._DEVICE_ATTRS_CACHE

    # Both should have required keys
    assert "max_shared_mem" in attrs_0
    assert "cc_major" in attrs_0
    assert "max_shared_mem" in attrs_1
    assert "cc_major" in attrs_1

    # Getting attrs again should return cached values
    attrs_0_again = _multi_gpu._get_device_attrs(device_id=0)
    assert attrs_0_again is attrs_0  # Same object (cached)


def test_multi_gpu_invalid_device_id_list(small_adata: AnnData) -> None:
    """Test that invalid device IDs in list raise ValueError."""
    distance = Distance(metric="edistance")

    n_available = cp.cuda.runtime.getDeviceCount()
    invalid_id = n_available + 10  # Definitely invalid

    with pytest.raises(ValueError, match=r"Invalid GPU device ID\(s\)"):
        distance.pairwise(small_adata, groupby="group", multi_gpu=[0, invalid_id])


def test_multi_gpu_invalid_device_id_string(small_adata: AnnData) -> None:
    """Test that invalid device IDs in string raise ValueError."""
    distance = Distance(metric="edistance")

    n_available = cp.cuda.runtime.getDeviceCount()
    invalid_id = n_available + 10

    with pytest.raises(ValueError, match=r"Invalid GPU device ID\(s\)"):
        distance.pairwise(small_adata, groupby="group", multi_gpu=f"0,{invalid_id}")


def test_multi_gpu_negative_device_id(small_adata: AnnData) -> None:
    """Test that negative device IDs raise ValueError."""
    distance = Distance(metric="edistance")

    with pytest.raises(ValueError, match=r"Invalid GPU device ID\(s\)"):
        distance.pairwise(small_adata, groupby="group", multi_gpu=[-1])


def test_multi_gpu_empty_list(small_adata: AnnData) -> None:
    """Test that empty device list raises ValueError."""
    distance = Distance(metric="edistance")

    with pytest.raises(ValueError, match="must specify at least one device"):
        distance.pairwise(small_adata, groupby="group", multi_gpu=[])


def test_multi_gpu_valid_string_format(small_adata: AnnData) -> None:
    """Test that valid string formats work correctly."""
    distance = Distance(metric="edistance")

    # Single device as string
    result = distance.pairwise(small_adata, groupby="group", multi_gpu="0")
    assert result.shape == (3, 3)

    # With spaces
    result = distance.pairwise(small_adata, groupby="group", multi_gpu=" 0 ")
    assert result.shape == (3, 3)


def test_multi_gpu_valid_list_format(small_adata: AnnData) -> None:
    """Test that valid list formats work correctly."""
    distance = Distance(metric="edistance")

    # Single device as list
    result = distance.pairwise(small_adata, groupby="group", multi_gpu=[0])
    assert result.shape == (3, 3)
