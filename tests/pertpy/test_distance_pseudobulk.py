from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from rapids_singlecell.pertpy_gpu import Distance

PSEUDOBULK_METRICS = (
    "euclidean",
    "root_mean_squared_error",
    "mse",
    "mean_absolute_error",
    "pearson_distance",
    "cosine_distance",
    "r2_distance",
)

TRUE_PSEUDOBULK_METRICS = (
    "euclidean",
    "root_mean_squared_error",
    "mean_absolute_error",
)


def _pseudobulk_reference(metric: str, mu_X: np.ndarray, mu_Y: np.ndarray) -> float:
    if metric in ("euclidean", "root_mean_squared_error"):
        return float(np.linalg.norm(mu_X - mu_Y))
    if metric == "mse":
        return float(np.mean((mu_X - mu_Y) ** 2))
    if metric == "mean_absolute_error":
        return float(np.mean(np.abs(mu_X - mu_Y)))
    if metric == "pearson_distance":
        return float(1 - np.corrcoef(mu_X, mu_Y)[0, 1])
    if metric == "cosine_distance":
        nx = float(np.linalg.norm(mu_X))
        ny = float(np.linalg.norm(mu_Y))
        return float(np.clip(1 - np.dot(mu_X, mu_Y) / (nx * ny), 0, 2))
    if metric == "r2_distance":
        ss_res = float(np.sum((mu_X - mu_Y) ** 2))
        ss_tot = float(np.sum((mu_X - np.mean(mu_X)) ** 2))
        if ss_tot == 0:
            return 0.0 if ss_res == 0 else 1.0
        return ss_res / ss_tot
    raise ValueError(metric)


def _group_means(X: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, list[str]]:
    df = pd.DataFrame(X)
    df["__g"] = groups
    g = df.groupby("__g").mean().sort_index()
    return g.to_numpy(), list(g.index)


@pytest.fixture
def pseudobulk_adata() -> tuple[AnnData, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(13)
    n_groups, cells_per_group, n_features = 4, 6, 7
    blocks = [
        rng.normal(loc=i * 0.5, size=(cells_per_group, n_features))
        for i in range(n_groups)
    ]
    X = np.vstack(blocks).astype(np.float32)
    groups = np.array(
        [f"g{i}" for i in range(n_groups) for _ in range(cells_per_group)]
    )
    obs = pd.DataFrame({"group": pd.Categorical(groups)})
    adata = AnnData(X.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(X)
    return adata, X, groups


@pytest.mark.parametrize("metric", PSEUDOBULK_METRICS)
def test_pseudobulk_call_matches_reference(
    metric: str,
    pseudobulk_adata: tuple[AnnData, np.ndarray, np.ndarray],
) -> None:
    _, X, groups = pseudobulk_adata
    A = X[groups == "g0"]
    B = X[groups == "g1"]
    expected = _pseudobulk_reference(metric, A.mean(axis=0), B.mean(axis=0))
    actual = Distance(metric=metric)(A, B)
    assert isinstance(actual, float)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("metric", PSEUDOBULK_METRICS)
def test_pseudobulk_pairwise_matches_reference(
    metric: str,
    pseudobulk_adata: tuple[AnnData, np.ndarray, np.ndarray],
) -> None:
    adata, X, groups = pseudobulk_adata
    means, names = _group_means(X, groups)
    result = Distance(metric=metric).pairwise(adata, groupby="group")

    assert list(result.index) == names
    assert list(result.columns) == names

    values = result.values
    np.testing.assert_allclose(np.diag(values), 0, atol=1e-7)
    np.testing.assert_allclose(values, values.T, rtol=1e-6, atol=1e-7)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            expected = _pseudobulk_reference(metric, means[i], means[j])
            np.testing.assert_allclose(values[i, j], expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("metric", TRUE_PSEUDOBULK_METRICS)
def test_pseudobulk_pairwise_triangle_inequality(
    metric: str,
    pseudobulk_adata: tuple[AnnData, np.ndarray, np.ndarray],
) -> None:
    adata, _, _ = pseudobulk_adata
    result = Distance(metric=metric).pairwise(adata, groupby="group")

    rng = np.random.default_rng(42)
    for _ in range(5):
        i, j, k = rng.choice(result.index, size=3, replace=False)
        lhs = result.loc[i, j] + result.loc[j, k]
        rhs = result.loc[i, k]
        assert lhs + 1e-6 >= rhs


@pytest.mark.parametrize("metric", PSEUDOBULK_METRICS)
def test_pseudobulk_onesided_matches_reference(
    metric: str,
    pseudobulk_adata: tuple[AnnData, np.ndarray, np.ndarray],
) -> None:
    adata, X, groups = pseudobulk_adata
    means, names = _group_means(X, groups)
    ref_idx = names.index("g0")

    result = Distance(metric=metric).onesided_distances(
        adata, groupby="group", selected_group="g0"
    )

    assert isinstance(result, pd.Series)
    assert result.loc["g0"] == pytest.approx(0.0, abs=1e-7)
    for i, name in enumerate(names):
        if name == "g0":
            continue
        expected = _pseudobulk_reference(metric, means[i], means[ref_idx])
        np.testing.assert_allclose(result.loc[name], expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("metric", PSEUDOBULK_METRICS)
def test_pseudobulk_groups_parameter_filters(
    metric: str,
    pseudobulk_adata: tuple[AnnData, np.ndarray, np.ndarray],
) -> None:
    adata, _, _ = pseudobulk_adata
    result = Distance(metric=metric).pairwise(
        adata, groupby="group", groups=["g0", "g2"]
    )
    assert list(result.index) == ["g0", "g2"]
    assert list(result.columns) == ["g0", "g2"]


@pytest.mark.parametrize("metric", PSEUDOBULK_METRICS)
def test_pseudobulk_multi_gpu_falls_back_with_warning(
    metric: str,
    pseudobulk_adata: tuple[AnnData, np.ndarray, np.ndarray],
) -> None:
    adata, _, _ = pseudobulk_adata
    with pytest.warns(UserWarning, match="does not support multi-GPU"):
        Distance(metric=metric).pairwise(adata, groupby="group", multi_gpu=True)


@pytest.mark.parametrize(
    "input_kind",
    [
        "X_dense",
        "X_sparse",
        "obsm_numpy",
        "obsm_dataframe",
        "layer_dense",
        "layer_sparse",
    ],
)
def test_pseudobulk_input_variants(input_kind: str) -> None:
    rng = np.random.default_rng(7)
    n_features = 5
    X = rng.normal(size=(24, n_features)).astype(np.float32)
    groups = np.array(["a", "b", "c"] * 8)
    obs = pd.DataFrame({"group": pd.Categorical(groups)})
    adata = AnnData(X.copy(), obs=obs)

    layer_key, obsm_key = None, None
    if input_kind == "X_dense":
        adata.X = cp.asarray(X)
        layer_key = "X"
    elif input_kind == "X_sparse":
        adata.X = sparse.csr_matrix(X)
        layer_key = "X"
    elif input_kind == "obsm_numpy":
        adata.obsm["X_pca"] = X.copy()
        obsm_key = "X_pca"
    elif input_kind == "obsm_dataframe":
        adata.obsm["X_pca"] = pd.DataFrame(
            X, index=adata.obs_names, columns=[f"PC{i}" for i in range(n_features)]
        )
        obsm_key = "X_pca"
    elif input_kind == "layer_dense":
        adata.layers["counts"] = cp.asarray(X)
        layer_key = "counts"
    elif input_kind == "layer_sparse":
        adata.layers["counts"] = sparse.csr_matrix(X)
        layer_key = "counts"

    result = (
        Distance(metric="euclidean", layer_key=layer_key, obsm_key=obsm_key)
        .pairwise(adata, groupby="group")
        .values
    )

    means, _ = _group_means(X, groups)
    k = len(means)
    expected = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                expected[i, j] = _pseudobulk_reference("euclidean", means[i], means[j])
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("metric", PSEUDOBULK_METRICS)
def test_pseudobulk_contrast_distances_matches_reference(metric: str) -> None:
    rng = np.random.default_rng(17)
    n_features, per_cond = 6, 8
    treatments = ["ctrl", "drugA", "drugB"]
    celltypes = ["T", "B"]

    blocks, obs_rows = [], []
    for ti, t in enumerate(treatments):
        for ci, c in enumerate(celltypes):
            blocks.append(
                rng.normal(loc=ti * 0.7 + ci * 0.2, size=(per_cond, n_features))
            )
            obs_rows.extend({"treatment": t, "celltype": c} for _ in range(per_cond))
    X = np.vstack(blocks).astype(np.float32)
    obs = pd.DataFrame(obs_rows).astype(
        {"treatment": "category", "celltype": "category"}
    )
    adata = AnnData(X.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(X)

    contrasts = Distance.create_contrasts(
        adata, groupby="treatment", selected_group="ctrl", split_by="celltype"
    )
    result = Distance(metric=metric).contrast_distances(adata, contrasts=contrasts)

    assert metric in result.columns
    treatments_col = adata.obs["treatment"].to_numpy()
    celltypes_col = adata.obs["celltype"].to_numpy()
    for _, row in result.iterrows():
        mask = celltypes_col == row["celltype"]
        target = X[mask & (treatments_col == row["treatment"])].mean(axis=0)
        ref = X[mask & (treatments_col == row["reference"])].mean(axis=0)
        expected = _pseudobulk_reference(metric, target, ref)
        np.testing.assert_allclose(row[metric], expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("metric", PSEUDOBULK_METRICS)
def test_pseudobulk_contrast_self_distance_is_zero(
    metric: str,
    pseudobulk_adata: tuple[AnnData, np.ndarray, np.ndarray],
) -> None:
    adata, _, _ = pseudobulk_adata
    contrasts = pd.DataFrame({"group": ["g0"], "reference": ["g0"]})
    result = Distance(metric=metric).contrast_distances(adata, contrasts=contrasts)
    assert result[metric].iloc[0] == pytest.approx(0.0, abs=1e-7)


_KERNEL_FEATURE_SIZES = [1, 3, 8, 17, 64, 257]


@pytest.mark.parametrize("n_features", _KERNEL_FEATURE_SIZES)
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_paired_squared_kernel_matches_cupy(n_features: int, dtype) -> None:
    from rapids_singlecell.pertpy_gpu._metrics._utils._pseudobulk import (
        paired_squared,
    )

    rng = cp.random.default_rng(0)
    X = rng.standard_normal((5, n_features), dtype=dtype)
    Y = rng.standard_normal((5, n_features), dtype=dtype)
    expected = cp.sum((X - Y) ** 2, axis=1)
    cp.testing.assert_allclose(paired_squared(X, Y), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_features", _KERNEL_FEATURE_SIZES)
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_pairwise_squared_kernel_matches_cupy(n_features: int, dtype) -> None:
    from rapids_singlecell.pertpy_gpu._metrics._utils._pseudobulk import (
        pairwise_squared,
    )

    rng = cp.random.default_rng(0)
    X = rng.standard_normal((4, n_features), dtype=dtype)
    Y = rng.standard_normal((6, n_features), dtype=dtype)
    expected = cp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    cp.testing.assert_allclose(pairwise_squared(X, Y), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_features", _KERNEL_FEATURE_SIZES)
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_paired_abs_mean_kernel_matches_cupy(n_features: int, dtype) -> None:
    from rapids_singlecell.pertpy_gpu._metrics._utils._pseudobulk import (
        paired_abs_mean,
    )

    rng = cp.random.default_rng(0)
    X = rng.standard_normal((5, n_features), dtype=dtype)
    Y = rng.standard_normal((5, n_features), dtype=dtype)
    expected = cp.mean(cp.abs(X - Y), axis=1)
    cp.testing.assert_allclose(paired_abs_mean(X, Y), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_features", _KERNEL_FEATURE_SIZES)
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_pairwise_abs_mean_kernel_matches_cupy(n_features: int, dtype) -> None:
    from rapids_singlecell.pertpy_gpu._metrics._utils._pseudobulk import (
        pairwise_abs_mean,
    )

    rng = cp.random.default_rng(0)
    X = rng.standard_normal((4, n_features), dtype=dtype)
    Y = rng.standard_normal((6, n_features), dtype=dtype)
    expected = cp.mean(cp.abs(X[:, None, :] - Y[None, :, :]), axis=2)
    cp.testing.assert_allclose(pairwise_abs_mean(X, Y), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("metric", PSEUDOBULK_METRICS)
def test_pseudobulk_pairwise_cross_check_pertpy(
    metric: str,
    pseudobulk_adata: tuple[AnnData, np.ndarray, np.ndarray],
) -> None:
    pt = pytest.importorskip("pertpy")
    adata, X, _ = pseudobulk_adata
    cpu_adata = AnnData(X.copy(), obs=adata.obs.copy())
    cpu_adata.obsm["X_pca"] = X.copy()

    expected = pt.tl.Distance(metric, obsm_key="X_pca").pairwise(
        cpu_adata, groupby="group", show_progressbar=False
    )
    actual = Distance(metric=metric).pairwise(adata, groupby="group")
    expected = expected.loc[actual.index, actual.columns]
    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-5, atol=1e-5)


def _make_adata(X: np.ndarray, group_sizes: list[int]) -> AnnData:
    groups = np.concatenate([np.repeat(f"g{i}", n) for i, n in enumerate(group_sizes)])
    obs = pd.DataFrame({"group": pd.Categorical(groups)})
    adata = AnnData(X.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(X)
    return adata


def test_pseudobulk_cosine_zero_norm_group_yields_nan() -> None:
    """Cosine matches scipy/pertpy: a zero-norm mean vector produces NaN.

    Documenting this deliberately — we don't guard the degenerate case so
    that our output matches ``scipy.spatial.distance.cosine`` exactly.
    """
    rng = np.random.default_rng(0)
    n_features = 5
    X = np.vstack(
        [
            np.zeros((4, n_features), dtype=np.float32),
            rng.normal(size=(4, n_features)).astype(np.float32),
            rng.normal(loc=1.0, size=(4, n_features)).astype(np.float32),
        ]
    )
    adata = _make_adata(X, [4, 4, 4])

    result = Distance(metric="cosine_distance").pairwise(adata, groupby="group").values
    assert np.isnan(result[0, 1])
    assert np.isnan(result[0, 2])
    assert np.isfinite(result[1, 2])


def test_pseudobulk_pearson_constant_mean_group_yields_nan() -> None:
    """Pearson matches scipy/pertpy: a constant mean vector produces NaN.

    Documenting this deliberately — we don't guard the degenerate case so
    that our output matches ``scipy.stats.pearsonr`` exactly.
    """
    rng = np.random.default_rng(0)
    n_features = 5
    X = np.vstack(
        [
            np.full((4, n_features), 0.7, dtype=np.float32),
            rng.normal(size=(4, n_features)).astype(np.float32),
            rng.normal(loc=1.0, size=(4, n_features)).astype(np.float32),
        ]
    )
    adata = _make_adata(X, [4, 4, 4])

    result = Distance(metric="pearson_distance").pairwise(adata, groupby="group").values
    assert np.isnan(result[0, 1])
    assert np.isnan(result[0, 2])
    assert np.isfinite(result[1, 2])


@pytest.mark.parametrize("metric", PSEUDOBULK_METRICS)
def test_pseudobulk_single_group(metric: str) -> None:
    """K=1 pairwise returns a 1x1 zero matrix; onesided returns a scalar zero."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 5)).astype(np.float32)
    adata = _make_adata(X, [10])

    pw = Distance(metric=metric).pairwise(adata, groupby="group").values
    assert pw.shape == (1, 1)
    assert pw[0, 0] == pytest.approx(0.0, abs=1e-7)

    os = Distance(metric=metric).onesided_distances(
        adata, groupby="group", selected_group="g0"
    )
    assert os.loc["g0"] == pytest.approx(0.0, abs=1e-7)


# ============================================================================
# Bootstrap
# ============================================================================

# r2_distance is asymmetric in onesided (ss_tot keyed off the row group), so
# its onesided bootstrap is not the symmetrized pairwise row.
SYMMETRIC_PSEUDOBULK_METRICS = tuple(
    m for m in PSEUDOBULK_METRICS if m != "r2_distance"
)


@pytest.fixture
def bootstrap_adata() -> tuple[AnnData, np.ndarray, np.ndarray]:
    """Larger, well-separated dataset for stable bootstrap statistics."""
    rng = np.random.default_rng(11)
    n_groups, cells_per_group, n_features = 3, 60, 6
    blocks = [
        rng.normal(loc=i * 0.8, size=(cells_per_group, n_features))
        for i in range(n_groups)
    ]
    X = np.vstack(blocks).astype(np.float32)
    groups = np.array(
        [f"g{i}" for i in range(n_groups) for _ in range(cells_per_group)]
    )
    obs = pd.DataFrame({"group": pd.Categorical(groups)})
    adata = AnnData(X.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(X)
    return adata, X, groups


@pytest.mark.parametrize("metric", PSEUDOBULK_METRICS)
def test_pseudobulk_pairwise_bootstrap_structure(
    metric: str,
    bootstrap_adata: tuple[AnnData, np.ndarray, np.ndarray],
) -> None:
    adata, _, _ = bootstrap_adata
    result = Distance(metric=metric).pairwise(
        adata, groupby="group", bootstrap=True, n_bootstrap=20, random_state=42
    )

    assert isinstance(result, tuple)
    distances, variances = result
    assert isinstance(distances, pd.DataFrame)
    assert isinstance(variances, pd.DataFrame)
    assert distances.shape == variances.shape == (3, 3)
    assert list(distances.index) == list(variances.index)

    # Self-distance mean and variance are 0; variances are non-negative.
    np.testing.assert_allclose(np.diag(distances.values), 0, atol=1e-7)
    np.testing.assert_allclose(np.diag(variances.values), 0, atol=1e-7)
    assert np.all(variances.values >= 0)
    # Off-diagonal variance is strictly positive for separated groups.
    off_diag = ~np.eye(3, dtype=bool)
    assert np.all(variances.values[off_diag] > 0)


@pytest.mark.parametrize("metric", PSEUDOBULK_METRICS)
def test_pseudobulk_bootstrap_reproducible(
    metric: str,
    bootstrap_adata: tuple[AnnData, np.ndarray, np.ndarray],
) -> None:
    adata, _, _ = bootstrap_adata
    dist = Distance(metric=metric)
    m1, v1 = dist.pairwise(
        adata, groupby="group", bootstrap=True, n_bootstrap=15, random_state=7
    )
    m2, v2 = dist.pairwise(
        adata, groupby="group", bootstrap=True, n_bootstrap=15, random_state=7
    )
    # Same seed reproduces results up to GPU float-accumulation noise.
    np.testing.assert_allclose(m1.values, m2.values, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(v1.values, v2.values, rtol=1e-5, atol=1e-6)

    # A different seed gives a genuinely different resample.
    m3, _ = dist.pairwise(
        adata, groupby="group", bootstrap=True, n_bootstrap=15, random_state=8
    )
    assert not np.allclose(m1.values, m3.values)


@pytest.mark.parametrize("metric", PSEUDOBULK_METRICS)
def test_pseudobulk_onesided_bootstrap_returns_tuple(
    metric: str,
    bootstrap_adata: tuple[AnnData, np.ndarray, np.ndarray],
) -> None:
    adata, _, _ = bootstrap_adata
    result = Distance(metric=metric).onesided_distances(
        adata,
        groupby="group",
        selected_group="g0",
        bootstrap=True,
        n_bootstrap=20,
        random_state=42,
    )
    assert isinstance(result, tuple)
    distances, variances = result
    assert isinstance(distances, pd.Series)
    assert isinstance(variances, pd.Series)
    assert len(distances) == len(variances) == 3

    assert distances.loc["g0"] == pytest.approx(0.0, abs=1e-7)
    assert variances.loc["g0"] == pytest.approx(0.0, abs=1e-7)
    assert np.all(variances.values >= 0)


def test_pseudobulk_onesided_bootstrap_multiple_controls(
    bootstrap_adata: tuple[AnnData, np.ndarray, np.ndarray],
) -> None:
    adata, _, _ = bootstrap_adata
    distances, variances = Distance(metric="euclidean").onesided_distances(
        adata,
        groupby="group",
        selected_group=["g0", "g1"],
        bootstrap=True,
        n_bootstrap=15,
        random_state=3,
    )
    assert isinstance(distances, pd.DataFrame)
    assert isinstance(variances, pd.DataFrame)
    assert list(distances.columns) == ["g0", "g1"]
    assert distances.loc["g0", "g0"] == pytest.approx(0.0, abs=1e-7)
    assert distances.loc["g1", "g1"] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("metric", SYMMETRIC_PSEUDOBULK_METRICS)
def test_pseudobulk_onesided_bootstrap_matches_pairwise(
    metric: str,
    bootstrap_adata: tuple[AnnData, np.ndarray, np.ndarray],
) -> None:
    """With the same seed, the onesided row equals the pairwise column
    (symmetric metrics): both resample identical group means."""
    adata, _, _ = bootstrap_adata
    dist = Distance(metric=metric)
    pw_mean, pw_var = dist.pairwise(
        adata, groupby="group", bootstrap=True, n_bootstrap=20, random_state=42
    )
    os_mean, os_var = dist.onesided_distances(
        adata,
        groupby="group",
        selected_group="g0",
        bootstrap=True,
        n_bootstrap=20,
        random_state=42,
    )
    np.testing.assert_allclose(
        os_mean.values, pw_mean.loc[:, "g0"].values, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        os_var.values, pw_var.loc[:, "g0"].values, rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("metric", TRUE_PSEUDOBULK_METRICS)
def test_pseudobulk_bootstrap_mean_close_to_point_estimate(
    metric: str,
    bootstrap_adata: tuple[AnnData, np.ndarray, np.ndarray],
) -> None:
    """Bootstrap mean tracks the point estimate for low-bias metrics."""
    adata, _, _ = bootstrap_adata
    dist = Distance(metric=metric)
    point = dist.pairwise(adata, groupby="group")
    boot_mean, _ = dist.pairwise(
        adata, groupby="group", bootstrap=True, n_bootstrap=100, random_state=0
    )
    off_diag = ~np.eye(3, dtype=bool)
    np.testing.assert_allclose(
        boot_mean.values[off_diag], point.values[off_diag], rtol=0.3
    )


@pytest.mark.parametrize("metric", PSEUDOBULK_METRICS)
def test_pseudobulk_bootstrap_arrays(
    metric: str,
    bootstrap_adata: tuple[AnnData, np.ndarray, np.ndarray],
) -> None:
    """Distance.bootstrap(X, Y) returns a reproducible MeanVar for arrays."""
    _, X, groups = bootstrap_adata
    A = X[groups == "g0"]
    B = X[groups == "g1"]
    dist = Distance(metric=metric)

    result = dist.bootstrap(A, B, n_bootstrap=50, random_state=1)
    assert result.variance >= 0

    again = dist.bootstrap(A, B, n_bootstrap=50, random_state=1)
    assert result.mean == again.mean
    assert result.variance == again.variance


def test_pseudobulk_bootstrap_rejects_nonpositive_n_bootstrap(
    bootstrap_adata: tuple[AnnData, np.ndarray, np.ndarray],
) -> None:
    adata, X, groups = bootstrap_adata
    dist = Distance(metric="euclidean")
    with pytest.raises(ValueError, match="n_bootstrap"):
        dist.pairwise(adata, groupby="group", bootstrap=True, n_bootstrap=0)
    with pytest.raises(ValueError, match="n_bootstrap"):
        dist.onesided_distances(
            adata, groupby="group", selected_group="g0", bootstrap=True, n_bootstrap=0
        )
    with pytest.raises(ValueError, match="n_bootstrap"):
        dist.bootstrap(X[groups == "g0"], X[groups == "g1"], n_bootstrap=0)
