from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import multipletests

from rapids_singlecell.ptg import Distance, DistanceTest


def _energy(distances, target):
    def within(mask):
        n = mask.sum()
        return distances[np.ix_(mask, mask)].sum() / (n * (n - 1)) if n > 1 else 0

    return 2 * distances[np.ix_(target, ~target)].mean() - within(target) - within(~target)


def test_stratified_contrast_permutations():
    rng = np.random.default_rng(93)
    # Include balanced, unbalanced and singleton comparisons in different strata.
    cells = [
        ("a", "T", 1, 9),
        ("ctrl", "T", 1, 4),
        ("a", "B", 2, 1),
        ("ref2", "B", 2, 6),
        ("a", "T", 2, 7),
        ("ctrl", "T", 2, 7),
        ("ctrl", "T", 3, 4),
        ("a", "other", 1, 3),
    ]
    obs = (
        pd.DataFrame([(g, t, b) for g, t, b, n in cells for _ in range(n)], columns=["drug", "celltype", "batch"])
        .iloc[rng.permutation(sum(n for _, _, _, n in cells))]
        .reset_index(drop=True)
        .rename(str)
    )
    X = rng.normal(size=(len(obs), 4))
    X[(obs.drug == "a") & (obs.batch == 1)] += 4
    X += 1e12  # Sparse Euclidean distances still resolve unit-scale differences.
    adata = AnnData(sparse.csr_matrix(X), obs=obs)
    comparisons = [("a", 2, "ctrl", "T"), ("a", 1, "ctrl", "T"), ("a", 2, "ref2", "B")]
    rows = [
        comparisons[0],
        ("a", 1, "a", "T"),
        comparisons[1],
        comparisons[0],
        comparisons[2],
        ("a", 2, "ctrl", "B"),
        ("ref2", 1, "ref2", "T"),
    ]
    contrasts = pd.DataFrame(rows, columns=["drug", "batch", "reference", "celltype"], index=[8, 8, 3, 1, 5, 9, 6])
    original = contrasts.copy()
    n_perms = 79
    test = DistanceTest("edistance", layer_key="X", n_perms=n_perms, random_state=14, correction="holm")
    actual = test.contrast_distances(adata, contrasts, show_progressbar=False)

    rng = np.random.default_rng(14)
    raw, observed = [], []
    for target, batch, reference, celltype in comparisons:
        keep = (obs.batch == batch) & (obs.celltype == celltype) & obs.drug.isin([target, reference])
        membership = (obs.drug[keep] == target).to_numpy()
        distances = cdist(X[keep], X[keep])
        score = _energy(distances, membership)
        threshold = score + 100 * np.finfo(float).eps * abs(score)
        greater = sum(_energy(distances, rng.permutation(membership)) > threshold for _ in range(n_perms))
        observed.append(score)
        raw.append(max(1, greater) / n_perms)
    rejected, adjusted, _, _ = multipletests(raw, method="holm")
    tested = actual.iloc[[0, 2, 4]]
    direct = Distance("edistance", layer_key="X", precision="float64").contrast_distances(
        adata, contrasts.iloc[[0, 2, 4]], multi_gpu=False
    )
    np.testing.assert_allclose(tested.edistance, direct.edistance, atol=1e-10)
    np.testing.assert_allclose(tested.edistance, observed)
    np.testing.assert_allclose(tested.pvalue, raw)
    np.testing.assert_allclose(tested.pvalue_adj, adjusted)
    np.testing.assert_array_equal(tested.significant, np.array(raw) < 0.05)
    np.testing.assert_array_equal(tested.significant_adj, rejected)
    pd.testing.assert_series_equal(actual.iloc[0], actual.iloc[3], check_names=False)
    assert actual.iloc[1][["edistance", "pvalue", "pvalue_adj"]].tolist() == [0, 1, 1]
    assert actual.iloc[-2:][["edistance", "pvalue", "pvalue_adj"]].isna().all().all()
    assert not actual.iloc[[1, 5, 6]][["significant", "significant_adj"]].any().any()
    pd.testing.assert_frame_equal(actual[contrasts.columns], original)
    pd.testing.assert_frame_equal(contrasts, original)


def test_no_split_gpu_contrasts_match_directional_distance():
    X = cp.asarray(np.random.default_rng(29).normal(size=(24, 4)))
    adata = AnnData(X)
    adata.obs["drug"] = pd.Categorical(["a", "ctrl"] * 12)
    contrasts = pd.DataFrame({"drug": ["a", "ctrl"], "reference": ["ctrl", "a"]})
    direct = Distance("r2_distance", layer_key="X").contrast_distances(adata, contrasts)
    test = DistanceTest("r2_distance", layer_key="X", n_perms=59, random_state=cp.random.default_rng(19))
    actual = test.contrast_distances(adata, contrasts, show_progressbar=False)
    np.testing.assert_allclose(actual.r2_distance, direct.r2_distance)
    assert actual.r2_distance.iloc[0] != pytest.approx(actual.r2_distance.iloc[1])
    oracle = DistanceTest("r2_distance", layer_key="X", n_perms=59, random_state=cp.random.default_rng(19))
    raw = [oracle(adata, "drug", ref, False).loc[target, "pvalue"] for target, ref in contrasts.itertuples(index=False)]
    np.testing.assert_array_equal(actual.pvalue, raw)


@pytest.mark.parametrize("n_target", [3, 509])
def test_imbalanced_energy_contrasts(n_target):
    rng = np.random.default_rng(37)
    X = rng.normal(size=(512, 3))
    labels = rng.permutation(len(X)) < n_target
    adata = AnnData(X)
    adata.obs["group"] = np.where(labels, "a", "ctrl")
    contrasts = Distance.create_contrasts(adata, "group", "ctrl")
    test = DistanceTest("edistance", layer_key="X", n_perms=67, random_state=8)
    actual = test.contrast_distances(adata, contrasts, show_progressbar=False).iloc[0]
    distances = cdist(X, X)
    score = _energy(distances, labels)
    rng = np.random.default_rng(8)
    greater = sum(_energy(distances, rng.permutation(labels)) > score for _ in range(test.n_perms))
    assert actual.edistance == pytest.approx(score)
    assert actual.pvalue == max(greater, 1) / test.n_perms


def test_constant_offdiagonal_contrast_energy_ties():
    # Every distinct pair is sqrt(2) apart, so every E statistic equals zero.
    adata = AnnData(cp.eye(27))
    adata.obs["group"] = ["a"] * 9 + ["ctrl"] * 18
    test = DistanceTest("edistance", layer_key="X", n_perms=71)
    contrasts = Distance.create_contrasts(adata, "group", "ctrl")
    result = test.contrast_distances(adata, contrasts, show_progressbar=False).iloc[0]
    assert result.edistance == pytest.approx(0, abs=1e-12)
    assert result.pvalue == 1 / test.n_perms


def test_empty_and_nonfinite_contrast_families():
    adata = AnnData(np.ones((6, 3)))
    adata.obs["group"] = ["ctrl", "a"] * 3
    test = DistanceTest("pearson_distance", layer_key="X", n_perms=5)
    contrasts = Distance.create_contrasts(adata, "group", "ctrl")
    for table in (contrasts, contrasts.iloc[:0]):
        actual = test.contrast_distances(adata, table, show_progressbar=False)
        assert actual.columns.tolist() == [
            *contrasts.columns,
            "pearson_distance",
            "pvalue",
            "significant",
            "pvalue_adj",
            "significant_adj",
        ]
        pd.testing.assert_index_equal(actual.index, table.index)
        assert actual[["pearson_distance", "pvalue", "pvalue_adj"]].isna().all().all()
        assert not actual[["significant", "significant_adj"]].any().any()

    with pytest.raises(ValueError, match="reference"):
        test.contrast_distances(adata, contrasts.drop(columns="reference"))
    adata.obs["pvalue"] = 1
    with pytest.raises(ValueError, match="pvalue"):
        test.contrast_distances(adata, contrasts.assign(pvalue=1))


def test_repeated_energy_contrasts_reuse_reference(monkeypatch):
    from rapids_singlecell.pertpy_gpu import _reference_energy

    rng = np.random.default_rng(93)
    groups = rng.permutation(["ref"] * 523 + ["a"] * 19 + ["b"] + ["c"] * 9)
    adata = AnnData(rng.normal(size=(len(groups), 4)))
    adata.obs["drug"] = groups
    contrasts = pd.DataFrame({"drug": ["a", "b", "c"], "reference": "ref"})
    seen = []
    distances = _reference_energy._distances

    def record(left, right):
        seen.append((len(left), len(right)))
        return distances(left, right)

    monkeypatch.setattr(_reference_energy, "_distances", record)
    shuffle, expected_rng = np.random.default_rng(45), np.random.default_rng(45)
    test = DistanceTest("edistance", layer_key="X", n_perms=67, random_state=shuffle)
    actual = test.contrast_distances(adata, contrasts, show_progressbar=False)
    assert seen == [(523, 523), (19, 523), (19, 19), (1, 523), (1, 1), (9, 523), (9, 9)]

    for row in actual.itertuples():
        keep = (groups == row.drug) | (groups == "ref")
        membership = groups[keep] == row.drug
        matrix = cdist(adata.X[keep], adata.X[keep])
        score = _energy(matrix, membership)
        cutoff = score + 100 * np.finfo(float).eps * abs(score)
        greater = sum(_energy(matrix, expected_rng.permutation(membership)) > cutoff for _ in range(test.n_perms))
        assert row.edistance == pytest.approx(score, abs=1e-12)
        assert row.pvalue == max(1, greater) / test.n_perms
    assert shuffle.bit_generator.state == expected_rng.bit_generator.state
