from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

from rapids_singlecell.ptg import Distance, DistanceTest


@pytest.mark.parametrize("use_sparse", [False, True], ids=["dense", "csr"])
@pytest.mark.parametrize(
    "metric",
    ["euclidean", "mean_absolute_error", "spearman_distance", "kendalltau_distance", "r2_distance", "mmd"],
)
def test_permuted_means_match_individual_distances(use_sparse, metric):
    rng = np.random.default_rng(75)
    X = rng.integers(0, 4, size=(19, 6)).astype(np.float32)
    group = rng.permutation(np.arange(len(X)) < 7)
    adata = AnnData(sparse.csr_matrix(X) if use_sparse else X)
    adata.obs["group"] = np.where(group, "target", "ctrl")
    test = DistanceTest(metric, layer_key="X", random_state=42, n_perms=55)
    result = test(adata, "group", "ctrl", False).loc["target"]
    distance = Distance(metric)
    shuffle = np.random.default_rng(42)
    cells = cp.asarray(X)
    observed = distance(cells[group], cells[~group])
    scores = []
    for _ in range(test.n_perms):
        mask = shuffle.permutation(group)
        scores.append(distance(cells[mask], cells[~mask]))
    threshold = observed + 100 * np.finfo(float).eps * abs(observed)
    assert result["distance"] == pytest.approx(observed)
    assert result["pvalue"] == max(1, np.count_nonzero(np.asarray(scores) > threshold)) / test.n_perms


def test_permuted_rank_means_preserve_large_offset_ties():
    rng = np.random.default_rng(397)
    group = rng.permutation(np.arange(110) % 3 == 0)
    X = cp.asarray(rng.normal(size=(len(group), 50)) + 1e12)
    test = DistanceTest("kendalltau_distance", n_perms=31)
    actual = test._xy_distances(X, group, np.random.default_rng(42))
    distance = Distance("kendalltau_distance")
    shuffle = np.random.default_rng(42)
    masks = [group, *(shuffle.permutation(group) for _ in range(test.n_perms))]
    expected = [distance(X[mask], X[~mask]) for mask in masks]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kernel", ["rbf", "poly"])
def test_kernel_mmd_permutations_match_separate_kernel_means(kernel):
    X = cp.asarray(np.random.default_rng(5).integers(0, 4, size=(23, 5)), dtype=cp.float64)
    group = np.arange(len(X)) < 9
    test = DistanceTest("mmd", kernel=kernel, gamma=0.2, n_perms=79)
    shuffle = np.random.default_rng(4)
    masks = [group, *(shuffle.permutation(group) for _ in range(test.n_perms))]
    expected = [test.distance(X[mask], X[~mask]) for mask in masks]
    scores = test._xy_distances(X, group, np.random.default_rng(4))
    np.testing.assert_allclose(scores, expected, rtol=1e-12, atol=1e-12)


def test_sparse_nonfinite_value_does_not_contaminate_other_group():
    X = np.array([[1.0, 1], [1, 1], [np.nan, 2], [3, 4], [4, 3]])
    adata = AnnData(sparse.csr_matrix(X))
    adata.obs["group"] = ["a", "a", "ref", "ref", "ref"]
    result = DistanceTest("r2_distance", layer_key="X", n_perms=19)(adata, "group", "ref", False)
    assert result.loc["a", "distance"] == 1  # R² convention for a constant target.
    assert np.isnan(result.loc["a", "pvalue"])


@pytest.mark.parametrize("metric", ["t_test", "sym_kldiv"])
@pytest.mark.parametrize("layout", ["dense", "csr", "csc", "offset"])
def test_permuted_variances_match_direct(metric, layout):
    from cupyx.scipy import sparse as gpu_sparse

    rng = np.random.default_rng(512)
    values = rng.normal(size=(41, 7))
    membership = np.arange(len(values)) % 3 == 0
    # Include an empty feature, a nearly constant group, and tied cells.
    values[:, 0] = 0
    values[membership, 1] = 1 + rng.normal(scale=1e-10, size=membership.sum())
    values[0] = values[3]
    if layout == "offset":
        values += 1e10
    cells = cp.asarray(values)
    inputs = cells
    if layout == "csr":
        inputs = gpu_sparse.csr_matrix(cells)
    elif layout == "csc":
        inputs = gpu_sparse.csc_matrix(cells)
    test = DistanceTest(metric, epsilon=0.01, n_perms=21)
    actual = test._xy_distances(inputs, membership, np.random.default_rng(42))
    shuffle = np.random.default_rng(42)
    masks = [membership, *(shuffle.permutation(membership) for _ in range(test.n_perms))]
    expected = [test.distance(cells[mask], cells[~mask]) for mask in masks]
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
