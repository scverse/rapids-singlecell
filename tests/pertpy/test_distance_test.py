from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import multipletests

from rapids_singlecell.ptg import DistanceTest


def _statistic(metric, cells, mask):
    left, right = cells[mask], cells[~mask]
    if metric == "euclidean":
        return np.linalg.norm(left.mean(axis=0) - right.mean(axis=0))
    between = cdist(left, right, "cityblock").mean()
    if metric == "mean_pairwise":
        return between
    return 2 * between - cdist(left, left, "cityblock").mean() - cdist(right, right, "cityblock").mean()


@pytest.mark.parametrize("metric", ["edistance", "mean_pairwise", "euclidean"])
def test_permutation_matches_cpu(metric):
    # Interleave unequal groups and leave a categorical level unused.
    labels = np.array(["b", "ctrl", "a", "ctrl"] * 4)
    X = np.random.default_rng(17).integers(0, 20, size=(len(labels), 3))
    adata = AnnData(X=sparse.csr_matrix(X))
    adata.obs["group"] = pd.Categorical(labels, categories=["ctrl", "a", "b", "unused"])
    n_perms = 79
    test = DistanceTest(
        metric, n_perms=n_perms, layer_key="X", cell_wise_metric="cityblock", random_state=5, correction="holm"
    )
    result = test(adata, "group", "ctrl", False)
    assert result.index.tolist() == ["b", "ctrl", "a"]
    assert list(result.columns) == ["distance", "pvalue", "significant", "pvalue_adj", "significant_adj"]
    rng = np.random.default_rng(5)
    raw = []
    for group in result.index:
        if group == "ctrl":
            raw.append(1 / n_perms)
            continue
        keep = np.isin(labels, [group, "ctrl"])
        membership = labels[keep] == group
        cells = X[keep]

        observed = _statistic(metric, cells, membership)
        count = sum(_statistic(metric, cells, rng.permutation(membership)) > observed for _ in range(n_perms))
        np.testing.assert_allclose(result.loc[group, "distance"], observed)
        raw.append(max(1, count) / n_perms)
    rejected, adjusted, _, _ = multipletests(raw, method="holm")
    raw[1] = adjusted[1] = 1
    rejected[1] = False
    np.testing.assert_allclose(result["pvalue"], raw)
    np.testing.assert_allclose(result["pvalue_adj"], adjusted)
    np.testing.assert_array_equal(result["significant"], np.array(raw) < 0.05)
    np.testing.assert_array_equal(result["significant_adj"], rejected)
    assert result.loc["ctrl", "distance"] == 0


@pytest.mark.parametrize("metric", ["edistance", "mean_pairwise"])
def test_nonzero_ties(metric):
    # All off-diagonal cell distances equal sqrt(2), independent of labeling.
    adata = AnnData(X=cp.eye(20))
    adata.obs["group"] = ["target"] * 7 + ["ctrl"] * 13
    tied = DistanceTest(metric, layer_key="X", n_perms=64)(adata, "group", "ctrl", False)
    assert tied.loc["target", "distance"] > 0
    assert tied.loc["target", "pvalue"] == 1 / 64


def test_permutation_ties_and_nonfinite_distances():
    adata = AnnData(shape=(6, 3))
    adata.obsm["X_pca"] = cp.ones(adata.shape)
    adata.obs["group"] = ["target", "ctrl"] * 3
    tied = DistanceTest("edistance", n_perms=64)(adata, "group", "ctrl", False)
    assert tied.loc["target", "pvalue"] == 1 / 64
    undefined = DistanceTest("pearson_distance", n_perms=3)(adata, "group", "ctrl", False)
    assert undefined.loc["target", ["distance", "pvalue", "pvalue_adj"]].isna().all()
    assert not undefined.loc["target", ["significant", "significant_adj"]].any()


def test_cosine_distance_with_zero_cells():
    from sklearn.metrics.pairwise import cosine_distances

    counts = sparse.csr_matrix([[0, 0], [0, 0], [0, 1], [1, 0], [1, 1]])
    adata = AnnData(X=counts)
    adata.obs["group"] = ["a", "control", "a", "control", "control"]
    test = DistanceTest("edistance", n_perms=7, layer_key="X", cell_wise_metric="cosine")
    result = test(adata, "group", "control", False)
    left = counts[[0, 2]]
    right = counts[[1, 3, 4]]
    expected = 2 * cosine_distances(left, right).mean() - cosine_distances(left).mean() - cosine_distances(right).mean()
    np.testing.assert_allclose(result.loc["a", "distance"], expected)


def test_permutation_input_errors():
    with pytest.raises(ValueError, match="n_perms"):
        DistanceTest("edistance", n_perms=0)
    with pytest.raises(ValueError, match="classifier_cp"):
        DistanceTest("classifier_cp")
    adata = AnnData(shape=(2, 2))
    adata.obs["group"] = ["ctrl", "test"]
    test = DistanceTest("euclidean")
    with pytest.raises(ValueError, match="missing"):
        test(adata, "group", "missing")
    with pytest.raises(ValueError, match="precomputed"):
        test.test_precomputed(adata, "group", "ctrl")


@pytest.mark.parametrize("precision", ["float16", "float32"])
def test_edistance_permutation_precision(precision):
    with pytest.raises(ValueError, match="permutation tests use float64"):
        DistanceTest("edistance", precision=precision)


@pytest.mark.parametrize("dtype,as_sparse", [(np.uint64, False), (np.int64, True)])
def test_edistance_permutations_preserve_large_count_differences(dtype, as_sparse):
    small = np.array([[1, 3], [5, 4], [0, 2], [3, 9], [1, 0], [9, 2], [2, 0]], dtype=dtype)
    if as_sparse:
        # Individual entries fit in double; summing duplicate coordinates crosses 2**53.
        data = np.tile([2**52, 0, 2**52, 0], len(small)).astype(dtype)
        data[1::2] = small.ravel() + 2**52
        shifted = sparse.csr_matrix(
            (data, np.tile([0, 0, 1, 1], len(small)), np.arange(len(small) + 1) * 4), shape=small.shape
        )
    else:
        shifted = small + dtype(2**63)
    test = DistanceTest("edistance", layer_key="X", precision="float64", n_perms=19)

    def run(x):
        adata = AnnData(x)
        adata.obs["group"] = ["a", "b", "b", "a", "a", "b", "a"]
        contrasts = pd.DataFrame({"group": ["a"], "reference": ["b"]})
        contrast = test.contrast_distances(adata, contrasts, show_progressbar=False).loc[0, ["edistance", "pvalue"]]
        xy = test.test_xy(adata, "group", "b", False).loc["a", ["distance", "pvalue"]]
        return np.array([contrast, xy], dtype=float)

    np.testing.assert_allclose(run(shifted), run(small.astype(float)))


@pytest.mark.parametrize("gpu", [False, True])
def test_edistance_permutations_sum_unsorted_duplicate_counts(gpu):
    from cupyx.scipy import sparse as gpu_sparse

    indices = np.tile([1, 0, 1], 7)
    a = sparse.csr_matrix((np.arange(21, dtype=float), indices, np.arange(0, 22, 3)), shape=(7, 2))
    test = DistanceTest("edistance", layer_key="X", n_perms=29)
    noncanonical = a
    if gpu:
        noncanonical = gpu_sparse.csr_matrix(a)
        noncanonical.indices = noncanonical.indices.astype(cp.int64)
    matrices = [noncanonical, a.toarray()]
    results = []
    for matrix in matrices:
        adata = AnnData(matrix)
        adata.obs["group"] = ["a", "b", "b", "a", "a", "b", "a"]
        results.append(test(adata, "group", "b", False))
    pd.testing.assert_frame_equal(*results)
    assert not a.has_canonical_format


@pytest.mark.parametrize("n_target", [3, 15])
def test_gpu_permutation_batches_have_fixed_sizes_and_advance(n_target):
    membership = np.arange(18) < n_target
    test = DistanceTest("euclidean", n_perms=67)

    def masks(rng):
        batches = test._permutation_batches(membership, rng, bytes_per_row=1000, max_batch=13)
        return cp.concatenate([batch for _, batch in batches]).get()

    rng = cp.random.default_rng(395)
    first = masks(rng)
    np.testing.assert_array_equal(first[0], membership)
    np.testing.assert_array_equal(first.sum(axis=1), np.full(68, n_target))
    np.testing.assert_array_equal(first, masks(cp.random.default_rng(395)))
    assert not np.array_equal(first, masks(rng))


def test_gpu_sampler_redraws_tied_cutoff(monkeypatch):
    partition = cp.partition
    calls = 0

    def tie_once(keys, kth, axis):
        nonlocal calls
        calls += 1
        ranked = partition(keys, kth, axis=axis)
        if calls == 1:
            ranked[0, kth[1]] = ranked[0, kth[0]]
        return ranked

    monkeypatch.setattr(cp, "partition", tie_once)
    test = DistanceTest("euclidean", n_perms=20)
    batches = test._permutation_batches(np.arange(8) < 3, cp.random.default_rng(2), bytes_per_row=100)
    _, result = next(batches)
    assert calls == 2
    np.testing.assert_array_equal(result.sum(axis=1).get(), np.full(21, 3))


def test_small_group_sampler_uniform_over_all_draws():
    from itertools import product

    # Enumerate every 4-draw sequence containing at least two distinct cells.
    draws = np.array([p for p in product(range(4), repeat=4) if len(set(p)) >= 2])
    batches = iter([cp.zeros_like(cp.asarray(draws)), cp.asarray(draws)])

    class FixedRNG:
        def integers(self, n, *, size, dtype):
            return next(batches)

    indices = DistanceTest._small_group_indices(FixedRNG(), len(draws), 4, 2).get()
    pairs, frequencies = np.unique(np.sort(indices, axis=1), axis=0, return_counts=True)
    assert len(pairs) == 6
    np.testing.assert_array_equal(frequencies, np.full(6, len(draws) // 6))
    expected = [sorted(list(dict.fromkeys(row))[:2]) for row in draws]
    np.testing.assert_array_equal(np.sort(indices, axis=1), expected)


@pytest.mark.parametrize("n_target", [1, 17, 8192 - 17])
def test_large_imbalanced_masks_keep_target_size(n_target):
    membership = np.random.default_rng(3).permutation(8192) < n_target
    test = DistanceTest("euclidean", n_perms=7)

    def batches(*, as_indices):
        batches = test._permutation_batches(
            membership, cp.random.default_rng(7), bytes_per_row=8192, max_batch=3, as_indices=as_indices
        )
        return cp.concatenate([rows for _, rows in batches]).get()

    masks = batches(as_indices=False)
    np.testing.assert_array_equal(masks[0], membership)
    np.testing.assert_array_equal(masks.sum(axis=1), np.full(8, n_target))
    selected = masks if n_target < 4096 else ~masks
    np.testing.assert_array_equal(batches(as_indices=True), [np.flatnonzero(row) for row in selected])


@pytest.mark.parametrize(("k", "legacy_bytes"), [(17, 16 * 8192 + 8 * 17**2), (300, 25 * 8192)])
def test_compact_indices_preserve_cupy_draws_across_batches(k, legacy_bytes):
    membership = np.random.default_rng(9).permutation(8192) < k
    boundary = (64 << 20) // legacy_bytes
    test = DistanceTest("edistance", n_perms=boundary + 5)
    legacy_rng, compact_rng = cp.random.default_rng(17), cp.random.default_rng(17)
    expected = test._permutation_batches(membership, legacy_rng, bytes_per_row=legacy_bytes)
    actual = test._permutation_batches(membership, compact_rng, bytes_per_row=32 * k, as_indices=True)
    starts = []
    for (old_start, masks), (new_start, indices) in zip(expected, actual, strict=True):
        starts.append(old_start)
        assert new_start == old_start
        cp.testing.assert_array_equal(indices, cp.nonzero(masks)[1].reshape(-1, k))
    assert starts == [0, boundary]
    cp.testing.assert_array_equal(legacy_rng.integers(2**31, size=256), compact_rng.integers(2**31, size=256))


def test_gpu_rng_reaches_scalar_metric():
    from rapids_singlecell.ptg import Distance

    rng = np.random.default_rng(19)
    X = cp.asarray(rng.normal(size=(17, 5)))
    group = np.arange(len(X)) < 7
    adata = AnnData(X)
    adata.obs["group"] = np.where(group, "target", "ref")
    test = DistanceTest("ks_test", layer_key="X", random_state=cp.random.default_rng(9), n_perms=31)
    result = test(adata, "group", "ref", False).loc["target"]
    _, masks = next(test._permutation_batches(group, cp.random.default_rng(9), bytes_per_row=100))
    distance = Distance("ks_test")
    scores = np.array([distance(X[mask], X[~mask]) for mask in masks])
    assert result["distance"] == pytest.approx(scores[0])
    assert result["pvalue"] == max(1, np.count_nonzero(scores[1:] > scores[0])) / test.n_perms


@pytest.mark.parametrize(("metric", "n_target"), [("edistance", 12), ("mean_pairwise", 500), ("mmd", 12)])
def test_imbalanced_permutations_against_precomputed_reference(metric, n_target):
    n = 512
    rng = np.random.default_rng(14)
    cells = rng.normal(size=(n, 4))
    target = rng.permutation(n) < n_target
    adata = AnnData(X=cells)
    adata.obs["group"] = np.where(target, "target", "control")
    n_perms = 67
    kwargs = {"kernel": "rbf", "gamma": 0.5} if metric == "mmd" else {}
    test = DistanceTest(metric, n_perms=n_perms, layer_key="X", random_state=12, **kwargs)
    actual = test(adata, "group", "control", False).loc["target"]
    matrix = cdist(cells, cells)
    if metric == "mmd":
        matrix = np.exp(-0.5 * matrix**2)

    def score(mask):
        cross = matrix[np.ix_(mask, ~mask)].mean()
        if metric == "mean_pairwise":
            return cross
        within = matrix[np.ix_(mask, mask)].mean() + matrix[np.ix_(~mask, ~mask)].mean()
        return 2 * cross - within if metric == "edistance" else within - 2 * cross

    rng = np.random.default_rng(12)
    observed = score(target)
    greater = sum(score(rng.permutation(target)) > observed for _ in range(n_perms))
    np.testing.assert_allclose(actual["distance"], observed)
    assert actual["pvalue"] == max(greater, 1) / n_perms
