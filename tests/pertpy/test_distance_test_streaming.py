from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
from scipy.spatial.distance import cdist

from rapids_singlecell.ptg import DistanceTest


@pytest.mark.parametrize("budget", [0, 7, True, 8.5])
def test_invalid_distance_memory(budget):
    with pytest.raises((ValueError, TypeError), match="max_distance_memory"):
        DistanceTest("edistance", max_distance_memory=budget)


@pytest.mark.parametrize(
    ("metric", "ground", "n_target", "offdiagonal"),
    [
        ("edistance", "euclidean", 19, False),
        ("edistance", "wide_counts", 1, True),
        ("edistance", "cosine", 32, True),
        ("edistance", "kldivergence", 22, False),
        ("edistance", "kldivergence", 27, True),
        ("mean_pairwise", "kldivergence", 7, False),
    ],
)
def test_tiled_permutation_scores_match_every_shuffle(monkeypatch, metric, ground, n_target, offdiagonal):
    from rapids_singlecell.pertpy_gpu import _pairwise_permutation

    rng = np.random.default_rng(24)
    n = 39
    target = rng.permutation(n) < n_target
    cells = rng.uniform(0.05, 1, (n, 5))
    if ground == "wide_counts":
        counts = rng.integers(0, 80, size=cells.shape, dtype=np.uint64)
        matrix = cdist(counts, counts)
        cells = counts + np.uint64(2**63)
        ground = "euclidean"
    elif ground == "cosine":
        cells[1::9] = 0
        norm = np.linalg.norm(cells, axis=1, keepdims=True)
        unit = cells / np.where(norm == 0, 1, norm)
        matrix = 1 - unit @ unit.T
        cells = sparse.csr_matrix(cells)
    elif ground == "kldivergence":
        from cuml.metrics import pairwise_distances

        cells /= cells.sum(axis=1, keepdims=True)
        matrix = cp.asarray(pairwise_distances(cp.asarray(cells), metric=ground)).get()
        assert not np.allclose(matrix, matrix.T)
    else:
        matrix = cdist(cells, cells)
    np.fill_diagonal(matrix, 0)

    def score(labels):
        cross = matrix[np.ix_(labels, ~labels)].mean()
        if metric == "mean_pairwise":
            return cross
        between = 2 * cross if offdiagonal else cross + matrix[np.ix_(~labels, labels)].mean()
        for side in (labels, ~labels):
            size = int(side.sum())
            between -= matrix[np.ix_(side, side)].sum() / (size * max(size - int(offdiagonal), 1))
        return between

    budget = 2048
    make_reader = _pairwise_permutation.cell_distance_blocks

    def bounded_reader(cells, metric):
        read = make_reader(cells, metric)

        def check(i, end_i, j, end_j):
            assert 8 * (end_i - i) * (end_j - j) <= budget
            return read(i, end_i, j, end_j)

        return check

    monkeypatch.setattr(_pairwise_permutation, "cell_distance_blocks", bounded_reader)
    test = DistanceTest(metric, cell_wise_metric=ground, n_perms=71, max_distance_memory=budget)
    expected_rng = np.random.default_rng(135)
    actual_rng = np.random.default_rng(135)
    expected = [score(target), *(score(expected_rng.permutation(target)) for _ in range(test.n_perms))]
    actual = test._precomputed_distances(cells, target, actual_rng, exclude_diagonal=offdiagonal)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
    assert actual_rng.bit_generator.state == expected_rng.bit_generator.state


def test_tiled_contrast_preserves_tied_zero_scores():
    adata = AnnData(np.eye(48))
    adata.obs["group"] = ["a", "b"] * 24
    contrasts = pd.DataFrame({"group": ["a"], "reference": ["b"]})
    test = DistanceTest("edistance", layer_key="X", n_perms=37, max_distance_memory=4096)
    result = test.contrast_distances(adata, contrasts, show_progressbar=False).iloc[0]
    assert result.edistance == pytest.approx(0, abs=1e-12)
    assert result.pvalue == 1 / test.n_perms


@pytest.mark.parametrize(
    "ground", ["euclidean", "cityblock", "canberra", "hellinger", "jaccard", "dice", "inner_product"]
)
def test_empty_sparse_tiles_match_complete_ground_distances(ground):
    from rapids_singlecell.pertpy_gpu._pairwise_permutation import cell_distance_blocks

    cells = sparse.csr_matrix([[0, 0, 0], [0, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=float)
    read = cell_distance_blocks(cells, ground)
    expected = read(0, 5, 0, 5).get()
    for i, end_i, j, end_j in ((0, 2, 2, 5), (2, 5, 0, 2), (0, 2, 0, 2)):
        np.testing.assert_allclose(read(i, end_i, j, end_j).get(), expected[i:end_i, j:end_j], equal_nan=True)


@pytest.mark.parametrize("automatic", [False, True])
def test_oversized_reference_contrasts_preserve_public_cupy_stream(monkeypatch, automatic):
    from rapids_singlecell.pertpy_gpu import _reference_energy

    rng = np.random.default_rng(12)
    groups = rng.permutation(["reference"] * 512 + ["a"] * 7 + ["b"] * 11)
    adata = AnnData(rng.normal(size=(len(groups), 3)))
    adata.obs["group"] = groups
    contrasts = pd.DataFrame({"group": ["a", "b"], "reference": "reference"})

    def run(budget):
        generator = cp.random.default_rng(12)
        test = DistanceTest("edistance", layer_key="X", random_state=generator, n_perms=67, max_distance_memory=budget)
        result = test.contrast_distances(adata, contrasts, show_progressbar=False)
        return result, generator.integers(2**31, size=32)

    expected, next_draws = run(None)

    budget = 128 << 10
    distances = _reference_energy._distances
    builds = []
    init = _reference_energy.ReferenceEnergy.__init__

    def record_build(self, *args, **kwargs):
        init(self, *args, **kwargs)
        builds.append(self.rr is None)

    def checked_distances(left, right):
        assert 8 * len(left) * len(right) <= budget
        return distances(left, right)

    monkeypatch.setattr(_reference_energy, "_distances", checked_distances)
    monkeypatch.setattr(_reference_energy.ReferenceEnergy, "__init__", record_build)
    if automatic:
        monkeypatch.setattr(DistanceTest, "_available_memory", staticmethod(lambda: budget))
    actual, draws = run(None if automatic else budget)
    assert builds == [True]
    np.testing.assert_allclose(actual.edistance, expected.edistance, atol=2e-12)
    pd.testing.assert_frame_equal(actual.drop(columns="edistance"), expected.drop(columns="edistance"))
    cp.testing.assert_array_equal(draws, next_draws)
