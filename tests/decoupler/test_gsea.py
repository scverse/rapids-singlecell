from __future__ import annotations

import pickle

import cupy as cp
import decoupler as dc_cpu
import numba as nb
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
from anndata import AnnData, read_h5ad
from scipy.stats import false_discovery_control

import rapids_singlecell.decoupler_gpu as dc
from rapids_singlecell.decoupler_gpu import _method_gsea as g


def _permutations(nvar, times, seed):
    permutations = np.tile(np.arange(nvar, dtype=np.int32), (times, 1))
    rng = np.random.default_rng(seed)
    if seed:
        for permutation in permutations:
            rng.shuffle(permutation)
    return permutations


@nb.njit
def _order(row):
    return np.argsort(-row)


@nb.njit
def _walk(values, membership):
    # Independent full walk, including every miss and zero-weight hit.
    mass = 0.0
    for i in range(values.size):
        if membership[i]:
            mass += abs(values[i])
    if mass == 0:
        return 0.0
    if membership.all():
        return 1.0
    decrement = 1.0 / (values.size - membership.sum())
    running = positive = negative = 0.0
    for i in range(values.size):
        running += abs(values[i]) / mass if membership[i] else -decrement
        positive = max(positive, running)
        negative = min(negative, running)
    return positive if positive > -negative else negative


def _reference(mat, sets, times, seed):
    permutations = _permutations(mat.shape[1], times, seed)
    scores = np.zeros((len(mat), len(sets)))
    pvals = np.ones_like(scores)
    for i, row in enumerate(mat.astype(np.float64)):
        order = _order(row)
        for j, targets in enumerate(sets):
            membership = np.isin(order, targets)
            es = scores[i, j] = _walk(row[order], membership)
            if times <= 1 or es == 0:
                continue
            null = np.array([_walk(row[order], membership[p]) for p in permutations])
            same = null[null >= 0] if es >= 0 else null[null < 0]
            mass = np.abs(same).sum()
            scores[i, j] = es * same.size / mass if mass else 0
            if mass:
                pvals[i, j] = np.mean(same >= es if es >= 0 else same <= es)
    return scores, pvals


def _run(mat, sets, **kwargs):
    sizes = np.array([len(targets) for targets in sets])
    return g._func_gsea(
        cp.asarray(mat, dtype=cp.float32),
        cnct=cp.asarray(np.concatenate(sets), dtype=cp.int32),
        starts=cp.asarray(np.r_[0, np.cumsum(sizes)[:-1]], dtype=cp.int32),
        offsets=cp.asarray(sizes, dtype=cp.int32),
        **kwargs,
    )


@pytest.mark.parametrize("nvar", [33, 4096, 4097, 25430, 32768, 49152, 65536, 65537])
def test_rank_preserves_numba_ties_and_inverse_at_storage_boundaries(nvar):
    rng = np.random.default_rng(9901)
    mat = np.zeros((12, nvar), dtype=np.float32)
    positions = rng.choice(nvar, max(2, nvar // 13), replace=False)
    mat[1] = 1
    mat[2, positions] = np.resize([1, 254, 255, 256, 1024, 2**24], positions.size)
    mat[3] = np.log1p(mat[2])
    mat[4, positions] = np.nextafter(np.float32(0), np.float32(1))
    mat[5] = rng.normal(size=nvar)
    mat[6] = 1
    mat[6, [0, (nvar - 1) // 2, nvar - 1]] = 0
    mat[7] = mat[2]
    mat[7, [0, (nvar - 1) // 2, nvar - 1]] = [1, 255, 256]
    mat[8, : nvar // 2] = 1
    mat[9] = -0.0
    mat[9, 0] = -np.nextafter(np.float32(0), np.float32(1))
    for row, leaf in zip(mat[10:], [14, 17], strict=True):
        while leaf <= nvar // 2:
            leaf = 2 * leaf + 1
        row[: nvar - leaf] = 1 + np.arange(nvar - leaf) % 513
    expected = np.stack([_order(row) for row in mat])
    order = cp.empty(mat.shape, dtype=cp.int32)
    ranks = cp.empty_like(order)
    g._gs.rank(cp.asarray(mat), order=order, ranks=ranks)
    np.testing.assert_array_equal(order.get(), expected)
    np.testing.assert_array_equal(ranks.get(), np.argsort(expected, axis=1))


@pytest.mark.parametrize(
    "kind", ["dense", "sparse", "signed", "large", "backed", "regression"]
)
def test_public_outputs_and_batching_match_decoupler(kind, tmp_path):
    rng = np.random.default_rng(825)
    nvar = 4097 if kind == "large" else 2000 if kind == "regression" else 513
    mat = rng.normal(size=(2, nvar)).astype(np.float32)
    sizes = [2050, 31, 3073] if kind == "large" else [5, 9, 17, 33, 129, 257]
    times = 1 if kind == "backed" else 11 if kind == "large" else 31
    if kind == "dense":
        mat = np.abs(mat)
    elif kind == "sparse":
        mat = np.abs(mat) * (rng.random(mat.shape) < 0.1)
    elif kind == "regression":
        # Preserve the RNG stream of the seven rows that exposed FDR errors.
        rng = np.random.default_rng(20260913)
        mat = np.tile(np.geomspace(0.1, 20, nvar).astype(np.float32), (100, 1))
        for row in mat:
            rng.shuffle(row)
        mat *= rng.random(mat.shape) < 0.1
        times = 1000
    sets = []
    for i in range(100 if kind == "regression" else len(sizes)):
        size = int(rng.integers(25, 101)) if kind == "regression" else sizes[i]
        sets.append(rng.choice(nvar, size, replace=False))
    if kind == "regression":
        mat = mat[[4, 7, 18, 28, 40, 50, 76]]
    frame = pd.DataFrame(mat, columns=[f"g{i}" for i in range(nvar)])
    net = pd.DataFrame(
        {
            "source": np.repeat(np.arange(len(sets)).astype(str), list(map(len, sets))),
            "target": frame.columns[np.concatenate(sets)],
        }
    )
    matrix, _, features = dc_cpu.pp.extract(frame, empty=False)
    sources, cnct, starts, offsets = dc_cpu.pp.idxmat(features=features, net=net)
    expected, raw_p = dc_cpu.mt._gsea._func_gsea(
        mat=np.asarray(matrix, dtype=np.float64),
        cnct=cnct,
        starts=starts,
        offsets=offsets,
        times=times,
        seed=42,
    )
    aligned = [cnct[start : start + length] for start, length in zip(starts, offsets)]
    actual, pval = _run(matrix, aligned, times=times, seed=42)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(pval, raw_p)
    data = frame if kind in {"dense", "large"} else AnnData(sps.csr_matrix(mat))
    if isinstance(data, AnnData):
        data.var_names = frame.columns
        if kind == "backed":
            path = tmp_path / "gsea.h5ad"
            data.write_h5ad(path)
            data = read_h5ad(path, backed="r")
    try:
        for bsize in [1, len(mat)]:
            result = dc.gsea(data, net, times=times, tmin=0, empty=False, bsize=bsize)
            score, padj = (
                result if result else (data.obsm["score_gsea"], data.obsm["padj_gsea"])
            )
            np.testing.assert_array_equal(score.columns, sources)
            np.testing.assert_allclose(score, expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(
                padj, false_discovery_control(raw_p, axis=1).astype(np.float32)
            )
    finally:
        if isinstance(data, AnnData) and data.isbacked:
            data.file.close()


@pytest.mark.parametrize("times", [0, 1, 35])
@pytest.mark.parametrize("nvar", [17, 2053])
def test_full_zero_subnormal_and_signed_scores(times, nvar):
    smallest = np.nextafter(np.float32(0), np.float32(1))
    mat = np.zeros((4, nvar), dtype=np.float32)
    mat[:3, :2] = [
        [2 * smallest, smallest],
        [-smallest, -2 * smallest],
        [2 * smallest, -smallest],
    ]
    mat[3] = -0.0
    sets = [np.arange(nvar), [0], [1], [2, 3]]
    expected, expected_p = _reference(mat, sets, times, 0)
    score, pval = _run(mat, sets, times=times, seed=0)
    np.testing.assert_allclose(score, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(pval, expected_p)
    score, _ = _run([[3, 2, 1]], [[1]], times=0)
    np.testing.assert_array_equal(score, [[-0.5]])


@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize(
    "case",
    [
        "product",
        "exact_product",
        "weight",
        "fraction",
        "overflow",
        "subnormal",
        "near_bound",
    ],
)
def test_integer_and_proof_boundaries_against_full_walk(case, seed):
    mat = np.zeros((3, 257), dtype=np.float32)
    sets = [[0, 65]]
    mat[:, :66] = 2048
    mat[:, 65] = 1024
    if case == "product":
        mat[:, 0] = np.iinfo(np.int32).max // 255 + np.arange(-1, 2) - 1024
    elif case == "exact_product":
        mat[:] = 0
        mat[:, 0] = np.float32(2**31 - 128)
        mat[:, 1] = [126, 127, 128]
        sets = [np.arange(256)]
    elif case == "weight":
        limit = np.float32(2**31)
        mat[:, 0] = [
            np.nextafter(limit, np.float32(0)),
            limit,
            np.nextafter(limit, np.float32(np.inf)),
        ]
    elif case == "fraction":
        mat[:, :66] = np.linspace(17.25, 0.125, 66)
    elif case == "overflow":
        mat[:, :66] = np.finfo(np.float32).max * np.linspace(
            1, 0.75, 66, dtype=np.float32
        )
    elif case == "subnormal":
        mat[:, :66] = np.nextafter(np.float32(0), np.float32(1)) * np.arange(66, 0, -1)
    else:
        # Float32 summation alone reverses the sufficient last-hit bound.
        a = np.float32(191 / 64 + 288 * 2**-22)
        mat[:, :66] = 2
        mat[:, 0] = [
            np.nextafter(a, np.float32(0)),
            a,
            np.nextafter(a, np.float32(np.inf)),
        ]
        mat[:, 65] = np.float32(1 + 193 * 2**-23)
    expected, expected_p = _reference(mat, sets, 35, seed)
    score, pval = _run(mat, sets, times=35, seed=seed)
    np.testing.assert_allclose(score, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(pval, expected_p)


@pytest.mark.parametrize("seed", [0, 1, 42, 2**63 + 17])
@pytest.mark.parametrize("nvar", [0, 1, 2, 255, 256, 257, 25430, 65537])
def test_numpy_permutations_are_independent_of_chunk_size(seed, nvar):
    expected = _permutations(nvar, 11, seed)
    for size in [1, 4, 11]:
        actual = np.concatenate(list(g._permutations(nvar, 11, seed, size)))
        np.testing.assert_array_equal(actual, expected)
    assert list(g._permutations(nvar, 0, seed)) == []


@pytest.mark.parametrize("forward", [False, True])
@pytest.mark.parametrize("fits", [False, True])
@pytest.mark.parametrize("nvar,times", [(257, 513), (65537, 129)])
def test_cache_chunk_handoff_budget_and_pickle(forward, fits, nvar, times):
    required = nvar * times * 4 * (1 + forward)
    cache = g._PermutationCache(required if fits else required - 1)
    expected = _permutations(nvar, times, 42)
    producer, consumer = (
        cp.cuda.Stream(non_blocking=True),
        cp.cuda.Stream(non_blocking=True),
    )
    chunks = []

    def consume(inverse, permutations):
        chunks.append(
            (inverse.copy(), None if permutations is None else permutations.copy())
        )

    with producer:
        cache.run(nvar, times, 42, consume, forward=forward)
    first, chunks = chunks, []
    assert all(2 * inverse.nbytes <= 64 * 1024**2 for inverse, _ in first)
    if not fits:
        producer.synchronize()  # Uncached calls do not publish a handoff event.
    with consumer:
        cache.run(nvar, times, 42, consume, forward=forward)
        for batch in [first, chunks]:
            inverse = cp.concatenate([chunk[0] for chunk in batch], axis=1).get(
                stream=consumer
            )
            np.testing.assert_array_equal(inverse, np.argsort(expected, axis=1).T)
            if forward:
                permutations = cp.concatenate([chunk[1] for chunk in batch]).get(
                    stream=consumer
                )
                np.testing.assert_array_equal(permutations, expected)
    assert (cache.entry is not None) == fits
    with cache.lock:
        restored = pickle.loads(pickle.dumps(cache))
    assert restored.entry is None and restored.max_bytes == cache.max_bytes
    assert restored.lock is not cache.lock


@pytest.mark.parametrize("max_bytes", [0, 256 * 1024**2])
def test_streamed_and_cached_scoring_counts_each_permutation_once(
    monkeypatch, max_bytes
):
    mat = np.zeros((2, 257), dtype=np.float32)
    mat[:, :66] = np.linspace(3, 1, 66)
    sets = [np.arange(8), np.arange(257), np.arange(65)]
    expected, expected_p = _reference(mat, sets, 513, 42)
    original = g._gs.sparse
    submitted = {}

    def sparse(*args, **kwargs):
        if "same_count" in kwargs:
            capacity = kwargs["capacity"]
            submitted[capacity] = submitted.get(capacity, 0) + args[7].shape[1]
        return original(*args, **kwargs)

    monkeypatch.setattr(g._gs, "sparse", sparse)
    cache = g._PermutationCache(max_bytes)
    for _ in range(2):
        submitted.clear()
        score, pval = _run(mat, sets, times=513, _permutation_cache=cache)
        np.testing.assert_allclose(score, expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(pval, expected_p)
        assert submitted == {8: 513, 128: 513}


@pytest.mark.parametrize("parameter", ["times", "seed"])
@pytest.mark.parametrize("value", [-1, "invalid", None, np.nan, np.inf])
def test_invalid_parameters(parameter, value):
    with pytest.raises(AssertionError, match=parameter):
        _run([[3, 2, 1]], [[0]], **{parameter: value})
