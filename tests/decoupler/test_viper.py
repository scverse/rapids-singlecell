from __future__ import annotations

import cupy as cp
import cupyx.scipy.sparse as csps
import decoupler as dc_cpu
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
import scipy.special
from anndata import AnnData, read_h5ad
from scipy.stats import false_discovery_control, norm, rankdata

import rapids_singlecell.decoupler_gpu as dc
from rapids_singlecell.decoupler_gpu._method_viper import _correct, _func_viper


def _run(mat, adj, **kwargs):
    return _func_viper(cp.asarray(mat), cp.asarray(adj), **kwargs)


def _assert_parity(mat, adj, **kwargs):
    expected, expected_p = dc_cpu.mt._viper._func_viper(mat, adj, **kwargs)
    score, pval = _run(mat, adj, **kwargs)
    assert isinstance(score, np.ndarray)
    assert isinstance(pval, np.ndarray)
    np.testing.assert_allclose(score, expected, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(pval, expected_p, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(pval, 2 * norm.sf(np.abs(score)), rtol=2e-6)
    return score, pval


@pytest.fixture
def pleiotropy_data():
    # Overlaps of 12 and 16 targets, with significant scores of both signs.
    values = np.arange(100)
    mat = np.stack([values[::-1], values, np.roll(values, 15)]).astype(np.float32)
    adj = np.zeros((100, 3), dtype=np.float32)
    adj[:20, 0] = 1
    adj[8:28, 1] = 0.8
    adj[12:32, 2] = -0.6
    return mat, adj


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nvar", [11, 257])
def test_weighted_signed_scores_with_ties(dtype, nvar):
    rng = np.random.default_rng(730)
    mat = rng.integers(-3, 4, size=(7, nvar)).astype(dtype)
    mat[0] = 0
    mat[1] = 3
    mat[2, : nvar // 2] = 0
    adj = rng.choice([0, 0.25, -0.75, 1.5, -3], size=(nvar, 3)).astype(dtype)
    # Unconnected features still belong to the ranking universe.
    adj[-2:] = 0
    score, pval = _assert_parity(mat, adj, pleiotropy=False)
    np.testing.assert_array_equal(score[:2], 0)
    np.testing.assert_array_equal(pval[:2], 1)


def test_all_constant_observations():
    mat = np.array([[0] * 7, [2] * 7, [-1] * 7], dtype=np.float32)
    adj = np.array([[1, -0.5]] * 7, dtype=np.float32)
    score, pval = _assert_parity(mat, adj)
    np.testing.assert_array_equal(score, 0)
    np.testing.assert_array_equal(pval, 1)


@pytest.mark.parametrize("kwargs", [{}, {"penalty": 2.5}, {"reg_sign": 0.001}])
def test_pleiotropy_refines_significant_overlapping_regulators(pleiotropy_data, kwargs):
    mat, adj = pleiotropy_data
    initial, initial_p = _run(mat, adj, pleiotropy=False)
    assert np.all(initial_p[0] < 0.05)
    refined, _ = _assert_parity(mat, adj, **kwargs)
    assert np.max(np.abs(refined - initial)) > 0.1


def test_pleiotropy_requires_strictly_more_than_n_targets(pleiotropy_data):
    mat, adj = pleiotropy_data
    mat, adj = mat[:1], adj[:, :2]
    assert np.count_nonzero(np.all(adj != 0, axis=1)) == 12
    initial, _ = _run(mat, adj, pleiotropy=False)
    included, _ = _assert_parity(mat, adj, n_targets=11)
    excluded, _ = _assert_parity(mat, adj, n_targets=12)
    assert np.max(np.abs(included - initial)) > 0.1
    np.testing.assert_array_equal(excluded, initial)


def test_pleiotropy_without_significant_pairs(pleiotropy_data):
    mat, adj = pleiotropy_data
    initial, _ = _run(mat, adj, pleiotropy=False)
    score, _ = _assert_parity(mat, adj, reg_sign=1e-12)
    np.testing.assert_array_equal(score, initial)
    # A single significant regulator cannot form a pair.
    score, _ = _assert_parity(mat, adj[:, :1])
    np.testing.assert_allclose(score, initial[:, :1], rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pleiotropy_chunks_and_call_state_are_independent(
    pleiotropy_data, monkeypatch, dtype
):
    mat, adj = (array.astype(dtype) for array in pleiotropy_data)
    mat = np.vstack([mat, mat[::-1], mat[:1]])
    changed = adj.copy()
    changed[:8, 0] = 0
    changed[:, 0] *= -0.5
    changed[:, 1] *= 0.5
    changed[32:48, 2] = -0.375
    # Keep the CPU parity fixture away from an exactly cancelling overlap.
    changed[8, 0] *= 1.001
    cases = [(adj, 10), (changed, 10), (changed, 16), (adj, 10)]
    reference = [
        dc_cpu.mt._viper._func_viper(mat, network, n_targets=n_targets)
        for network, n_targets in cases
    ]
    assert np.max(np.abs(reference[0][0] - reference[1][0])) > 0.1
    assert np.max(np.abs(reference[1][0] - reference[2][0])) > 0.1
    gpu_mat, gpu_adj = cp.asarray(mat), cp.asarray(adj)
    baseline = []
    for budget in [1, 24 * 1024, 1024**3]:
        monkeypatch.setattr(dc._method_viper, "_PLEIOTROPY_BYTES", budget)
        for i, (network, n_targets) in enumerate(cases):
            # Reuse the same allocation while changing weights and topology.
            cp.copyto(gpu_adj, cp.asarray(network))
            result = _func_viper(gpu_mat, gpu_adj, n_targets=n_targets)
            for actual, expected in zip(result, reference[i], strict=True):
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
            if budget == 1:
                baseline.append(result)
            else:
                for actual, expected in zip(result, baseline[i], strict=True):
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("factor", [1e308, np.inf, np.nan])
def test_ordered_penalties_preserve_underflow_and_nan(factor):
    mat = np.arange(5, dtype=np.float64)[None, :]
    adj = np.vstack([np.ones(4), np.eye(4)])
    initial = dc_cpu.mt._viper._aREA(mat, adj)
    quantiles = rankdata(mat, axis=1) / 6
    magnitude = np.abs(quantiles - 0.5) * 2
    magnitude += (1 - magnitude.max()) / 2
    # Source 0 loses its shared target before it shadows source 3. A zero
    # likelihood prevents that last penalty; a NaN likelihood does not.
    likelihood = adj.copy()
    likelihood[0, 0] = np.nan if np.isnan(factor) else 0
    likelihood[0, 3] = 0.5 if np.isnan(factor) else 1
    expected = dc_cpu.mt._viper._aREA(mat, adj, wts=likelihood)
    actual = _correct(
        cp.asarray(norm.ppf(quantiles)),
        cp.asarray(norm.ppf(magnitude)),
        cp.asarray(adj),
        cp.asarray(initial),
        rows=cp.asarray([0, 0, 0]),
        losers=cp.asarray([0, 0, 3]),
        winners=cp.asarray([1, 2, 0]),
        factors=cp.asarray([factor, factor, 2]),
    )
    np.testing.assert_allclose(actual.get(), expected, rtol=1e-12, atol=1e-12)


def test_multiple_overlapping_regulators_across_observations():
    rng = np.random.default_rng(135)
    mat = rng.normal(size=(7, 200))
    mat[:, :60] += np.array([4, -4, 4, -4, 0, 2, -2])[:, None]
    adj = np.zeros((200, 6))
    adj[:80] = rng.uniform(0.5, 1.5, size=(80, 6)) * (rng.random((80, 6)) < 0.7)
    adj[:, ::2] *= -1
    initial, _ = _run(mat, adj, pleiotropy=False)
    refined, _ = _assert_parity(mat, adj)
    assert np.max(np.abs(refined - initial)) > 0.1


def test_symmetric_overlap_uses_zero_signed_contribution():
    mat = np.arange(300, dtype=np.float32)[None, :]
    adj = np.zeros((300, 2), dtype=np.float32)
    adj[-24:, 0] = -0.5
    adj[np.r_[276:282, 294:300], 1] = 0.5
    initial, initial_p = _run(mat, adj, pleiotropy=False)
    assert np.all(initial_p < 0.05)
    # Both overlaps have exactly opposing signed ranks. Their one-sided
    # probabilities depend only on the unsigned contribution, with sign +1.
    shared = np.r_[:6, 18:24]
    q_left = np.arange(1, 25) / 25
    q_right = np.arange(1, 13) / 13
    unsigned_left = norm.ppf(abs(q_left - 0.5) * 2 + 1 / 25)[shared]
    unsigned_right = norm.ppf(abs(q_right - 0.5) * 2 + 1 / 13)
    p_left = norm.sf(0.5 * unsigned_left.sum() / np.sqrt(12))
    p_right = norm.sf(0.5 * unsigned_right.sum() / np.sqrt(12))
    gpu_mat, gpu_adj = cp.asarray(mat), cp.asarray(adj)
    _, _, overlap_p = dc._method_viper._get_inter_pvals(
        cp.asarray(initial),
        gpu_mat,
        gpu_adj,
        cp.asarray(initial_p < 0.05),
        10,
        sparse_net=dc._method_viper._sparse_network(gpu_adj),
        cache={},
    )
    np.testing.assert_allclose(
        overlap_p.get(), [[np.nan, p_left], [p_right, np.nan]], rtol=1e-12
    )
    # Source 0 loses the overlap; the remaining signed rescore is unambiguous.
    assert p_left < p_right
    likelihood = (adj != 0).astype(float)
    likelihood[276 + shared, 0] /= (1 + np.log10(p_right / p_left)) ** 20
    expected = dc_cpu.mt._viper._aREA(mat.astype(float), adj, wts=likelihood)
    actual, pval = _run(mat, adj)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pval, 2 * norm.sf(abs(expected)), rtol=1e-12)
    assert abs(actual[0, 0] - initial[0, 0]) > 1
    assert actual[0, 1] == initial[0, 1]


def test_pleiotropy_with_unequal_regulon_sizes():
    mat = np.arange(512, dtype=np.float32)[None, :]
    adj = np.zeros((512, 2), dtype=np.float32)
    adj[-192:, 0] = 0.8
    adj[-16:, 1] = 0.5
    initial, _ = _run(mat, adj, pleiotropy=False)
    corrected, _ = _assert_parity(mat, adj)
    assert np.max(np.abs(corrected - initial)) > 0.1


@pytest.mark.parametrize(("nvar", "ntail"), [(24, 6), (25430, 1)])
def test_symmetric_initial_cancellation_is_zero(nvar, ntail):
    mat = np.arange(nvar, dtype=np.float64)[None, :]
    adj = np.zeros((nvar, 1))
    adj[np.r_[:ntail, nvar - ntail : nvar], 0] = 0.5
    score, pval = _run(mat, adj, pleiotropy=False)
    np.testing.assert_array_equal(score, 0)
    np.testing.assert_array_equal(pval, 1)


@pytest.mark.parametrize(("nvar", "ntail"), [(24, 6), (25430, 1)])
@pytest.mark.parametrize("perturbation", [-1e-6, 1e-6])
def test_resolvable_initial_cancellation_keeps_direction(nvar, ntail, perturbation):
    mat = np.arange(nvar, dtype=np.float64)[None, :]
    adj = np.zeros((nvar, 1))
    adj[np.r_[:ntail, nvar - ntail : nvar], 0] = 0.5
    adj[-1, 0] += perturbation
    score, pval = _assert_parity(mat, adj, pleiotropy=False)
    np.testing.assert_array_equal(np.sign(score), np.sign(perturbation))
    assert np.all(pval < 1)


@pytest.mark.parametrize("perturbation", [-1e-6, 0, 1e-6])
def test_corrected_cancellation_keeps_resolvable_direction(perturbation, monkeypatch):
    # Equal penalties preserve symmetric terms. Positive initial scores force
    # rescoring even though the exactly symmetric network has no direction.
    signed = np.array([[-3, -2, -1, 1, 2, 3 + perturbation]], dtype=float)
    original_get = cp.ndarray.get

    def reject_host_transfer(array, *args, **kwargs):
        assert array.ndim == 0, "Corrected scoring arrays copied to CPU"
        return original_get(array, *args, **kwargs)

    monkeypatch.setattr(cp.ndarray, "get", reject_host_transfer)
    result = _correct(
        cp.asarray(signed),
        cp.ones_like(cp.asarray(signed)),
        cp.full((6, 2), 0.5),
        cp.asarray([[7.0, 9.0]]),
        rows=cp.asarray([0]),
        losers=cp.asarray([0]),
        winners=cp.asarray([1]),
        factors=cp.asarray([2.0]),
    )
    actual = original_get(result)
    # Six equally likely targets, each with weight 1/2, leave a signed
    # contribution of perturbation/12 and an unsigned contribution of 1/2.
    expected = (abs(perturbation) / 12 + 0.5) * np.sign(perturbation) * np.sqrt(6)
    np.testing.assert_allclose(actual, [[expected, 9.0]], rtol=1e-12, atol=1e-12)
    if perturbation == 0:
        assert actual[0, 0] == 0


@pytest.mark.parametrize(
    ("nvar", "ntail", "nsrc", "kwargs"),
    [
        (100, 12, 2, {}),
        (100, 12, 3, {}),
        (12, 2, 2, {"reg_sign": 0.99, "n_targets": 0, "tmin": 0}),
    ],
)
def test_public_symmetric_cancellation_is_zero(nvar, ntail, nsrc, kwargs):
    # Undo the public extractor's feature permutation so both target tails
    # have exactly opposing ranks. CPU BLAS roundoff is not a direction oracle.
    permutation = np.random.default_rng(0).choice(nvar, nvar, replace=False)
    mat = np.empty((1, nvar))
    mat[:, permutation] = np.arange(nvar)
    frame = pd.DataFrame(mat, index=["cell"], columns=[f"g{i}" for i in range(nvar)])
    targets = frame.columns[permutation[np.r_[:ntail, nvar - ntail : nvar]]]
    net = pd.DataFrame(
        {
            "source": np.repeat(["A", "B", "C"][:nsrc], 2 * ntail),
            "target": np.tile(targets, nsrc),
            "weight": 0.5,
        }
    )
    actual, actual_p = dc.viper(frame, net, empty=False, **kwargs)
    np.testing.assert_array_equal(actual, 0)
    np.testing.assert_array_equal(actual_p, 1)
    pd.testing.assert_index_equal(actual.index, frame.index)
    pd.testing.assert_index_equal(actual_p.columns, actual.columns)
    assert actual.columns.tolist() == ["A", "B", "C"][:nsrc]


@pytest.mark.parametrize("pleiotropy", [False, True])
def test_scoring_stays_on_gpu(pleiotropy_data, monkeypatch, pleiotropy):
    import rapids_singlecell.decoupler_gpu._method_viper as module

    if pleiotropy:
        mat, adj = pleiotropy_data
    else:
        mat = np.arange(24, dtype=np.float64)[None, :]
        adj = np.zeros((24, 1))
        adj[np.r_[:6, 18:24], 0] = 0.5
    output_shape = (mat.shape[0], adj.shape[1])
    transfers = []
    original_get = cp.ndarray.get

    def get_output_or_metadata(array, *args, **kwargs):
        if array.ndim == 2:
            assert array.shape == output_shape, "Scoring matrix copied to CPU"
            transfers.append(array.shape)
        return original_get(array, *args, **kwargs)

    def reject_cpu_quantiles(*args, **kwargs):
        pytest.fail("Normal quantiles must be computed on GPU")

    monkeypatch.setattr(cp.ndarray, "get", get_output_or_metadata)
    monkeypatch.setattr(scipy.special, "ndtri", reject_cpu_quantiles)
    # Also reject the former import alias if a fallback is reintroduced.
    monkeypatch.setattr(module, "cpu_ndtri", reject_cpu_quantiles, raising=False)
    score, pval = _run(mat, adj, pleiotropy=pleiotropy)
    assert transfers == [output_shape, output_shape]
    assert np.all(np.isfinite(score)) and np.all(np.isfinite(pval))


def _frame_and_net(mat, adj):
    frame = pd.DataFrame(
        mat,
        index=[f"cell_{i}" for i in range(len(mat))],
        columns=[f"gene_{i}" for i in range(mat.shape[1])],
    )
    targets, sources = np.nonzero(adj)
    names = np.array(["regulator_z", "regulator_a", "regulator_m"])
    net = pd.DataFrame(
        {
            "source": names[sources],
            "target": frame.columns[targets],
            "weight": adj[targets, sources],
        }
    )
    return frame, net


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_noncontiguous_gpu_inputs_match_decoupler(pleiotropy_data, dtype):
    mat, adj = (array.astype(dtype) for array in pleiotropy_data)
    gpu_mat = cp.asarray(np.repeat(mat, 2, axis=1))[:, ::2]
    gpu_adj = cp.asarray(np.repeat(adj, 2, axis=0))[::2]
    assert not gpu_mat.flags.c_contiguous
    assert not gpu_adj.flags.c_contiguous
    expected, expected_p = dc_cpu.mt._viper._func_viper(mat, adj)
    actual, pval = _func_viper(gpu_mat, gpu_adj)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(pval, expected_p, rtol=2e-6, atol=2e-7)

    frame, net = _frame_and_net(mat, adj)
    expected, expected_p = dc_cpu.mt.viper(frame, net, tmin=0, empty=False)
    adata = AnnData(gpu_mat)
    adata.obs_names, adata.var_names = frame.index, frame.columns
    dc.viper(adata, net, tmin=0, empty=False)
    actual, pval = adata.obsm["score_viper"], adata.obsm["padj_viper"]
    pd.testing.assert_frame_equal(actual, expected, rtol=2e-6, atol=2e-7)
    pd.testing.assert_frame_equal(pval, expected_p, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_viper_uses_current_nonblocking_stream(pleiotropy_data, dtype):
    mat, adj = (array.astype(dtype) for array in pleiotropy_data)
    expected, expected_p = dc_cpu.mt._viper._func_viper(-mat, adj)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        gpu_mat, gpu_adj = cp.asarray(mat), cp.asarray(adj)
        gpu_mat *= -1
        actual, pval = _func_viper(gpu_mat, gpu_adj)
    stream.synchronize()
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(pval, expected_p, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize(
    "kind", ["dataframe", "sparse", "cupy_sparse", "backed", "dask"]
)
def test_public_routing_batching_and_adjustment(pleiotropy_data, kind, tmp_path):
    mat, adj = pleiotropy_data
    frame, net = _frame_and_net(mat, adj)
    matrix, obs, var = dc_cpu.pp.extract(frame, empty=False)
    sources, _, ordered_adj = dc_cpu.pp.adjmat(features=var, net=net)
    expected, raw_p = dc_cpu.mt._viper._func_viper(matrix, ordered_adj)
    expected_p = false_discovery_control(raw_p, axis=1)
    assert np.any(expected_p > raw_p)
    if kind == "dataframe":
        data = frame
    else:
        data = AnnData(sps.csr_matrix(mat))
        data.obs_names = frame.index
        data.var_names = frame.columns
        if kind == "cupy_sparse":
            data.X = csps.csr_matrix(cp.asarray(mat))
        elif kind == "backed":
            path = tmp_path / "viper.h5ad"
            data.write_h5ad(path)
            data = read_h5ad(path, backed="r")
        elif kind == "dask":
            import dask.array as da

            # Feature rechunking and an incomplete final observation block.
            data.X = da.from_array(mat, chunks=(2, 37))
    try:
        for bsize in [1, 2] if kind in {"sparse", "cupy_sparse", "backed"} else [2]:
            result = dc.viper(data, net, tmin=0, empty=False, bsize=bsize)
            if isinstance(data, AnnData):
                assert result is None
                score, padj = data.obsm["score_viper"], data.obsm["padj_viper"]
            else:
                score, padj = result
            np.testing.assert_array_equal(score.index, obs)
            np.testing.assert_array_equal(score.columns, sources)
            pd.testing.assert_index_equal(padj.index, score.index)
            pd.testing.assert_index_equal(padj.columns, score.columns)
            np.testing.assert_allclose(score, expected, rtol=2e-6, atol=2e-7)
            np.testing.assert_allclose(padj, expected_p, rtol=2e-6, atol=2e-7)
    finally:
        if isinstance(data, AnnData) and data.isbacked:
            data.file.close()


@pytest.mark.parametrize("bsize", [1, 2])
def test_tied_extrema_match_upstream_with_same_sparse_batches(bsize):
    # Upstream aREA offsets absolute ranks using the maximum across a batch.
    mat = np.array([[0, 0, 0, 1, 1, 1], [0, 1, 2, 3, 4, 5]], dtype=np.float32)
    adj = np.array([[0.2, 0], [0, 1], [0, 0], [0, 0], [0.4, 0], [0.5, 0]])
    frame, net = _frame_and_net(mat, adj)
    cpu = AnnData(sps.csr_matrix(mat))
    cpu.obs_names, cpu.var_names = frame.index, frame.columns
    gpu = cpu.copy()
    dc_cpu.mt.viper(cpu, net, tmin=0, empty=False, bsize=bsize, pleiotropy=False)
    dc.viper(gpu, net, tmin=0, empty=False, bsize=bsize, pleiotropy=False)
    for key in ["score_viper", "padj_viper"]:
        pd.testing.assert_frame_equal(
            gpu.obsm[key], cpu.obsm[key], check_dtype=False, rtol=2e-6, atol=2e-7
        )


def test_metadata():
    assert dc.viper.adj and dc.viper.weight and dc.viper.test
    meta = dc.viper.meta().iloc[0]
    assert meta["name"] == "viper"
    assert meta["stype"] == "numerical"
    assert meta["limits"] == (-np.inf, np.inf)
    assert meta["reference"] == "https://doi.org/10.1038/ng.3593"
