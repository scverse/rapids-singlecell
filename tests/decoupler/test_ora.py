from __future__ import annotations

import inspect
from importlib.metadata import version

import anndata as ad
import cupy as cp
import decoupler as dc
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
from decoupler.mt._ora import _test1t
from packaging.version import Version
from scipy.stats import rankdata

import rapids_singlecell.decoupler_gpu as rdc
from rapids_singlecell.decoupler_gpu._method_ora import (
    _func_ora,
    _log_odds,
    _ora_tables,
)

cpu_220 = pytest.mark.skipif(
    Version(version("decoupler")) >= Version("2.2.1"),
    reason="Compatibility target is released decoupler 2.2.0, before the ORA rank fix",
)


@pytest.fixture
def compatibility_frame():
    return pd.DataFrame(
        [
            [-3, 2, 1, -2, 3, 4, -4, 0.5, 1.5, 2.5, -0.5, 0],
            [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 0],
            [0] * 12,
            1
            + np.array([11, 0, 10, 1, 9, 2, 8, 3, 7, 4, 6, 5])
            * np.finfo(np.float32).eps,
        ],
        index=["sample_z", "sample_a", "sample_empty", "sample_tie"],
        columns=[f"g{i:02}" for i in range(12)],
        dtype=np.float64,
    )


@pytest.fixture
def compatibility_net():
    return pd.DataFrame(
        {
            "source": ["z_set"] * 5 + ["a_set"] * 4 + ["other"] * 4,
            "target": [f"g{i:02}" for i in [0, 2, 4, 6, 8, 1, 2, 3, 5, 8, 9, 10, 11]],
        }
    )


def _assert_cpu_equal(actual, expected):
    for observed, reference in zip(actual, expected, strict=True):
        pd.testing.assert_frame_equal(observed, reference, rtol=1e-7, atol=1e-12)


def _anndata_results(data, returned):
    result = data if returned is None else returned
    return result.obsm["score_ora"], result.obsm["padj_ora"]


@pytest.mark.parametrize("xp", [np, cp])
def test_ora_integer_correction_precision(xp):
    a = xp.asarray([0], dtype=xp.int32)
    b = c = xp.asarray([50_000], dtype=xp.int32)
    actual = _log_odds(a, b, c, xp.asarray([10_000.0]), 1)
    np.testing.assert_allclose(cp.asnumpy(actual), np.log(10_001 / 50_001**2))


@cpu_220
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"n_up": 3},
        {"n_up": 3, "n_bm": 2},
        {"n_up": 3.2, "n_bm": 2.2},
        {"n_up": 1, "n_bm": 5},
        {"n_up": 0.2},
        {"n_up": 100, "n_bm": 2},
        {"n_up": 100, "n_bm": 100},
    ],
)
@pytest.mark.parametrize("background", [None, 0, 20_000])
def test_ora_identical_cpu_calls(
    compatibility_frame, compatibility_net, kwargs, background
):
    original_frame = compatibility_frame.copy(deep=True)
    original_net = compatibility_net.copy(deep=True)
    expected = dc.mt.ora(
        compatibility_frame, compatibility_net, n_bg=background, tmin=2, **kwargs
    )
    actual = rdc.ora(
        compatibility_frame, compatibility_net, n_bg=background, tmin=2, **kwargs
    )
    _assert_cpu_equal(actual, expected)
    pd.testing.assert_frame_equal(compatibility_frame, original_frame)
    pd.testing.assert_frame_equal(compatibility_net, original_net)


@cpu_220
def test_ora_all_defaults_match_cpu(compatibility_frame, compatibility_net):
    # The five-target z_set survives upstream's default tmin=5.
    _assert_cpu_equal(
        rdc.ora(compatibility_frame, compatibility_net),
        dc.mt.ora(compatibility_frame, compatibility_net),
    )


@cpu_220
@pytest.mark.parametrize("pre_load", [False, True])
@pytest.mark.parametrize("kind", ["list", "dense", "sparse", "gpu", "dask", "backed"])
def test_ora_input_formats_match_cpu(
    compatibility_frame, compatibility_net, tmp_path, kind, pre_load
):
    kwargs = {
        "tmin": 2,
        "n_up": 10,
        "n_bg": None,
        "empty": False,
        "bsize": 2,
    }
    # Matrix data is float32; nearby float64 values become tied before ranking.
    compatibility_frame = compatibility_frame.copy()
    compatibility_frame.iloc[-1] = (
        1 + np.array([11, 0, 10, 1, 9, 2, 8, 3, 7, 4, 6, 5]) * np.finfo(np.float64).eps
    )
    rounded = compatibility_frame.astype(np.float32)
    expected = dc.mt.ora(rounded, compatibility_net, **kwargs)
    if kind == "list":
        data = [
            compatibility_frame.values,
            compatibility_frame.index.values,
            compatibility_frame.columns.values,
        ]
        _assert_cpu_equal(
            rdc.ora(data, compatibility_net, pre_load=pre_load, **kwargs),
            expected,
        )
        return
    gpu_data = ad.AnnData(compatibility_frame.copy())
    if kind == "sparse":
        gpu_data.X = sps.csr_matrix(gpu_data.X)
    elif kind == "gpu":
        gpu_data.X = cp.asarray(gpu_data.X)
    elif kind == "dask":
        import dask.array as da

        gpu_data.X = da.from_array(gpu_data.X, chunks=(2, 4))
    elif kind == "backed":
        gpu_data.write_h5ad(tmp_path / "gpu.h5ad")
        gpu_data = ad.read_h5ad(tmp_path / "gpu.h5ad", backed="r")
        # Backed extraction orders tied features differently from DataFrames.
        ad.AnnData(rounded).write_h5ad(tmp_path / "cpu.h5ad")
        cpu_data = ad.read_h5ad(tmp_path / "cpu.h5ad", backed="r")
        try:
            returned = dc.mt.ora(cpu_data, compatibility_net, **kwargs)
            expected = _anndata_results(cpu_data, returned)
        finally:
            cpu_data.file.close()
    try:
        actual = rdc.ora(gpu_data, compatibility_net, pre_load=pre_load, **kwargs)
        assert actual is None
        _assert_cpu_equal(_anndata_results(gpu_data, actual), expected)
    finally:
        if kind == "backed":
            gpu_data.file.close()


@cpu_220
@pytest.mark.parametrize(
    "raw,layer", [(False, None), (True, None), (False, "other"), (True, "other")]
)
def test_ora_raw_layer_match_cpu(compatibility_frame, compatibility_net, raw, layer):
    cpu_data = ad.AnnData(compatibility_frame.copy())
    cpu_data.layers["other"] = cpu_data.X[:, ::-1].copy()
    cpu_data.raw = cpu_data.copy()
    cpu_data = cpu_data[:, :8].copy()
    gpu_data = cpu_data.copy()
    kwargs = {"layer": layer, "n_up": 3, "n_bg": None}
    expected = dc.mt.ora(cpu_data, compatibility_net, 2, raw, False, 2, False, **kwargs)
    actual = rdc.ora(gpu_data, compatibility_net, 2, raw, False, 2, False, **kwargs)
    assert (actual is None) == (expected is None)
    _assert_cpu_equal(
        _anndata_results(gpu_data, actual), _anndata_results(cpu_data, expected)
    )


@cpu_220
def test_ora_and_query_set_public_parameter_contract():
    for implementation, reference in [
        (rdc.ora, dc.mt.ora),
        (rdc.query_set, dc.mt.query_set),
    ]:
        actual = inspect.signature(implementation).parameters
        expected = inspect.signature(reference).parameters
        for name, parameter in expected.items():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            assert name in actual
            assert actual[name].kind == parameter.kind
            assert actual[name].default == parameter.default


@cpu_220
@pytest.mark.parametrize(
    "background,correction", [(20_000.5, 0.5), (8.5, 0.5), (6.5, 0.5), (8, -0.1)]
)
def test_ora_numeric_options(background, correction):
    # Every contingency-table cell stays positive after this correction.
    data = pd.DataFrame([np.arange(1, 9)], columns=list("abcdefgh"))
    net = pd.DataFrame(
        {"source": ["four"] * 4 + ["three"] * 3, "target": list("abefabe")}
    )
    kwargs = {"n_up": 4, "n_bg": background, "ha_corr": correction, "tmin": 2}
    _assert_cpu_equal(rdc.ora(data, net, **kwargs), dc.mt.ora(data, net, **kwargs))


@cpu_220
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("background", [200, 20_000])
def test_ora_warp_boundaries(monkeypatch, fallback, background):
    if fallback:
        import rapids_singlecell.decoupler_gpu._method_ora as module

        monkeypatch.setattr(module, "_ora_tables", lambda *args, **kwargs: None)
    # Exercise partial warps, multiple warp iterations, and a partial final block.
    rng = np.random.default_rng(42)
    nobs, nvar, count = 80, 200, 70
    sizes = np.array([1, 31, 32, 33, 63, 64, 65], dtype=np.int32)
    starts = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int32)
    cnct = np.concatenate(
        [rng.choice(nvar, size, replace=False) for size in sizes]
    ).astype(np.int32)
    x = rng.normal(size=(nobs, nvar)).astype(np.float32)
    es, pv = _func_ora(
        cp.asarray(x),
        cnct=cp.asarray(cnct),
        starts=cp.asarray(starts),
        offsets=cp.asarray(sizes),
        n_up=nvar - count,
        n_bg=background,
    )
    for i, row in enumerate(x):
        selected = set(np.argsort(row)[-count:])
        for j, (start, size) in enumerate(zip(starts, sizes, strict=True)):
            overlap = len(selected & set(cnct[start : start + size]))
            b, c = int(size) - overlap, count - overlap
            d = background - int(size) - c
            expected = _test1t(overlap, b, c, d)
            np.testing.assert_allclose(pv[i, j], expected, rtol=1e-7)
            np.testing.assert_allclose(
                es[i, j], np.log((overlap + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5)))
            )


@cpu_220
@pytest.mark.parametrize("background", range(2, 13))
def test_ora_compat_lookup_exhaustive_small_tables(background):
    sizes = cp.arange(1, background + 1, dtype=cp.int32)
    for count in range(background + 1):
        _, pvals, groups = _ora_tables(sizes, count, background, 0.5)
        pvals = pvals.get()
        groups = groups.get()
        actual, expected = [], []
        for size, group in zip(range(1, background + 1), groups, strict=True):
            for a in range(max(0, size + count - background), min(size, count) + 1):
                actual.append(pvals[group, a])
                expected.append(
                    _test1t(a, size - a, count - a, background - size - count + a)
                )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"n_up": 0}, AssertionError, "n_up"),
        ({"n_bm": -1}, AssertionError, "n_bm"),
        ({"n_up": np.inf}, AssertionError, "n_up"),
        ({"ha_corr": np.nan}, AssertionError, "ha_corr"),
        ({"n_bg": "auto"}, AssertionError, "n_bg"),
        ({"n_bg": 1}, ValueError, "union"),
        ({"n_up": 10, "n_bg": 6.9999999, "empty": False}, ValueError, "union"),
        ({"n_bg": 2**31}, ValueError, "int32"),
        ({"n_up": 100}, ValueError, "no features"),
    ],
)
def test_ora_invalid_parameters(
    compatibility_frame, compatibility_net, kwargs, error, match
):
    with pytest.raises(error, match=match):
        rdc.ora(compatibility_frame, compatibility_net, **kwargs)


@pytest.mark.parametrize("nvar", [1, 31, 32, 33, 257, 20_000])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ora_selection_ties(nvar, dtype):
    from rapids_singlecell.decoupler_gpu._method_ora import _select_ora

    rng = np.random.default_rng(17)
    x = np.round(rng.normal(size=(5, nvar)), 1).astype(dtype)
    x[0] = 0
    x[0, ::2] = -0.0
    x[1] = np.arange(nvar)
    x[2] = x[1, ::-1]
    x[3] = 1 + rng.permutation(nvar) * np.finfo(np.float64).eps
    up = max(1, int(np.ceil(nvar * 0.05)))
    bm = min(3, nvar - up)
    ranks = rankdata(x.astype(np.float32), method="ordinal", axis=1)
    expected = (ranks > nvar - up) | (ranks <= bm)
    actual = _select_ora(cp.asarray(x), up, bm).get().T
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "background,nvar,count,targets,pinned",
    [
        (7, 7, 2, [0, 5], 11 / 21),
        (12, 12, 6, [6, 7, 8], 1 / 11),
        (1.5, 2, 1, [1], 2 / 3),
    ],
)
@pytest.mark.parametrize("fallback", [False, True])
def test_ora_cpu_fisher_edge_cases(
    monkeypatch, *, background, nvar, count, targets, pinned, fallback
):
    if fallback:
        import rapids_singlecell.decoupler_gpu._method_ora as module

        monkeypatch.setattr(module, "_ora_tables", lambda *args, **kwargs: None)
    scores, pvals = _func_ora(
        cp.tile(cp.arange(1, nvar + 1, dtype=cp.float32), (4, 1)),
        cnct=cp.asarray(targets, dtype=cp.int32),
        starts=cp.asarray([0], dtype=cp.int32),
        offsets=cp.asarray([len(targets)], dtype=cp.int32),
        n_up=nvar - count,
        n_bg=background,
    )
    a = sum(target >= nvar - count for target in targets)
    b, c = len(targets) - a, count - a
    d = background - a - b - c
    # Pin CPU 2.2.0 probability ties and fractional support independently.
    np.testing.assert_allclose(pvals, pinned, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        scores,
        np.log((a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5))),
        rtol=1e-12,
        atol=1e-12,
    )
