from __future__ import annotations

import cupy as cp
import decoupler as dc
import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control, fisher_exact

import rapids_singlecell as rsc
from rapids_singlecell.decoupler_gpu._method_ora import _fisher


@pytest.fixture
def query_net():
    return pd.DataFrame(
        {
            "source": ["enriched"] * 4 + ["other"] * 4 + ["small"] * 2,
            "target": ["a", "b", "c", "d", "d", "e", "f", "g", "h", "i"],
        }
    )


@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
@pytest.mark.parametrize("n_bg", [None, 0, 20_000, 20_000.5])
@pytest.mark.parametrize(
    "features,correction",
    [
        (["a", "b", "c", "outside", "a"], 0.5),
        ([], 0.5),
        (["a"], 1),
        (["a", "e", "outside"], -0.5),
    ],
)
def test_query_set_decoupler(query_net, alternative, n_bg, features, correction):
    original = query_net.copy(deep=True)
    args = (features, query_net, alternative, n_bg, correction, 3, False)
    expected = dc.mt.query_set(*args)
    actual = rsc.dcg.query_set(*args)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-7, atol=1e-12)
    assert set(actual["source"]) == {"enriched", "other"}
    pd.testing.assert_frame_equal(query_net, original)


@pytest.mark.parametrize("factory", [list, set, tuple, np.asarray, pd.Index, iter])
def test_query_set_iterables(query_net, factory):
    actual = rsc.dcg.query_set(factory(["a", "b", "a"]), query_net, tmin=3)
    expected = rsc.dcg.query_set(["a", "b"], query_net, tmin=3)
    pd.testing.assert_frame_equal(actual, expected)


def test_query_set_zero_correction_matches_decoupler_error(query_net):
    for func in (dc.mt.query_set, rsc.dcg.query_set):
        with pytest.raises(ZeroDivisionError):
            func(["a"], query_net, ha_corr=0, tmin=3)


@pytest.mark.parametrize("factory", [list, set, pd.Index, iter])
@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
def test_query_set_named_background(query_net, factory, alternative):
    # Unannotated selected and unselected genes still contribute to the table.
    background = ["a", "b", "c", "d", "e", "f", "outside", "untargeted", "a"]
    result = rsc.dcg.query_set(
        ["a", "b", "outside"],
        query_net,
        background=factory(background),
        tmin=3,
        alternative=alternative,
    ).set_index("source")
    tables = {"enriched": [[2, 1], [2, 3]], "other": [[0, 3], [3, 2]]}
    expected_pvals = []
    for source in result.index:
        (a, b), (c, d) = tables[source]
        expected_pvals.append(
            fisher_exact(tables[source], alternative=alternative).pvalue
        )
        np.testing.assert_allclose(
            result.loc[source, "stat"],
            np.log((a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5))),
        )
    np.testing.assert_allclose(result["pval"], expected_pvals, rtol=1e-7)
    np.testing.assert_allclose(result["padj"], false_discovery_control(expected_pvals))
    # 'other' has four annotation targets but only three tested targets.
    result = rsc.dcg.query_set(["a"], query_net, background=background, tmin=4)
    assert result["source"].tolist() == ["enriched"]


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"features": "TP53"}, AssertionError, "iterable"),
        ({"features": b"TP53"}, AssertionError, "iterable"),
        ({"features": 123}, AssertionError, "iterable"),
        ({"features": None}, AssertionError, "iterable"),
        ({"background": []}, ValueError, "background"),
        ({"background": "abc"}, ValueError, "background"),
        ({"background": ["a", None]}, ValueError, "background"),
        ({"background": ["a", np.nan]}, ValueError, "background"),
        ({"background": list("abcdefg")}, ValueError, "selected.*background"),
        ({"alternative": "invalid"}, AssertionError, "alternative"),
        ({"n_bg": -1}, AssertionError, "n_bg"),
        ({"n_bg": np.inf}, AssertionError, "n_bg"),
        ({"ha_corr": np.nan}, AssertionError, "ha_corr"),
        ({"n_bg": 2}, ValueError, "union"),
        ({"n_bg": 2**31}, ValueError, "int32"),
    ],
)
def test_query_set_invalid_input(query_net, kwargs, error, match):
    with pytest.raises(error, match=match):
        rsc.dcg.query_set(
            net=query_net, tmin=3, **{"features": ["a", "outside"], **kwargs}
        )


@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
@pytest.mark.parametrize(
    "background,count,size", [(10, 4, 5), (20, 20, 5), (20_000, 100, 150)]
)
def test_query_fisher_tails(alternative, background, count, size):
    a = np.arange(max(0, count + size - background), min(count, size) + 1)
    actual = _fisher(
        cp.asarray(a), count, size, background, alternative=alternative
    ).get()
    expected = [
        fisher_exact(
            [
                [int(k), size - int(k)],
                [count - int(k), background - size - count + int(k)],
            ],
            alternative=alternative,
        ).pvalue
        for k in a
    ]
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=0)
