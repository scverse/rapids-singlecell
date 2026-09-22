from __future__ import annotations

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
from scipy.stats import false_discovery_control, fisher_exact

import rapids_singlecell.decoupler_gpu as dc

SELECTED = {"none": "", "small": "ab", "large": "acdfg", "dense": "abcdfg"}
TERMS = {"term_a": set("abc"), "term_b": set("cde"), "term_zero": set("eh")}


@pytest.fixture
def deg_data():
    # e and h are always zero; f and g are tested but have no annotation.
    labels = pd.DataFrame(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 1, 1, 0],
            [1, 1, 1, 1, 0, 1, 1, 0],
        ],
        index=list(SELECTED),
        columns=list("abcdefgh"),
        dtype=bool,
    )
    net = pd.DataFrame(
        {
            "source": ["term_a"] * 4 + ["term_b"] * 3 + ["term_zero"] * 2,
            "target": ["a", "b", "c", "outside", "c", "d", "e", "e", "h"],
        }
    )
    return labels, net


def _assert_fisher(result, selected, *, background=8, alternative="two-sided"):
    scores, padj = result
    expected_scores = np.empty((len(selected), len(scores.columns)))
    pvals = np.empty_like(expected_scores)
    for i, genes in enumerate(selected.values()):
        genes = set(genes)
        for j, name in enumerate(scores.columns):
            term = TERMS[name]
            a, b, c = len(genes & term), len(genes - term), len(term - genes)
            d = background - a - b - c
            pvals[i, j] = fisher_exact([[a, b], [c, d]], alternative=alternative).pvalue
            expected_scores[i, j] = np.log(
                (a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5))
            )
    for actual, values, tolerance in (
        (scores, expected_scores, 1e-10),
        (padj, false_discovery_control(pvals, axis=1), 1e-7),
    ):
        expected = pd.DataFrame(values, index=list(selected), columns=scores.columns)
        pd.testing.assert_frame_equal(actual, expected, rtol=tolerance, atol=1e-12)


@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
def test_value_cutoff_fisher(deg_data, alternative):
    labels, net = deg_data
    original = labels.copy(deep=True)
    result = dc.ora(
        labels,
        net,
        n_bg=None,
        value_cutoff=0.5,
        alternative=alternative,
        tmin=2,
        bsize=2,
    )
    # Empty rows/columns and unannotated genes must remain in the universe.
    assert result[0].columns.tolist() == list(TERMS)
    _assert_fisher(result, SELECTED, alternative=alternative)
    np.testing.assert_array_equal(result[1].loc["none"], 1)
    pd.testing.assert_frame_equal(labels, original)


@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_ranked_one_sided(deg_data, alternative):
    _, net = deg_data
    data = pd.DataFrame([np.arange(1, 9)], index=["up"], columns=list("abcdefgh"))
    result = dc.ora(data, net, n_up=2, n_bg=8, alternative=alternative, tmin=2)
    _assert_fisher(result, {"up": "cdefgh"}, alternative=alternative)


@pytest.mark.parametrize(
    "kwargs,background,columns",
    [
        ({}, 20_000, list(TERMS)),
        ({"n_bg": 0, "tmin": 3}, 8, ["term_a", "term_b"]),
    ],
)
def test_value_cutoff_background_and_pruning(deg_data, kwargs, background, columns):
    labels, net = deg_data
    options = {"tmin": 2, **kwargs}
    result = dc.ora(labels, net, value_cutoff=0.5, **options)
    assert result[0].columns.tolist() == columns
    _assert_fisher(result, SELECTED, background=background)


@pytest.mark.parametrize("direction", ["greater", "less"])
@pytest.mark.parametrize("pre_load", [False, True])
def test_value_cutoff_float32_threshold(deg_data, direction, pre_load):
    _, net = deg_data
    cutoff = np.float32(0.05)
    lower = np.nextafter(cutoff, -np.inf, dtype=np.float32)
    upper = np.nextafter(cutoff, np.inf, dtype=np.float32)
    # Distinct float32 neighbors straddle the strict threshold; equality
    # must remain excluded after matrix conversion and optional preloading.
    assert lower < cutoff < upper
    values = pd.DataFrame(
        [[lower, cutoff, upper, 0.1, 0.01, 0.9, 0.99, 1.0], [cutoff] * 8, [0.0] * 8],
        index=["adjacent", "equal", "zero"],
        columns=list("abcdefgh"),
        dtype=np.float32,
    )
    original = values.copy(deep=True)
    selected = {"adjacent": "ae" if direction == "less" else "cdfgh", "equal": ""}
    selected["zero"] = "abcdefgh" if direction == "less" else ""
    result = dc.ora(
        values,
        net,
        n_bg=None,
        value_cutoff=float(cutoff),
        value_direction=direction,
        pre_load=pre_load,
        tmin=2,
    )
    _assert_fisher(result, selected)
    pd.testing.assert_frame_equal(values, original)


@pytest.mark.parametrize(
    "kind,offset,pre_load",
    [
        ("sparse", 0, False),
        ("sparse", 0, True),
        ("dask", 0, False),
        ("sparse", 2**24, False),
        ("sparse", 2**24, True),
        ("backed", 0, False),
        ("backed", 2**24, False),
    ],
)
def test_value_cutoff_sparse_precision_and_batches(
    deg_data, kind, offset, pre_load, tmp_path
):
    labels, net = deg_data
    values = labels if not offset else labels.astype(np.int64) + offset
    original = values.copy(deep=True)
    if offset:
        # ORA converts matrix values to float32 before thresholding, so
        # these adjacent integers both become equal to the cutoff.
        assert np.float32(offset) == np.float32(offset + 1)
    data = ad.AnnData(values)
    data.X = sps.csr_matrix(data.X)
    if kind == "dask":
        data.X = da.from_array(data.X, chunks=(2, 3))
    elif kind == "backed":
        data.write_h5ad(tmp_path / "labels.h5ad")
        data = ad.read_h5ad(tmp_path / "labels.h5ad", backed="r")
    try:
        result = dc.ora(
            data,
            net,
            n_bg=None,
            value_cutoff=offset or 0.5,
            tmin=2,
            bsize=2,
            pre_load=pre_load,
        )
        assert result is None
        assert data.shape == labels.shape
        assert data.X.dtype == values.values.dtype
        selected = dict.fromkeys(SELECTED, "") if offset else SELECTED
        _assert_fisher((data.obsm["score_ora"], data.obsm["padj_ora"]), selected)
        matrix = data.X.compute() if kind == "dask" else data.X[:]
        np.testing.assert_array_equal(matrix.toarray(), original.values)
        pd.testing.assert_frame_equal(values, original)
    finally:
        if kind == "backed":
            data.file.close()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_up": 2}, "n_up"),
        ({"n_bm": 1}, "n_bm"),
        ({"value_cutoff": np.nan}, "value_cutoff"),
        ({"value_cutoff": np.inf}, "value_cutoff"),
        ({"value_cutoff": "0.5"}, "value_cutoff"),
        ({"value_direction": "invalid"}, "value_direction"),
        ({"alternative": "invalid"}, "alternative"),
        ({"n_bg": "auto"}, "n_bg"),
        ({"n_bg": 1}, "union"),
        ({"n_bg": 2**31}, "int32"),
    ],
)
def test_value_cutoff_validation(deg_data, kwargs, match):
    labels, net = deg_data
    options = {"n_bg": None, "value_cutoff": 0.5, **kwargs}
    with pytest.raises(ValueError, match=match):
        dc.ora(labels, net, tmin=2, **options)


@pytest.mark.parametrize("kind,value", [("dataframe", np.nan), ("dask", np.inf)])
def test_value_cutoff_nonfinite(deg_data, kind, value):
    labels, net = deg_data
    data = labels.astype(float)
    data.iloc[0, 0] = value
    if kind == "dask":
        data = ad.AnnData(data)
        data.X = da.from_array(data.X, chunks=(2, 3))
    with pytest.raises(ValueError) as exc:
        dc.ora(data, net, n_bg=None, value_cutoff=0.05, value_direction="less", tmin=2)
    message = str(exc.value).lower()
    assert "finite" in message
