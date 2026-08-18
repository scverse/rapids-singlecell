from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import scanpy as sc
from scanpy.datasets import pbmc3k_processed

import rapids_singlecell as rsc

MULTI_GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() >= 2


def _pbmc3k_with_missing_leiden():
    adata = pbmc3k_processed()
    var_names = adata.var_names
    adata = adata.raw.to_adata()[:, var_names]

    groups = adata.obs["louvain"].value_counts().index[:3].tolist()
    rows = np.concatenate(
        [
            np.flatnonzero((adata.obs["louvain"] == group).to_numpy())[:40]
            for group in groups
        ]
    )
    adata = adata[rows].copy()
    X = adata.X.toarray()
    test_and_reference = adata.obs["louvain"].isin(groups[:2]).to_numpy()
    variable = np.flatnonzero(np.ptp(X[test_and_reference], axis=0) > 0)[:96]
    assert len(variable) == 96
    adata = adata[:, variable].copy()
    adata.X = X[:, variable]
    adata.uns["log1p"] = {"base": None}

    adata.obs["leiden"] = adata.obs["louvain"].copy()
    missing = np.flatnonzero((adata.obs["leiden"] == groups[2]).to_numpy())[:8]
    adata.obs.iloc[missing, adata.obs.columns.get_loc("leiden")] = np.nan

    assert adata.obs["leiden"].isna().sum() == len(missing)
    return adata, groups[0], groups[1]


def _assert_group_results_match(actual, expected, group):
    actual_result = actual.uns["rank_genes_groups"]
    expected_result = expected.uns["rank_genes_groups"]

    actual_names = list(actual_result["names"][group])
    expected_names = list(expected_result["names"][group])
    assert set(actual_names) == set(expected_names)

    for field in ("scores", "logfoldchanges", "pvals", "pvals_adj"):
        actual_by_gene = dict(
            zip(actual_names, np.asarray(actual_result[field][group], dtype=float))
        )
        expected_by_gene = dict(
            zip(expected_names, np.asarray(expected_result[field][group], dtype=float))
        )
        for gene, value in actual_by_gene.items():
            np.testing.assert_allclose(
                value,
                expected_by_gene[gene],
                rtol=1e-5,
                atol=1e-7,
                equal_nan=True,
                err_msg=f"{field} mismatch for gene {gene!r}",
            )

    for field in ("pts", "pts_rest"):
        if field not in expected_result:
            continue
        assert list(actual_result[field].columns) == list(
            expected_result[field].columns
        )
        assert list(actual_result[field].index) == list(expected_result[field].index)
        np.testing.assert_allclose(
            actual_result[field].to_numpy(),
            expected_result[field].to_numpy(),
            rtol=1e-7,
            atol=1e-8,
        )


@pytest.mark.parametrize(
    ("method", "execution"),
    [
        pytest.param("t-test", "host"),
        pytest.param("t-test", "device"),
        pytest.param(
            "t-test",
            "multi-gpu",
            marks=pytest.mark.skipif(
                not MULTI_GPU_AVAILABLE, reason="requires at least two GPUs"
            ),
        ),
        pytest.param("wilcoxon", "host"),
        pytest.param("wilcoxon", "device"),
    ],
)
@pytest.mark.parametrize("reference_kind", ["rest", "explicit"])
def test_rank_genes_groups_missing_leiden_matches_scanpy(
    method, execution, reference_kind
):
    adata, test_group, reference_group = _pbmc3k_with_missing_leiden()
    expected = adata.copy()
    actual = adata.copy()
    reference = "rest" if reference_kind == "rest" else reference_group

    if execution == "device":
        actual.X = cp.asarray(actual.X)

    kwargs = {
        "groups": [test_group],
        "reference": reference,
        "method": method,
        "use_raw": False,
        "pts": True,
        "n_genes": actual.n_vars,
    }
    if method == "wilcoxon":
        kwargs["tie_correct"] = True
    if execution == "multi-gpu":
        kwargs["multi_gpu"] = [0, 1]

    rsc.tl.rank_genes_groups(actual, "leiden", **kwargs)
    sc.tl.rank_genes_groups(
        expected,
        "leiden",
        **{key: value for key, value in kwargs.items() if key != "multi_gpu"},
    )

    _assert_group_results_match(actual, expected, test_group)

    if reference == "rest":
        labels = adata.obs["leiden"]
        missing = labels.isna().to_numpy()
        rest = missing | (labels != test_group).to_numpy()
        expected_pts_rest = np.count_nonzero(adata.X[rest], axis=0) / rest.sum()
        dropped_pts_rest = (
            np.count_nonzero(adata.X[rest & ~missing], axis=0) / (rest & ~missing).sum()
        )
        assert not np.array_equal(expected_pts_rest, dropped_pts_rest)
        np.testing.assert_allclose(
            actual.uns["rank_genes_groups"]["pts_rest"][test_group].to_numpy(),
            expected_pts_rest,
            rtol=0,
            atol=0,
        )
