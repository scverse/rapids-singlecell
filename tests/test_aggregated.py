from __future__ import annotations

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata.tests.helpers import assert_equal
from packaging.version import Version
from scanpy._utils import _resolve_axis
from scanpy.datasets import pbmc3k_processed
from scipy.sparse import csr_matrix

import rapids_singlecell as rsc
from testing.rapids_singlecell._helper import ARRAY_TYPES_MEM


@pytest.fixture
def df_base():
    ax_base = ["A", "B"]
    return pd.DataFrame(index=ax_base)


@pytest.fixture
def df_groupby():
    ax_groupby = [
        *["v0", "v1", "v2"],
        *["w0", "w1"],
        *["a1", "a2", "a3"],
        *["b1", "b2"],
        *["c1", "c2"],
        "d0",
    ]

    df_groupby = pd.DataFrame(index=pd.Index(ax_groupby, name="cell"))
    df_groupby["key"] = pd.Categorical([c[0] for c in ax_groupby])
    df_groupby["key_superset"] = pd.Categorical([c[0] for c in ax_groupby]).map(
        {"v": "v", "w": "v", "a": "a", "b": "a", "c": "a", "d": "a"}
    )
    df_groupby["key_subset"] = pd.Categorical([c[1] for c in ax_groupby])
    df_groupby["weight"] = 2.0
    return df_groupby


@pytest.fixture
def X():
    data = [
        *[[0, -2], [1, 13], [2, 1]],  # v
        *[[3, 12], [4, 2]],  # w
        *[[5, 11], [6, 3], [7, 10]],  # a
        *[[8, 4], [9, 9]],  # b
        *[[10, 5], [11, 8]],  # c
        [12, 6],  # d
    ]
    return np.array(data, dtype=np.float32)


def gen_adata(data_key, dim, df_base, df_groupby, X):
    if (data_key == "varm" and dim == "obs") or (data_key == "obsm" and dim == "var"):
        pytest.skip("invalid parameter combination")

    obs_df, var_df = (df_groupby, df_base) if dim == "obs" else (df_base, df_groupby)
    data = X.T if dim == "var" and data_key != "varm" else X
    if data_key != "X":
        data_dict_sparse = {data_key: {"test": csr_matrix(data)}}
        data_dict_dense = {data_key: {"test": data}}
    else:
        data_dict_sparse = {data_key: csr_matrix(data)}
        data_dict_dense = {data_key: data}

    adata_sparse = ad.AnnData(obs=obs_df, var=var_df, **data_dict_sparse)
    adata_dense = ad.AnnData(obs=obs_df, var=var_df, **data_dict_dense)
    return adata_sparse, adata_dense


@pytest.mark.parametrize("axis", [0, 1])
def test_mask(axis):
    blobs = sc.datasets.blobs()
    mask = blobs.obs["blobs"] == 0
    blobs.obs["mask_col"] = mask
    if axis == 1:
        blobs = blobs.T
    rsc.get.anndata_to_GPU(blobs)
    by_name = rsc.get.aggregate(blobs, "blobs", "sum", axis=axis, mask="mask_col")
    by_value = rsc.get.aggregate(blobs, "blobs", "sum", axis=axis, mask=mask)

    assert_equal(by_name, by_value)

    assert np.all(by_name["0"].layers["sum"] == 0)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_MEM)
def test_masked_mean_var(array_type):
    X = np.array(
        [
            [1, 2],
            [3, 6],
            [100, 100],
            [2, 4],
            [4, 8],
            [200, 200],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {
            "group": pd.Categorical(["a", "a", "a", "b", "b", "b"]),
            "mask": [True, True, False, True, True, False],
        }
    )
    adata = ad.AnnData(X=array_type(X), obs=obs)
    rsc.get.anndata_to_GPU(adata)

    result = rsc.get.aggregate(adata, by="group", func=["mean", "var"], mask="mask")

    cp.testing.assert_allclose(
        result.layers["mean"], cp.array([[2, 4], [3, 6]], dtype=cp.float64)
    )
    cp.testing.assert_allclose(
        result.layers["var"], cp.array([[2, 8], [2, 8]], dtype=cp.float64)
    )


@pytest.mark.parametrize("array_type", ARRAY_TYPES_MEM)
@pytest.mark.parametrize("metric", ["sum", "mean", "var", "count_nonzero"])
def test_aggregate_vs_pandas(metric, array_type):
    adata = pbmc3k_processed().raw.to_adata()

    adata = adata[
        adata.obs["louvain"].isin(adata.obs["louvain"].cat.categories[:5]), :1_000
    ].copy()
    adata.X = array_type(adata.X)
    rsc.get.anndata_to_GPU(adata)
    adata.obs["percent_mito_binned"] = pd.cut(adata.obs["percent_mito"], bins=5)
    result = rsc.get.aggregate(adata, ["louvain", "percent_mito_binned"], metric)
    rsc.get.anndata_to_CPU(adata)
    if metric == "count_nonzero":
        expected = (
            (adata.to_df() != 0)
            .astype(np.float64)
            .join(adata.obs[["louvain", "percent_mito_binned"]])
            .groupby(["louvain", "percent_mito_binned"], observed=True)
            .agg("sum")
        )
    else:
        expected = (
            adata.to_df()
            .astype(np.float64)
            .join(adata.obs[["louvain", "percent_mito_binned"]])
            .groupby(["louvain", "percent_mito_binned"], observed=True)
            .agg(metric)
        )
    expected.index = expected.index.to_frame().apply(
        lambda x: "_".join(map(str, x)), axis=1
    )
    expected.index.name = None
    expected.columns.name = None
    rsc.get.anndata_to_CPU(result, convert_all=True)
    result_df = result.to_df(layer=metric)
    result_df.index.name = None
    result_df.columns.name = None

    if Version(pd.__version__) < Version("2"):
        # Order of results returned by groupby changed in pandas 2
        assert expected.shape == result_df.shape
        assert expected.index.isin(result_df.index).all()

        expected = expected.loc[result_df.index]

    pd.testing.assert_frame_equal(result_df, expected, check_dtype=False, atol=1e-5)


@pytest.mark.parametrize("array_type", ARRAY_TYPES_MEM)
@pytest.mark.parametrize("metric", ["sum", "mean", "var", "count_nonzero"])
def test_aggregate_axis(array_type, metric):
    adata = pbmc3k_processed().raw.to_adata()
    adata = adata[
        adata.obs["louvain"].isin(adata.obs["louvain"].cat.categories[:5]), :1_000
    ].copy()
    adata.X = array_type(adata.X)
    rsc.get.anndata_to_GPU(adata)
    expected = rsc.get.aggregate(adata, ["louvain"], metric)
    actual = rsc.get.aggregate(adata.T, ["louvain"], metric, axis=1).T

    assert_equal(expected, actual)


def test_aggregate_entry():
    args = ("blobs", ["mean", "var", "count_nonzero"])

    adata = sc.datasets.blobs()
    rsc.get.anndata_to_GPU(adata)
    X_result = rsc.get.aggregate(adata, *args)

    # layer adata
    layer_adata = ad.AnnData(
        obs=adata.obs,
        var=adata.var,
        layers={"test": adata.X.copy()},
    )
    layer_result = rsc.get.aggregate(layer_adata, *args, layer="test")
    obsm_adata = ad.AnnData(
        obs=adata.obs,
        var=adata.var,
        obsm={"test": adata.X.copy()},
    )
    obsm_result = rsc.get.aggregate(obsm_adata, *args, obsm="test")
    varm_adata = ad.AnnData(
        obs=adata.var,
        var=adata.obs,
        varm={"test": adata.X.copy()},
    )
    varm_result = rsc.get.aggregate(varm_adata, *args, varm="test")

    X_result_min = X_result.copy()
    del X_result_min.var
    X_result_min.var_names = [str(x) for x in np.arange(X_result_min.n_vars)]

    assert_equal(X_result, layer_result)
    assert_equal(X_result_min, obsm_result)
    assert_equal(X_result.layers, obsm_result.layers)
    assert_equal(X_result.layers, varm_result.T.layers)


def test_aggregate_incorrect_dim():
    adata = pbmc3k_processed().raw.to_adata()
    rsc.get.anndata_to_GPU(adata)
    with pytest.raises(ValueError, match="was 'foo'"):
        rsc.get.aggregate(adata, ["louvain"], "sum", axis="foo")


@pytest.mark.parametrize("axis_name", ["obs", "var"])
def test_aggregate_axis_specification(axis_name):
    axis, axis_name = _resolve_axis(axis_name)
    by = "blobs" if axis == 0 else "labels"

    adata = sc.datasets.blobs()
    rsc.get.anndata_to_GPU(adata)
    adata.var["labels"] = np.tile(["a", "b"], adata.shape[1])[: adata.shape[1]]

    agg_index = rsc.get.aggregate(adata, by=by, func="mean", axis=axis)
    agg_name = rsc.get.aggregate(adata, by=by, func="mean", axis=axis_name)

    cp.testing.assert_array_almost_equal(
        agg_index.layers["mean"], agg_name.layers["mean"]
    )

    if axis_name == "obs":
        agg_unspecified = rsc.get.aggregate(adata, by=by, func="mean")
        cp.testing.assert_array_almost_equal(
            agg_name.layers["mean"], agg_unspecified.layers["mean"]
        )


@pytest.mark.parametrize(
    ("matrix", "df", "keys", "metrics", "expected"),
    [
        pytest.param(
            np.block(
                [
                    [np.ones((2, 2)), np.zeros((2, 2))],
                    [np.zeros((2, 2)), np.ones((2, 2))],
                ]
            ),
            pd.DataFrame(
                {
                    "a": ["a", "a", "b", "b"],
                    "b": ["c", "d", "d", "d"],
                }
            ),
            ["a", "b"],
            ["count_nonzero"],  # , "sum", "mean"],
            ad.AnnData(
                obs=pd.DataFrame(
                    {"a": ["a", "a", "b"], "b": ["c", "d", "d"]},
                    index=["a_c", "a_d", "b_d"],
                ).astype("category"),
                var=pd.DataFrame(index=[f"gene_{i}" for i in range(4)]),
                layers={
                    "count_nonzero": np.array(
                        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2]]
                    ),
                    # "sum": np.array([[2, 0], [0, 2]]),
                    # "mean": np.array([[1, 0], [0, 1]]),
                },
            ),
            id="count_nonzero",
        ),
        pytest.param(
            np.block(
                [
                    [np.ones((2, 2)), np.zeros((2, 2))],
                    [np.zeros((2, 2)), np.ones((2, 2))],
                ]
            ),
            pd.DataFrame(
                {
                    "a": ["a", "a", "b", "b"],
                    "b": ["c", "d", "d", "d"],
                }
            ),
            ["a", "b"],
            ["sum", "mean", "count_nonzero"],
            ad.AnnData(
                obs=pd.DataFrame(
                    {"a": ["a", "a", "b"], "b": ["c", "d", "d"]},
                    index=["a_c", "a_d", "b_d"],
                ).astype("category"),
                var=pd.DataFrame(index=[f"gene_{i}" for i in range(4)]),
                layers={
                    "sum": np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2]]),
                    "mean": np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]]),
                    "count_nonzero": np.array(
                        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2]]
                    ),
                },
            ),
            id="sum-mean-count_nonzero",
        ),
        pytest.param(
            np.block(
                [
                    [np.ones((2, 2)), np.zeros((2, 2))],
                    [np.zeros((2, 2)), np.ones((2, 2))],
                ]
            ),
            pd.DataFrame(
                {
                    "a": ["a", "a", "b", "b"],
                    "b": ["c", "d", "d", "d"],
                }
            ),
            ["a", "b"],
            ["mean"],
            ad.AnnData(
                obs=pd.DataFrame(
                    {"a": ["a", "a", "b"], "b": ["c", "d", "d"]},
                    index=["a_c", "a_d", "b_d"],
                ).astype("category"),
                var=pd.DataFrame(index=[f"gene_{i}" for i in range(4)]),
                layers={
                    "mean": np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]]),
                },
            ),
            id="mean",
        ),
    ],
)
def test_aggregate_examples(matrix, df, keys, metrics, expected):
    adata = ad.AnnData(
        X=matrix,
        obs=df,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(matrix.shape[1])]),
    )
    rsc.get.anndata_to_GPU(adata)
    result = rsc.get.aggregate(adata, by=keys, func=metrics)

    assert_equal(expected, result)


def test_factors():
    from itertools import product

    obs = pd.DataFrame(
        product(range(5), range(5), range(5), range(5)), columns=list("abcd")
    )
    obs.index = [f"cell_{i:04d}" for i in range(obs.shape[0])]
    adata = ad.AnnData(
        X=cp.arange(obs.shape[0]).reshape(-1, 1),
        obs=obs,
    )
    adata.X = adata.X.astype(np.float32)
    res = rsc.get.aggregate(adata, by=["a", "b", "c", "d"], func="sum")
    cp.testing.assert_array_equal(res.layers["sum"], adata.X)


@pytest.mark.parametrize("metric", ["sum", "mean", "var", "count_nonzero", "sq_sum"])
def test_sparse_vs_dense(metric):
    adata = pbmc3k_processed().raw.to_adata()
    rsc.get.anndata_to_GPU(adata)
    mask = adata.obs.louvain == "Megakaryocytes"
    rsc_get = rsc.get.aggregate(adata, by="louvain", func=metric, mask=mask)
    kwargs = {
        "by": "louvain",
        "func": metric,
        "mask": mask,
        "return_sparse": True,
    }
    if metric in {"mean", "var"}:
        with pytest.warns(RuntimeWarning, match="implicit-zero entries"):
            rsc_get_sparse = rsc.get.aggregate(adata, **kwargs)
    else:
        rsc_get_sparse = rsc.get.aggregate(adata, **kwargs)

    a = rsc_get_sparse.layers[metric].toarray()
    b = rsc_get.layers[metric]
    if metric in {"mean", "var"}:
        selected_group = np.flatnonzero(
            rsc_get.obs["louvain"].to_numpy() == "Megakaryocytes"
        )
        cp.testing.assert_allclose(a[selected_group], b[selected_group])
        assert cp.any(a[cp.isnan(b)] == 0)
    else:
        cp.testing.assert_allclose(a, b)


def test_c_contiguous_vs_fortran_contiguous():
    adata = pbmc3k_processed().raw.to_adata()
    adata = adata[:, :1000].copy()
    adata.X = adata.X.toarray()
    rsc.get.anndata_to_GPU(adata)
    adata.X = cp.asfortranarray(adata.X)
    mask = adata.obs.louvain == "Megakaryocytes"
    rsc_get_F = rsc.get.aggregate(
        adata, by="louvain", func=["sum", "mean", "var", "count_nonzero"], mask=mask
    )
    adata.X = cp.ascontiguousarray(adata.X)
    rsc_get_C = rsc.get.aggregate(
        adata,
        by="louvain",
        func=["sum", "mean", "var", "count_nonzero"],
        mask=mask,
    )

    for i in range(4):
        c = ["sum", "mean", "var", "count_nonzero"][i]
        a = rsc_get_C.layers[c]
        b = rsc_get_F.layers[c]
        cp.testing.assert_allclose(a, b)
