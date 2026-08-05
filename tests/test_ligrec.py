from __future__ import annotations

import pickle
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData, read_h5ad

from rapids_singlecell.gr import ligrec

HERE = Path(__file__).parent

_CK = "leiden"
Interactions_t = tuple[Sequence[str], Sequence[str]]
Complexes_t = Sequence[tuple[str, str]]


@pytest.fixture()
def adata() -> AnnData:
    file = HERE / "_data/test_data.h5ad"
    adata = read_h5ad(file)
    adata.raw = adata.copy()
    return adata


@pytest.fixture()
def interactions(adata: AnnData) -> tuple[Sequence[str], Sequence[str]]:
    return tuple(product(adata.raw.var_names[:5], adata.raw.var_names[:5]))


@pytest.fixture()
def paul15() -> AnnData:
    # session because we don't modify this dataset
    adata = sc.datasets.paul15()
    sc.pp.normalize_total(adata)
    adata.raw = adata.copy()

    return adata


@pytest.fixture()
def paul15_means() -> pd.DataFrame:
    with (HERE / "_data/paul15_means.pickle").open("rb") as fin:
        return pickle.load(fin)


class TestInvalidBehavior:
    def test_not_adata(self, interactions: Interactions_t):
        with pytest.raises(
            TypeError, match=r"Expected `adata` to be of type `anndata.AnnData`"
        ):
            ligrec(None, _CK, interactions=interactions)

    def test_adata_no_raw(self, adata: AnnData, interactions: Interactions_t):
        del adata.raw
        with pytest.raises(AttributeError, match=r"No `.raw` attribute"):
            ligrec(adata, _CK, use_raw=True, interactions=interactions)

    def test_invalid_cluster_key(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(KeyError, match=r"Cluster key `foobar` not found"):
            ligrec(adata, cluster_key="foobar", interactions=interactions)

    def test_cluster_key_is_not_categorical(
        self, adata: AnnData, interactions: Interactions_t
    ):
        adata.obs[_CK] = adata.obs[_CK].astype("string")
        with pytest.raises(
            TypeError, match=rf"Expected `adata.obs\[{_CK!r}\]` to be `categorical`"
        ):
            ligrec(adata, _CK, interactions=interactions)

    def test_only_1_cluster(self, adata: AnnData, interactions: Interactions_t):
        adata.obs["foo"] = 1
        adata.obs["foo"] = adata.obs["foo"].astype("category")
        with pytest.raises(
            ValueError, match=r"Expected at least `2` clusters, found `1`."
        ):
            ligrec(adata, "foo", interactions=interactions)

    def test_invalid_complex_policy(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(
            ValueError, match=r"Invalid option `'foobar'` for `ComplexPolicy`."
        ):
            ligrec(adata, _CK, interactions=interactions, complex_policy="foobar")

    def test_invalid_fdr_axis(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(
            ValueError, match=r"Invalid option `foobar` for `CorrAxis`."
        ):
            ligrec(
                adata,
                _CK,
                interactions=interactions,
                corr_axis="foobar",
                corr_method="fdr_bh",
            )

    def test_too_few_permutations(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(ValueError, match=r"Expected `n_perms` to be positive"):
            ligrec(adata, _CK, interactions=interactions, n_perms=0)

    def test_invalid_interactions_type(self, adata: AnnData):
        with pytest.raises(TypeError, match=r"Expected either a `pandas.DataFrame`"):
            ligrec(adata, _CK, interactions=42)

    def test_invalid_interactions_dict(self, adata: AnnData):
        with pytest.raises(KeyError, match=r"Column .* is not in `interactions`."):
            ligrec(adata, _CK, interactions={"foo": ["foo"], "target": ["bar"]})
        with pytest.raises(KeyError, match=r"Column .* is not in `interactions`."):
            ligrec(adata, _CK, interactions={"source": ["foo"], "bar": ["bar"]})

    def test_invalid_interactions_dataframe(
        self, adata: AnnData, interactions: Interactions_t
    ):
        df = pd.DataFrame(interactions, columns=["foo", "target"])
        with pytest.raises(KeyError, match=r"Column .* is not in `interactions`."):
            ligrec(adata, _CK, interactions=df)

        df = pd.DataFrame(interactions, columns=["source", "bar"])
        with pytest.raises(KeyError, match=r"Column .* is not in `interactions`."):
            ligrec(adata, _CK, interactions=df)

    def test_interactions_invalid_sequence(
        self, adata: AnnData, interactions: Interactions_t
    ):
        interactions += ("foo", "bar", "bar")  # type: ignore
        with pytest.raises(
            ValueError, match=r"Not all interactions are of length `2`."
        ):
            ligrec(adata, _CK, interactions=interactions)

    def test_interactions_only_invalid_names(self, adata: AnnData):
        with pytest.raises(ValueError, match=r"After filtering by genes"):
            ligrec(adata, _CK, interactions=["foo", "bar", "baz"])

    def test_invalid_clusters(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(ValueError, match=r"Invalid cluster `'foo'`."):
            ligrec(adata, _CK, interactions=interactions, clusters=["foo"])

    def test_invalid_clusters_mix(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(
            ValueError, match=r"Expected a `tuple` of length `2`, found `3`."
        ):
            ligrec(
                adata, _CK, interactions=interactions, clusters=["foo", ("bar", "baz")]
            )


class TestValidBehavior:
    def test_fdr_axis_works(self, adata: AnnData, interactions: Interactions_t):
        rc = ligrec(
            adata,
            _CK,
            interactions=interactions,
            n_perms=5,
            corr_axis="clusters",
            copy=True,
        )
        ri = ligrec(
            adata,
            _CK,
            interactions=interactions,
            n_perms=5,
            corr_axis="interactions",
            copy=True,
        )

        np.testing.assert_array_equal(
            np.where(np.isnan(rc["pvalues"])), np.where(np.isnan(ri["pvalues"]))
        )
        mask = np.isnan(rc["pvalues"])

        assert not np.allclose(rc["pvalues"].values[mask], ri["pvalues"].values[mask])

    def test_inplace_key_added(self, adata: AnnData, interactions: Interactions_t):
        assert "foobar" not in adata.uns
        res = ligrec(
            adata,
            _CK,
            interactions=interactions,
            n_perms=5,
            copy=False,
            key_added="foobar",
        )

        assert res is None
        assert isinstance(adata.uns["foobar"], dict)
        r = adata.uns["foobar"]
        assert len(r) == 3
        assert isinstance(r["means"], pd.DataFrame)
        assert isinstance(r["pvalues"], pd.DataFrame)
        assert isinstance(r["metadata"], pd.DataFrame)

    def test_return_no_write(self, adata: AnnData, interactions: Interactions_t):
        assert "foobar" not in adata.uns
        r = ligrec(
            adata,
            _CK,
            interactions=interactions,
            n_perms=5,
            copy=True,
            key_added="foobar",
        )

        assert "foobar" not in adata.uns
        assert len(r) == 3
        assert isinstance(r["means"], pd.DataFrame)
        assert isinstance(r["pvalues"], pd.DataFrame)
        assert isinstance(r["metadata"], pd.DataFrame)

    @pytest.mark.parametrize("fdr_method", [None, "fdr_bh"])
    def test_pvals_in_correct_range(
        self, adata: AnnData, interactions: Interactions_t, fdr_method: str | None
    ):
        r = ligrec(
            adata,
            _CK,
            interactions=interactions,
            n_perms=5,
            copy=True,
            corr_method=fdr_method,
            threshold=0,
        )

        if np.sum(np.isnan(r["pvalues"].values)) == np.prod(r["pvalues"].shape):
            assert fdr_method == "fdr_bh"
        else:
            assert np.nanmax(r["pvalues"].values) <= 1.0, np.nanmax(r["pvalues"].values)
            assert np.nanmin(r["pvalues"].values) >= 0, np.nanmin(r["pvalues"].values)

    def test_result_correct_index(self, adata: AnnData, interactions: Interactions_t):
        r = ligrec(adata, _CK, interactions=interactions, n_perms=5, copy=True)

        np.testing.assert_array_equal(r["means"].index, r["pvalues"].index)
        np.testing.assert_array_equal(r["pvalues"].index, r["metadata"].index)

        np.testing.assert_array_equal(r["means"].columns, r["pvalues"].columns)
        assert not np.array_equal(r["means"].columns, r["metadata"].columns)
        assert not np.array_equal(r["pvalues"].columns, r["metadata"].columns)

    def test_result_is_sparse(self, adata: AnnData, interactions: Interactions_t):
        interactions = pd.DataFrame(interactions, columns=["source", "target"])
        if TYPE_CHECKING:
            assert isinstance(interactions, pd.DataFrame)
        interactions["metadata"] = "foo"
        r = ligrec(adata, _CK, interactions=interactions, n_perms=5, copy=True)

        assert r["means"].sparse.density <= 0.15
        assert r["pvalues"].sparse.density <= 0.95

        with pytest.raises(
            AttributeError,
            match=r"Can only use the '.sparse' accessor with Sparse data.",
        ):
            _ = r["metadata"].sparse

        np.testing.assert_array_equal(r["metadata"].columns, ["metadata"])
        np.testing.assert_array_equal(
            r["metadata"]["metadata"], interactions["metadata"]
        )

    def test_paul15_correct_means(self, paul15: AnnData, paul15_means: pd.DataFrame):
        res = ligrec(
            paul15,
            "paul15_clusters",
            interactions=list(paul15_means.index.to_list()),
            corr_method=None,
            copy=True,
            threshold=0.01,
            n_perms=1,
        )

        np.testing.assert_array_equal(res["means"].index, paul15_means.index)
        np.testing.assert_array_equal(res["means"].columns, paul15_means.columns)
        np.testing.assert_allclose(
            res["means"].values, paul15_means.values, rtol=1e-5, atol=1e-6
        )

    def test_non_uniqueness(self, adata: AnnData, interactions: Interactions_t):
        # add complexes
        expected = {(r.upper(), l.upper()) for r, l in interactions}
        interactions += (  # type: ignore
            (
                f"{interactions[-1][0]}_{interactions[-1][1]}",
                f"{interactions[-2][0]}_{interactions[-2][1]}",
            ),
        ) * 2
        interactions += interactions[:3]  # type: ignore
        res = ligrec(
            adata,
            _CK,
            interactions=interactions,
            n_perms=1,
            copy=True,
        )

        assert len(res["pvalues"]) == len(expected)
        assert set(res["pvalues"].index.to_list()) == expected

    @pytest.mark.parametrize("use_raw", [False, True])
    def test_gene_symbols(self, adata: AnnData, use_raw: bool):
        gene_ids = adata.var["gene_ids"]
        interactions = tuple(product(gene_ids[:5], gene_ids[:5]))
        res = ligrec(
            adata,
            _CK,
            interactions=interactions,
            n_perms=5,
            use_raw=use_raw,
            copy=True,
            gene_symbols="gene_ids",
        )

        np.testing.assert_array_equal(
            res["means"].index, pd.MultiIndex.from_tuples(interactions)
        )
        np.testing.assert_array_equal(
            res["pvalues"].index, pd.MultiIndex.from_tuples(interactions)
        )
        np.testing.assert_array_equal(
            res["metadata"].index, pd.MultiIndex.from_tuples(interactions)
        )
