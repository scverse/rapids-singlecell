from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from sklearn.neighbors import KDTree

import rapids_singlecell as rsc
from rapids_singlecell.tools._ingest import _knn_classify, _query_neighbors, ingest

N_FEATURES = 8
N_COMPONENTS = 5
N_NEIGHBORS = 5

# Scanpy parity cases are semantically adapted from its BSD-3-Clause
# ``tests/test_ingest.py`` at tag 1.12.1.


def _clustered_data(*, gpu: bool = False) -> tuple[AnnData, AnnData]:
    rng = np.random.default_rng(42)
    ref_values = np.concatenate(
        [
            rng.normal(0.0, 0.4, size=(40, N_FEATURES)),
            rng.normal(8.0, 0.4, size=(40, N_FEATURES)),
        ]
    ).astype(np.float32)
    query_values = np.concatenate(
        [
            rng.normal(0.0, 0.4, size=(3, N_FEATURES)),
            rng.normal(8.0, 0.4, size=(3, N_FEATURES)),
        ]
    ).astype(np.float32)

    ref_X = cp.asarray(ref_values) if gpu else ref_values
    query_X = cp.asarray(query_values) if gpu else query_values
    var_names = [f"gene_{i}" for i in range(N_FEATURES)]
    ref = AnnData(ref_X, var=pd.DataFrame(index=var_names))
    query = AnnData(query_X, var=pd.DataFrame(index=var_names))
    ref.obs["label"] = pd.Categorical(
        ["low"] * 40 + ["high"] * 40,
        categories=["high", "low", "unused"],
        ordered=True,
    )
    ref.obs["second"] = pd.Categorical(["left"] * 40 + ["right"] * 40)
    return ref, query


@pytest.fixture
def fitted_data() -> tuple[AnnData, AnnData]:
    ref, query = _clustered_data()
    rsc.pp.pca(ref, n_comps=N_COMPONENTS)
    rsc.pp.neighbors(
        ref,
        n_neighbors=N_NEIGHBORS,
        use_rep="X_pca",
    )
    return ref, query


def test_public_api() -> None:
    assert rsc.tl.ingest is ingest


def test_pca_projection_uses_reference_mean(fitted_data) -> None:
    ref, query = fitted_data

    result = rsc.tl.ingest(
        query,
        ref,
        embedding_method="pca",
        inplace=False,
    )

    expected = (query.X - ref.X.mean(axis=0)) @ ref.varm["PCs"]
    np.testing.assert_allclose(
        result.obsm["X_pca"],
        expected,
        rtol=2e-5,
        atol=2e-5,
    )
    assert isinstance(result.obsm["X_pca"], np.ndarray)


@pytest.mark.parametrize("zero_center", [False, True])
def test_pca_projection_matches_scanpy(zero_center: bool) -> None:
    """Compare where Scanpy and RSC apply the same fitted PCA transform."""
    ref, query = _clustered_data()
    if zero_center:
        query.X += ref.X.mean(axis=0) - query.X.mean(axis=0)
    sc.pp.pca(
        ref,
        n_comps=N_COMPONENTS,
        zero_center=zero_center,
        random_state=0,
    )
    sc.pp.neighbors(ref, n_neighbors=N_NEIGHBORS, use_rep="X")

    expected = sc.tl.ingest(
        query,
        ref,
        embedding_method="pca",
        inplace=False,
    )
    result = rsc.tl.ingest(
        query,
        ref,
        embedding_method="pca",
        inplace=False,
    )

    np.testing.assert_allclose(
        result.obsm["X_pca"],
        expected.obsm["X_pca"],
        rtol=1e-5,
        atol=1e-5,
    )


def test_pca_only_does_not_require_neighbor_representation(fitted_data) -> None:
    ref, query = fitted_data
    ref.obsm["shared"] = np.zeros((ref.n_obs, 2), dtype=np.float32)
    ref.uns["neighbors"]["params"]["use_rep"] = "shared"

    result = rsc.tl.ingest(
        query,
        ref,
        embedding_method="pca",
        inplace=False,
    )

    assert result.obsm["X_pca"].shape == (query.n_obs, N_COMPONENTS)


def test_pca_projection_is_query_batch_invariant(fitted_data) -> None:
    ref, query = fitted_data
    whole = rsc.tl.ingest(
        query,
        ref,
        obs="label",
        embedding_method="pca",
        inplace=False,
    )

    single_results = [
        rsc.tl.ingest(
            query[[i]].copy(),
            ref,
            obs="label",
            embedding_method="pca",
            inplace=False,
        )
        for i in range(query.n_obs)
    ]
    split_pca = np.vstack([result.obsm["X_pca"] for result in single_results])
    split_labels = [result.obs["label"].iloc[0] for result in single_results]

    np.testing.assert_allclose(
        split_pca,
        whole.obsm["X_pca"],
        rtol=1e-6,
        atol=5e-6,
    )
    assert split_labels == whole.obs["label"].tolist()
    assert np.linalg.norm(split_pca[-1]) > 1.0


def test_copy_inplace_and_multiple_obs(fitted_data) -> None:
    ref, query = fitted_data
    result = rsc.tl.ingest(
        query,
        ref,
        obs=["label", "second"],
        embedding_method="pca",
        inplace=False,
    )

    assert "X_pca" not in query.obsm
    assert "label" not in query.obs
    assert {"label", "second"}.issubset(result.obs)
    assert "rep" not in result.obsm
    assert result.obs["label"].cat.ordered
    assert result.obs["label"].cat.categories.tolist() == [
        "high",
        "low",
        "unused",
    ]

    returned = rsc.tl.ingest(
        query,
        ref,
        obs="label",
        embedding_method="pca",
        inplace=True,
    )
    assert returned is None
    assert "X_pca" in query.obsm
    assert "label" in query.obs


def test_sparse_layer_mask_and_neighbors_key() -> None:
    rng = np.random.default_rng(7)
    ref_values = rng.normal(size=(40, 12)).astype(np.float32)
    query_values = rng.normal(size=(6, 12)).astype(np.float32)
    mask = np.zeros(12, dtype=bool)
    mask[::2] = True
    ref_values[:, ~mask] = np.nan
    query_values[:, ~mask] = np.inf

    ref = AnnData(np.zeros_like(ref_values))
    query = AnnData(np.zeros_like(query_values))
    ref.layers["scaled"] = sparse.csr_matrix(ref_values)
    query.layers["scaled"] = sparse.csr_matrix(query_values)
    ref.var["feature_mask"] = mask
    rsc.pp.pca(
        ref,
        n_comps=4,
        layer="scaled",
        mask_var="feature_mask",
    )
    rsc.pp.neighbors(
        ref,
        n_neighbors=4,
        use_rep="X_pca",
        key_added="alternate",
    )

    result = rsc.tl.ingest(
        query,
        ref,
        embedding_method="pca",
        neighbors_key="alternate",
        inplace=False,
    )

    expected = (query_values[:, mask] - ref_values[:, mask].mean(axis=0)) @ ref.varm[
        "PCs"
    ][mask]
    assert np.isfinite(result.obsm["X_pca"]).all()
    np.testing.assert_allclose(
        result.obsm["X_pca"],
        expected,
        rtol=2e-5,
        atol=2e-5,
    )


def test_uncentered_sparse_pca_projection() -> None:
    rng = np.random.default_rng(8)
    ref_values = rng.normal(size=(40, 12)).astype(np.float64)
    query_values = rng.normal(size=(6, 12)).astype(np.float64)
    ref = AnnData(sparse.csr_matrix(ref_values))
    query = AnnData(sparse.csr_matrix(query_values))
    rsc.pp.pca(ref, n_comps=4, zero_center=False, dtype="float64")
    rsc.pp.neighbors(ref, n_neighbors=4, use_rep="X_pca")

    result = rsc.tl.ingest(
        query,
        ref,
        embedding_method="pca",
        inplace=False,
    )

    expected = query_values @ ref.varm["PCs"]
    np.testing.assert_allclose(
        result.obsm["X_pca"],
        expected,
        rtol=1e-10,
        atol=1e-10,
    )
    assert result.obsm["X_pca"].dtype == np.float64


def test_neighbors_on_gpu_X_do_not_require_pca() -> None:
    ref, query = _clustered_data(gpu=True)
    rsc.pp.neighbors(
        ref,
        n_neighbors=3,
        n_pcs=3,
    )

    result = rsc.tl.ingest(
        query,
        ref,
        obs="label",
        embedding_method=[],
        inplace=False,
    )

    assert result.obs["label"].tolist() == ["low"] * 3 + ["high"] * 3
    assert "X_pca" not in result.obsm


def test_sparse_X_label_transfer() -> None:
    ref, query = _clustered_data()
    ref.X = sparse.csr_matrix(ref.X)
    query.X = sparse.csr_matrix(query.X)
    rsc.pp.neighbors(ref, n_neighbors=3, use_rep="X")

    result = rsc.tl.ingest(
        query,
        ref,
        obs="label",
        embedding_method=[],
        inplace=False,
    )

    assert result.obs["label"].tolist() == ["low"] * 3 + ["high"] * 3


def test_label_transfer_matches_scanpy_on_X() -> None:
    ref, query = _clustered_data()
    sc.pp.neighbors(ref, n_neighbors=N_NEIGHBORS, use_rep="X", random_state=0)

    expected = sc.tl.ingest(
        query,
        ref,
        obs="label",
        embedding_method=(),
        inplace=False,
    )
    result = rsc.tl.ingest(
        query,
        ref,
        obs="label",
        embedding_method=(),
        inplace=False,
    )

    np.testing.assert_array_equal(
        result.obs["label"].to_numpy(), expected.obs["label"].to_numpy()
    )
    assert result.obs["label"].cat.categories.equals(
        expected.obs["label"].cat.categories
    )


@pytest.mark.parametrize("n_neighbors", [3, 4])
def test_brute_neighbors_match_scanpy_oracle(n_neighbors: int) -> None:
    """Adapt Scanpy's ingest-neighbor test to the exact GPU search."""
    ref, query = _clustered_data()
    expected = KDTree(ref.X).query(
        query.X,
        k=n_neighbors,
        return_distance=False,
    )

    result = cp.asnumpy(
        _query_neighbors(
            cp.asarray(ref.X),
            cp.asarray(query.X),
            n_neighbors=n_neighbors,
            algorithm="brute",
            metric="euclidean",
            metric_kwds={},
            algorithm_kwds={},
        )
    )

    np.testing.assert_array_equal(result, expected)


def test_automatic_neighbors_reuse_all_stored_pcs() -> None:
    n_features = 60
    ref_values = np.zeros((4, n_features), dtype=np.float32)
    ref_values[:2, 55] = 100.0
    ref_values[2:, 0] = 10.0
    query_values = np.zeros((1, n_features), dtype=np.float32)
    var_names = [f"gene_{i}" for i in range(n_features)]
    ref = AnnData(ref_values, var=pd.DataFrame(index=var_names))
    query = AnnData(query_values, var=pd.DataFrame(index=var_names))
    ref.obs["label"] = pd.Categorical(["first-50", "first-50", "all-pcs", "all-pcs"])
    ref.varm["PCs"] = np.eye(n_features, dtype=np.float32)
    ref.obsm["X_pca"] = ref_values.copy()
    ref.uns["pca"] = {"params": {"zero_center": False}}
    rsc.pp.neighbors(ref, n_neighbors=2)

    result = rsc.tl.ingest(
        query,
        ref,
        obs="label",
        embedding_method=[],
        inplace=False,
    )

    assert result.obs["label"].iloc[0] == "all-pcs"


def test_custom_neighbor_representation() -> None:
    ref, query = _clustered_data()
    ref.obsm["shared"] = np.concatenate(
        [np.zeros((40, 2)), np.full((40, 2), 10.0)]
    ).astype(np.float32)
    query.obsm["shared"] = np.concatenate(
        [np.zeros((3, 2)), np.full((3, 2), 10.0)]
    ).astype(np.float32)
    rsc.pp.neighbors(ref, n_neighbors=3, use_rep="shared")

    result = rsc.tl.ingest(
        query,
        ref,
        obs="label",
        embedding_method=[],
        inplace=False,
    )
    assert result.obs["label"].tolist() == ["low"] * 3 + ["high"] * 3

    del query.obsm["shared"]
    with pytest.raises(ValueError, match="shared representation"):
        rsc.tl.ingest(
            query,
            ref,
            obs="label",
            embedding_method=[],
        )


def test_knn_classify_preserves_categories_and_ignores_missing() -> None:
    labels = pd.Series(
        pd.Categorical(
            ["z", None, "a", "z"],
            categories=["z", "a", "unused"],
            ordered=True,
        )
    )
    indices = cp.asarray(
        [
            [0, 2],
            [1, 2],
            [1, 1],
            [-1, 99],
        ],
        dtype=cp.int32,
    )

    result = _knn_classify(labels, indices)

    assert result.categories.tolist() == ["z", "a", "unused"]
    assert result.ordered
    assert result[0] == "z"
    assert result[1] == "a"
    assert pd.isna(result[2])
    assert pd.isna(result[3])


def test_umap_mapping_uses_reference_state() -> None:
    ref, query = _clustered_data()
    rsc.pp.neighbors(ref, n_neighbors=8, use_rep="X")
    rsc.tl.umap(ref, random_state=0)
    ref_embedding = ref.obsm["X_umap"].copy()

    result = rsc.tl.ingest(
        query,
        ref,
        embedding_method="umap",
        inplace=False,
    )
    repeated = rsc.tl.ingest(
        query,
        ref,
        embedding_method="umap",
        inplace=False,
    )
    scanpy_result = sc.tl.ingest(
        query,
        ref,
        embedding_method="umap",
        inplace=False,
    )

    assert result.obsm["X_umap"].shape == (query.n_obs, 2)
    assert isinstance(result.obsm["X_umap"], np.ndarray)
    assert np.isfinite(result.obsm["X_umap"]).all()
    np.testing.assert_array_equal(result.obsm["X_umap"], repeated.obsm["X_umap"])
    np.testing.assert_array_equal(ref.obsm["X_umap"], ref_embedding)
    assert "rep" not in result.obsm

    centroids = np.vstack(
        [
            ref_embedding[ref.obs["label"] == "low"].mean(axis=0),
            ref_embedding[ref.obs["label"] == "high"].mean(axis=0),
        ]
    )
    distances = np.linalg.norm(
        result.obsm["X_umap"][:, None, :] - centroids[None, :, :], axis=2
    )
    scanpy_distances = np.linalg.norm(
        scanpy_result.obsm["X_umap"][:, None, :] - centroids[None, :, :], axis=2
    )
    expected_clusters = [0] * 3 + [1] * 3
    assert distances.argmin(axis=1).tolist() == expected_clusters
    assert scanpy_distances.argmin(axis=1).tolist() == expected_clusters


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"embedding_method": "tsne"}, "Invalid embedding_method"),
        ({"obs": "label", "labeling_method": "svm"}, "Invalid labeling_method"),
        ({"algorithm": "unknown"}, "Invalid algorithm"),
    ],
)
def test_invalid_methods_raise(fitted_data, kwargs, match) -> None:
    ref, query = fitted_data
    with pytest.raises(ValueError, match=match):
        rsc.tl.ingest(query, ref, **kwargs)


def test_mismatched_variables_raise(fitted_data) -> None:
    ref, query = fitted_data
    query.var_names = [*query.var_names[:-1], "different"]

    with pytest.raises(ValueError, match="Variables in `adata`"):
        rsc.tl.ingest(query, ref, embedding_method="pca")


def test_missing_prerequisites_raise() -> None:
    ref, query = _clustered_data()
    with pytest.raises(ValueError, match="neighbors"):
        rsc.tl.ingest(query, ref, embedding_method=[])

    rsc.pp.neighbors(ref, n_neighbors=3, use_rep="X")
    with pytest.raises(ValueError, match="missing PCA"):
        rsc.tl.ingest(query, ref, embedding_method="pca")
    with pytest.raises(ValueError, match="missing UMAP"):
        rsc.tl.ingest(query, ref, embedding_method="umap")
