from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, get_args

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy import sparse as cp_sparse
from scipy import sparse as sc_sparse

from rapids_singlecell.get import X_to_GPU, _get_obs_rep
from rapids_singlecell.preprocessing._neighbors._helper import (
    _check_metrics,
    _check_neighbors_X,
)
from rapids_singlecell.preprocessing._neighbors._neighbors import KNN_ALGORITHMS

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from anndata import AnnData

_EmbeddingMethod = Literal["umap", "pca"]
_LabelingMethod = Literal["knn"]
_IngestAlgorithm = Literal[
    "brute",
    "cagra",
    "ivfflat",
    "ivfpq",
    "mg_ivfflat",
    "mg_ivfpq",
]

_DEFAULT_N_PCS = 50
_MAX_VOTE_COMPARISONS = 16_000_000


def ingest(
    adata: AnnData,
    adata_ref: AnnData,
    *,
    obs: str | Iterable[str] | None = None,
    embedding_method: _EmbeddingMethod | Iterable[_EmbeddingMethod] = (
        "umap",
        "pca",
    ),
    labeling_method: _LabelingMethod | Iterable[_LabelingMethod] = "knn",
    neighbors_key: str | None = None,
    algorithm: _IngestAlgorithm = "brute",
    algorithm_kwds: Mapping[str, Any] = MappingProxyType({}),
    inplace: bool = True,
) -> AnnData | None:
    """Map labels and embeddings from reference data to new data.

    This is a GPU implementation of :func:`scanpy.tl.ingest`. It projects the
    query into the representation used to construct the reference neighbor
    graph, transfers labels by k-nearest-neighbor majority vote, and can map
    reference PCA and UMAP embeddings.

    For centered PCA, the query is centered with the reference feature means.
    This is the fixed-reference PCA transform and makes the projection
    independent of which other query observations are ingested in the same
    call.

    Parameters
    ----------
    adata
        Query data whose annotations and embeddings will be inferred.
    adata_ref
        Reference data with a precomputed neighbor graph and the annotations
        and embeddings to map. Variables must match ``adata`` in the same order.
    obs
        One or more columns from ``adata_ref.obs`` to map to ``adata.obs``.
    embedding_method
        Reference embeddings to map. Supported values are ``"pca"`` and
        ``"umap"``.
    labeling_method
        Label transfer method for each requested ``obs`` column. Only ``"knn"``
        is supported. A single value is reused for all columns.
    neighbors_key
        Key containing the reference neighbor metadata in ``adata_ref.uns``.
        Defaults to ``"neighbors"``.
    algorithm
        GPU nearest-neighbor algorithm used for label transfer. ``"brute"`` is
        exact and is the only option that supports sparse representations.
    algorithm_kwds
        Options passed to the nearest-neighbor algorithm.
    inplace
        Update ``adata`` in place. If ``False``, return an updated copy.

    Returns
    -------
    If ``inplace=False``, returns an updated :class:`~anndata.AnnData`.
    Otherwise updates ``adata`` and returns ``None``. Requested labels are
    stored in ``adata.obs`` and embeddings in ``adata.obsm['X_pca']`` and/or
    ``adata.obsm['X_umap']``.

    Notes
    -----
    A custom representation recorded in the reference neighbor parameters must
    also exist in ``adata.obsm`` and already share the reference coordinate
    system. Query ``X_pca`` is never reused: it is always projected with the
    reference PCA model.
    """
    embedding_methods = _normalize_arg(embedding_method)
    obs_keys = _normalize_optional_arg(obs)
    labeling_methods = _normalize_arg(labeling_method)

    _validate_methods(embedding_methods, obs_keys, labeling_methods)
    _validate_var_names(adata, adata_ref)
    _validate_obs_keys(adata_ref, obs_keys)

    if algorithm not in get_args(_IngestAlgorithm):
        raise ValueError(
            f"Invalid algorithm {algorithm!r}. "
            f"Valid options are: {get_args(_IngestAlgorithm)}."
        )

    neighbors_key = "neighbors" if neighbors_key is None else neighbors_key
    if neighbors_key not in adata_ref.uns:
        raise ValueError(
            f"Did not find `adata_ref.uns[{neighbors_key!r}]`. "
            "Run `rsc.pp.neighbors(adata_ref)` first."
        )

    neighbor_params = adata_ref.uns[neighbors_key].get("params")
    if neighbor_params is None or "n_neighbors" not in neighbor_params:
        raise ValueError(
            f"`adata_ref.uns[{neighbors_key!r}]` does not contain valid "
            "neighbor parameters."
        )

    rep_key, n_rep_dims = _resolve_representation(adata_ref, neighbor_params)
    need_representation = "umap" in embedding_methods or bool(obs_keys)
    need_pca = "pca" in embedding_methods or (
        need_representation and rep_key == "X_pca"
    )
    query_pca = _project_query_pca(adata, adata_ref) if need_pca else None

    ref_rep = query_rep = None
    if need_representation:
        ref_rep, query_rep = _get_representations(
            adata,
            adata_ref,
            rep_key=rep_key,
            n_dims=n_rep_dims,
            query_pca=query_pca,
        )

    mapped_pca = cp.asnumpy(query_pca) if "pca" in embedding_methods else None
    mapped_umap = None
    if "umap" in embedding_methods:
        mapped_umap = cp.asnumpy(
            _map_umap(
                adata_ref,
                ref_rep=ref_rep,
                query_rep=query_rep,
                neighbor_params=neighbor_params,
            )
        )

    mapped_obs = {}
    if obs_keys:
        metric = neighbor_params.get("metric", "euclidean")
        metric_kwds = neighbor_params.get("metric_kwds") or {}
        _check_metrics(algorithm, metric)
        knn_indices = _query_neighbors(
            ref_rep,
            query_rep,
            n_neighbors=int(neighbor_params["n_neighbors"]),
            algorithm=algorithm,
            metric=metric,
            metric_kwds=metric_kwds,
            algorithm_kwds=algorithm_kwds,
        )
        for key in obs_keys:
            mapped_obs[key] = _knn_classify(adata_ref.obs[key], knn_indices)

    out = adata if inplace else adata.copy()
    if mapped_pca is not None:
        out.obsm["X_pca"] = mapped_pca
    if mapped_umap is not None:
        out.obsm["X_umap"] = mapped_umap
    for key, values in mapped_obs.items():
        out.obs[key] = values

    return None if inplace else out


def _normalize_arg[T](value: T | Iterable[T]) -> list[T]:
    return [value] if isinstance(value, str) else list(value)


def _normalize_optional_arg[T](value: T | Iterable[T] | None) -> list[T]:
    if value is None:
        return []
    return _normalize_arg(value)


def _validate_methods(
    embedding_methods: list[str],
    obs_keys: list[str],
    labeling_methods: list[str],
) -> None:
    invalid_embeddings = set(embedding_methods).difference(get_args(_EmbeddingMethod))
    if invalid_embeddings:
        raise ValueError(
            f"Invalid embedding_method values: {sorted(invalid_embeddings)}. "
            f"Valid options are: {get_args(_EmbeddingMethod)}."
        )

    if len(labeling_methods) == 1 and len(obs_keys) > 1:
        labeling_methods *= len(obs_keys)
    if obs_keys and len(labeling_methods) != len(obs_keys):
        raise ValueError(
            "Provide one labeling_method or one method for each requested obs key."
        )
    invalid_labeling = set(labeling_methods).difference(get_args(_LabelingMethod))
    if invalid_labeling:
        raise ValueError(
            f"Invalid labeling_method values: {sorted(invalid_labeling)}. "
            "Only 'knn' is supported."
        )


def _validate_var_names(adata: AnnData, adata_ref: AnnData) -> None:
    if not adata.var_names.str.upper().equals(adata_ref.var_names.str.upper()):
        raise ValueError(
            "Variables in `adata` are different from variables in `adata_ref`."
        )


def _validate_obs_keys(adata_ref: AnnData, obs_keys: list[str]) -> None:
    missing = [key for key in obs_keys if key not in adata_ref.obs]
    if missing:
        raise ValueError(f"Did not find obs keys in `adata_ref`: {missing}.")


def _resolve_representation(
    adata_ref: AnnData, neighbor_params: Mapping[str, Any]
) -> tuple[str, int | None]:
    use_rep = neighbor_params.get("use_rep")
    n_pcs = neighbor_params.get("n_pcs")

    if use_rep == "X":
        return "X", None
    if use_rep is not None:
        return use_rep, n_pcs
    if n_pcs == 0 or (adata_ref.n_vars <= _DEFAULT_N_PCS and adata_ref.X is not None):
        return "X", None
    if "X_pca" not in adata_ref.obsm:
        return "X_pca", None if n_pcs is None else int(n_pcs)
    if n_pcs is not None:
        return "X_pca", int(n_pcs)
    return "X_pca", None


def _project_query_pca(adata: AnnData, adata_ref: AnnData) -> cp.ndarray:
    if "pca" not in adata_ref.uns:
        raise ValueError(
            "`adata_ref` is missing PCA parameters. Run `rsc.pp.pca(adata_ref)` first."
        )

    pca_info = adata_ref.uns["pca"]
    pca_params = pca_info.get("params", {})
    source_obsm = pca_params.get("obsm")

    if source_obsm is not None:
        if source_obsm not in adata_ref.obsm or source_obsm not in adata.obsm:
            raise ValueError(
                f"PCA was fitted from obsm[{source_obsm!r}], which must exist in "
                "both `adata_ref` and `adata`."
            )
        if "components" not in pca_info:
            raise ValueError("Reference PCA metadata does not contain `components`.")
        X_ref = adata_ref.obsm[source_obsm]
        X_query = adata.obsm[source_obsm]
        components = pca_info["components"]
    else:
        if "PCs" not in adata_ref.varm:
            raise ValueError("`adata_ref.varm['PCs']` is missing.")
        layer = pca_params.get("layer")
        X_ref = _get_obs_rep(adata_ref, layer=layer)
        X_query = _get_obs_rep(adata, layer=layer)
        components = adata_ref.varm["PCs"]

    if X_ref.shape[1] != X_query.shape[1]:
        raise ValueError(
            "The query and reference PCA source representations have different "
            "numbers of features."
        )
    if components.shape[0] != X_ref.shape[1]:
        raise ValueError(
            "Reference PCA loadings do not match the PCA source representation."
        )

    if source_obsm is None:
        feature_mask = _resolve_pca_feature_mask(adata_ref, pca_params, components)
        if feature_mask is not None:
            X_ref = X_ref[:, feature_mask]
            X_query = X_query[:, feature_mask]
            components = components[feature_mask]

    compute_dtype = (
        cp.float64 if np.dtype(components.dtype).itemsize > 4 else cp.float32
    )
    output_source = adata_ref.obsm.get("X_pca", components)
    output_dtype = (
        cp.float64 if np.dtype(output_source.dtype).itemsize > 4 else cp.float32
    )
    query_gpu = _as_gpu_matrix(X_query, dtype=compute_dtype)
    components_gpu = cp.asarray(components, dtype=compute_dtype, order="C")
    query_pca = query_gpu @ components_gpu

    if pca_params.get("zero_center", True):
        reference_mean = _column_mean(X_ref, dtype=compute_dtype)
        query_pca -= reference_mean @ components_gpu

    return cp.ascontiguousarray(query_pca, dtype=output_dtype)


def _resolve_pca_feature_mask(
    adata_ref: AnnData,
    pca_params: Mapping[str, Any],
    components,
) -> np.ndarray | None:
    mask = pca_params.get("mask_var")
    if mask is None and pca_params.get("use_highly_variable", False):
        mask = "highly_variable"

    if isinstance(mask, str):
        if mask not in adata_ref.var:
            raise ValueError(f"Did not find `adata_ref.var[{mask!r}]`.")
        mask = adata_ref.var[mask].to_numpy()
    elif mask is None:
        components_host = (
            cp.asnumpy(components) if isinstance(components, cp.ndarray) else components
        )
        active = np.any(np.asarray(components_host) != 0, axis=1)
        mask = active if active.any() and not active.all() else None
    else:
        mask = np.asarray(mask)

    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.ndim != 1 or mask.size != adata_ref.n_vars:
        raise ValueError("Reference PCA variable mask is invalid.")
    return mask


def _column_mean(X, *, dtype) -> cp.ndarray:
    if isinstance(X, cp.ndarray) or cp_sparse.issparse(X):
        mean = X.mean(axis=0)
    elif isinstance(X, np.ndarray) or sc_sparse.issparse(X):
        mean = np.asarray(X.mean(axis=0))
    else:
        raise TypeError(f"Unsupported PCA source type: {type(X)!r}.")
    return cp.asarray(mean, dtype=dtype).reshape(-1)


def _get_representations(
    adata: AnnData,
    adata_ref: AnnData,
    *,
    rep_key: str,
    n_dims: int | None,
    query_pca: cp.ndarray | None,
):
    if rep_key == "X":
        ref_rep = adata_ref.X
        query_rep = adata.X
    elif rep_key == "X_pca":
        if "X_pca" not in adata_ref.obsm or query_pca is None:
            raise ValueError("Reference neighbor metadata requires PCA results.")
        ref_rep = adata_ref.obsm["X_pca"]
        query_rep = query_pca
    else:
        if rep_key not in adata_ref.obsm:
            raise ValueError(f"Did not find `adata_ref.obsm[{rep_key!r}]`.")
        if rep_key not in adata.obsm:
            raise ValueError(
                f"Reference neighbors use obsm[{rep_key!r}]. The same shared "
                "representation must exist in `adata.obsm`."
            )
        ref_rep = adata_ref.obsm[rep_key]
        query_rep = adata.obsm[rep_key]

    if n_dims is not None:
        if n_dims < 1:
            raise ValueError(
                "The neighbor representation must use at least one dimension."
            )
        if ref_rep.shape[1] < n_dims or query_rep.shape[1] < n_dims:
            raise ValueError(
                f"The neighbor metadata requests {n_dims} dimensions, but the "
                "stored representation has fewer."
            )
        ref_rep = ref_rep[:, :n_dims]
        query_rep = query_rep[:, :n_dims]

    ref_gpu = _as_gpu_matrix(ref_rep)
    query_gpu = _as_gpu_matrix(query_rep)
    if ref_gpu.shape[1] != query_gpu.shape[1]:
        raise ValueError(
            "The query and reference neighbor representations have different "
            "numbers of dimensions."
        )
    return ref_gpu, query_gpu


def _as_gpu_matrix(X, *, dtype=cp.float32):
    X = X_to_GPU(X)
    if cp_sparse.issparse(X):
        X = X.tocsr()
        return X if X.dtype == dtype else X.astype(dtype)
    if isinstance(X, cp.ndarray):
        return cp.ascontiguousarray(X, dtype=dtype)
    raise TypeError(f"Unsupported representation type: {type(X)!r}.")


def _query_neighbors(
    ref_rep,
    query_rep,
    *,
    n_neighbors: int,
    algorithm: _IngestAlgorithm,
    metric: str,
    metric_kwds: Mapping[str, Any],
    algorithm_kwds: Mapping[str, Any],
) -> cp.ndarray:
    if ref_rep.shape[0] == 0:
        raise ValueError("`adata_ref` must contain at least one observation.")
    k = min(n_neighbors, ref_rep.shape[0])
    if k < 1:
        raise ValueError("The reference neighbor count must be positive.")
    if query_rep.shape[0] == 0:
        return cp.empty((0, k), dtype=cp.int32)

    ref_search = _check_neighbors_X(ref_rep, algorithm)
    query_search = _check_neighbors_X(query_rep, algorithm)
    indices, _ = KNN_ALGORITHMS[algorithm](
        ref_search,
        query_search,
        k=k,
        metric=metric,
        metric_kwds=metric_kwds,
        algorithm_kwds=algorithm_kwds,
    )
    return cp.asarray(indices, dtype=cp.int32)


def _knn_classify(ref_labels: pd.Series, knn_indices: cp.ndarray) -> pd.Categorical:
    categorical = ref_labels.astype("category")
    categories = categorical.cat.categories
    ordered = categorical.cat.ordered
    n_query, n_neighbors = knn_indices.shape

    if n_query == 0 or len(categories) == 0:
        return pd.Categorical.from_codes(
            np.full(n_query, -1, dtype=np.int32),
            categories=categories,
            ordered=ordered,
        )

    codes = cp.asarray(categorical.cat.codes.to_numpy(), dtype=cp.int32)
    valid_indices = (knn_indices >= 0) & (knn_indices < codes.size)
    safe_indices = cp.where(valid_indices, knn_indices, 0)
    gathered = cp.where(valid_indices, codes[safe_indices], -1)
    mode_codes = cp.empty(n_query, dtype=cp.int32)
    rows_per_batch = max(1, _MAX_VOTE_COMPARISONS // max(1, n_neighbors * n_neighbors))

    for start in range(0, n_query, rows_per_batch):
        stop = min(start + rows_per_batch, n_query)
        values = cp.sort(gathered[start:stop], axis=1)
        counts = (values[:, :, None] == values[:, None, :]).sum(axis=2)
        counts = cp.where(values >= 0, counts, 0)
        best = counts.argmax(axis=1)
        mode_codes[start:stop] = values[cp.arange(stop - start), best]

    return pd.Categorical.from_codes(
        cp.asnumpy(mode_codes), categories=categories, ordered=ordered
    )


def _map_umap(
    adata_ref: AnnData,
    *,
    ref_rep,
    query_rep,
    neighbor_params: Mapping[str, Any],
) -> cp.ndarray:
    if "umap" not in adata_ref.uns or "X_umap" not in adata_ref.obsm:
        raise ValueError(
            "`adata_ref` is missing UMAP results. Run `rsc.tl.umap(adata_ref)` first."
        )

    ref_embedding = cp.asarray(adata_ref.obsm["X_umap"], dtype=cp.float32, order="C")
    if ref_rep.shape[0] == 0:
        raise ValueError("`adata_ref` must contain at least one observation.")
    if ref_embedding.ndim != 2 or ref_embedding.shape[0] != ref_rep.shape[0]:
        raise ValueError("Reference UMAP coordinates do not match `adata_ref`.")
    if query_rep.shape[0] == 0:
        return cp.empty((0, ref_embedding.shape[1]), dtype=cp.float32)

    umap_params = adata_ref.uns["umap"].get("params", {})
    if "a" not in umap_params or "b" not in umap_params:
        raise ValueError("Reference UMAP metadata does not contain `a` and `b`.")

    from cuml.internals.array import CumlArray
    from cuml.internals.array_sparse import SparseCumlArray
    from cuml.manifold import UMAP

    ref_is_sparse = cp_sparse.issparse(ref_rep)
    if ref_is_sparse and not cp_sparse.issparse(query_rep):
        query_rep = cp_sparse.csr_matrix(query_rep)
    elif not ref_is_sparse and cp_sparse.issparse(query_rep):
        query_rep = cp.ascontiguousarray(query_rep.toarray(), dtype=cp.float32)

    n_neighbors = min(int(neighbor_params["n_neighbors"]), ref_rep.shape[0])
    if n_neighbors < 1:
        raise ValueError("The reference neighbor count must be positive.")
    a = float(umap_params["a"])
    b = float(umap_params["b"])
    model = UMAP(
        n_neighbors=n_neighbors,
        n_components=ref_embedding.shape[1],
        metric=neighbor_params.get("metric", "euclidean"),
        metric_kwds=neighbor_params.get("metric_kwds") or None,
        n_epochs=umap_params.get("transform_epochs"),
        a=a,
        b=b,
        random_state=umap_params.get("random_state", 0),
        output_type="cupy",
    )
    model._a = a
    model._b = b
    model._n_neighbors = n_neighbors
    model._raw_data = (
        SparseCumlArray(ref_rep) if ref_is_sparse else CumlArray(data=ref_rep)
    )
    model._sparse_data = ref_is_sparse
    model._supervised = False
    model._input_hash = None
    model.n_features_in_ = ref_rep.shape[1]
    model.embedding_ = CumlArray(data=ref_embedding)

    return cp.asarray(model.transform(query_rep), dtype=cp.float32)
