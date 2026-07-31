from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
import pandas as pd
from anndata import AnnData
from cupyx.scipy import sparse as sparse_gpu

import rapids_singlecell as rsc

if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = ["calculate_niche"]


def calculate_niche(
    adata: AnnData,
    *,
    flavor: Literal["neighborhood", "utag", "cellcharter"],
    groups: str | None = None,
    n_neighbors: int = 15,
    resolutions: float | Sequence[float] = (0.5,),
    distance: int | None = None,
    n_hop_weights: Sequence[float] | None = None,
    abs_nhood: bool = False,
    scale: bool = True,
    min_niche_size: int | None = None,
    aggregation: Literal["mean", "variance"] = "mean",
    n_components: int = 10,
    use_rep: str | None = None,
    gmm_init: Literal[
        "random_from_data", "kmeans", "sklearn_kmeans"
    ] = "random_from_data",
    spatial_connectivities_key: str = "spatial_connectivities",
    random_state: int = 42,
    copy: bool = False,
) -> AnnData | None:
    """\
    Compute spatial niches on the GPU.

    Mirrors :func:`squidpy.gr.calculate_niche` for the ``"neighborhood"``,
    ``"utag"`` and ``"cellcharter"`` flavors. The spatial graph in
    ``adata.obsp[spatial_connectivities_key]`` must be precomputed
    (e.g. via :func:`squidpy.gr.spatial_neighbors`).

    Parameters
    ----------
    adata
        Annotated data matrix.
    flavor
        - ``"neighborhood"`` cluster cell-type frequency profiles among spatial neighbors
          :cite:p:`monkeybread`.
        - ``"utag"`` cluster gene expression smoothed across spatial neighbors
          :cite:p:`UTAG2022`.
        - ``"cellcharter"`` shell-aggregate gene expression over n-hop neighborhoods,
          PCA-reduce, then cluster with a Gaussian mixture :cite:p:`CellCharter2024`.
    groups
        Column in ``adata.obs`` with cell-type labels. Required for ``flavor="neighborhood"``.
    n_neighbors
        Neighbors for the post-aggregation kNN graph passed to leiden.
    resolutions
        Resolution(s) for leiden. A label column is written for each value.
        Ignored for ``flavor="cellcharter"``.
    distance
        Number of n-hop neighborhoods to include. Defaults to 3 for ``cellcharter``,
        1 for ``neighborhood``.
    n_hop_weights
        Per-hop weights when ``distance > 1`` (``flavor="neighborhood"`` only).
    abs_nhood
        Use absolute neighbor counts instead of per-cell relative frequencies
        (``flavor="neighborhood"`` only).
    scale
        Z-score the neighborhood profile before clustering (``flavor="neighborhood"`` only).
    min_niche_size
        Discard niches with fewer cells than this; relabel as ``"not_a_niche"``.
    aggregation
        Per-shell aggregation for ``flavor="cellcharter"``. ``"mean"`` (default) or ``"variance"``.
    n_components
        Number of mixture components for ``flavor="cellcharter"``.
    use_rep
        Key in ``adata.obsm`` to use as the embedding for ``flavor="cellcharter"``;
        if provided, the first ``n_components`` columns are used and the shell-aggregation
        + PCA step is skipped.
    gmm_init
        GMM initialization for ``flavor="cellcharter"``. ``"random_from_data"``
        (default) matches Squidpy's CellCharter path. ``"kmeans"`` uses native
        cuML KMeans. ``"sklearn_kmeans"`` uses sklearn-compatible k-means++ seeding
        followed by cuML KMeans.
    spatial_connectivities_key
        Key in ``adata.obsp`` with the spatial connectivity matrix.
    random_state
        Random seed for leiden / GMM.
    copy
        Return a copy with the niche columns instead of writing in place.
    """
    if spatial_connectivities_key not in adata.obsp:
        raise KeyError(
            f"'{spatial_connectivities_key}' not found in `adata.obsp`. "
            "Compute it first with `squidpy.gr.spatial_neighbors`."
        )
    if flavor not in ("neighborhood", "utag", "cellcharter"):
        raise ValueError(
            f"Unknown flavor '{flavor}'. Use 'neighborhood', 'utag', or 'cellcharter'."
        )
    if distance is None:
        distance = 3 if flavor == "cellcharter" else 1
    if flavor in ("neighborhood",) and distance < 1:
        raise ValueError(f"`distance` must be >= 1, got {distance}.")
    if flavor == "cellcharter" and distance < 0:
        raise ValueError(f"`distance` must be >= 0, got {distance}.")

    adata = adata.copy() if copy else adata

    if flavor == "cellcharter":
        if gmm_init not in ("random_from_data", "kmeans", "sklearn_kmeans"):
            raise ValueError(
                "`gmm_init` must be one of 'random_from_data', 'kmeans', or "
                f"'sklearn_kmeans', got {gmm_init!r}."
            )
        _run_cellcharter(
            adata,
            distance=distance,
            aggregation=aggregation,
            n_components=n_components,
            use_rep=use_rep,
            gmm_init=gmm_init,
            random_state=random_state,
            key=spatial_connectivities_key,
        )
        return adata if copy else None

    if flavor == "neighborhood":
        if groups is None:
            raise ValueError("`groups` is required for flavor='neighborhood'.")
        if groups not in adata.obs.columns:
            raise KeyError(f"'{groups}' not found in `adata.obs`.")
        profile = _neighborhood_profile(
            adata,
            groups=groups,
            distance=distance,
            weights=n_hop_weights,
            abs_nhood=abs_nhood,
            key=spatial_connectivities_key,
        )
        prefix = "nhood_niche"
    else:
        profile = _utag_features(adata, spatial_connectivities_key)
        prefix = "utag_niche"

    inner = AnnData(X=profile, obs=pd.DataFrame(index=adata.obs_names.copy()))

    if flavor == "neighborhood":
        if scale:
            rsc.pp.scale(inner, zero_center=True)
        rsc.pp.neighbors(
            inner, n_neighbors=n_neighbors, use_rep="X", random_state=random_state
        )
    else:
        rsc.pp.pca(inner)
        rsc.pp.neighbors(
            inner, n_neighbors=n_neighbors, use_rep="X_pca", random_state=random_state
        )

    res_list = (
        [float(resolutions)]
        if isinstance(resolutions, (int, float))
        else [float(r) for r in resolutions]
    )
    base = "_niche_leiden"
    rsc.tl.leiden(
        inner,
        resolution=res_list,
        key_added=base,
        random_state=random_state,
        dtype=np.float64,
    )
    for res in res_list:
        src = f"{base}_{res}" if len(res_list) > 1 else base
        out_key = f"{prefix}_res={res}"
        labels = inner.obs[src].astype(str)
        if min_niche_size is not None and flavor == "neighborhood":
            counts = labels.value_counts()
            small = counts[counts < min_niche_size].index
            labels = labels.where(~labels.isin(small), other="not_a_niche")
        adata.obs[out_key] = pd.Categorical(labels.values)

    return adata if copy else None


def _neighborhood_profile(
    adata: AnnData,
    *,
    groups: str,
    distance: int,
    weights: Sequence[float] | None,
    abs_nhood: bool,
    key: str,
) -> cp.ndarray:
    """Cells x categories matrix of cell-type counts (or relative frequencies) over n-hop neighbors."""
    cats = pd.Categorical(adata.obs[groups])
    n_cats = len(cats.categories)
    n_obs = adata.n_obs

    one_hot = cp.zeros((n_obs, n_cats), dtype=cp.float32)
    one_hot[cp.arange(n_obs), cp.asarray(cats.codes, dtype=cp.int64)] = 1.0

    adj = rsc.get.X_to_GPU(adata.obsp[key]).astype(cp.float32)
    adj.eliminate_zeros()
    # Binarize so adj.data == 1: each existing edge contributes one neighbor count.
    adj_bin = adj.copy()
    adj_bin.data[:] = 1.0

    if weights is None:
        weights = [1.0] * distance
    elif len(weights) < distance:
        weights = list(weights) + [weights[-1]] * (distance - len(weights))
    if not abs_nhood and sum(weights) == 0:
        raise ValueError("`n_hop_weights` must not sum to zero.")

    profile = cp.zeros((n_obs, n_cats), dtype=cp.float32)
    adj_k = adj_bin
    for hop in range(distance):
        if hop == 0:
            adj_hop = adj_bin
        else:
            adj_k = adj_k @ adj_bin
            adj_hop = adj_k.copy()
            adj_hop.data[:] = 1.0
        counts = adj_hop @ one_hot  # (n_obs, n_cats) dense
        if not abs_nhood:
            row_sum = adj_hop.sum(axis=1).reshape(-1, 1)
            row_sum = cp.where(row_sum == 0, cp.float32(1.0), row_sum)
            counts = counts / row_sum
        profile += cp.float32(weights[hop]) * counts

    if not abs_nhood:
        profile /= cp.float32(sum(weights))

    return profile


def _utag_features(adata: AnnData, key: str) -> cp.ndarray | sparse_gpu.csr_matrix:
    """L1-row-normalize the spatial adjacency and propagate expression: D^-1 A @ X."""
    from rapids_singlecell._cuda import _norm_cuda as _nc

    adj = rsc.get.X_to_GPU(adata.obsp[key])
    if adj.dtype != cp.float32:
        adj = adj.astype(cp.float32)
    _nc.mul_csr(
        adj.indptr,
        adj.data,
        nrows=adj.shape[0],
        target_sum=1.0,
        stream=cp.cuda.get_current_stream().ptr,
    )

    X = rsc.get.X_to_GPU(adata.X).astype(cp.float32)
    if sparse_gpu.issparse(X):
        out = adj @ X
        return out.tocsr()
    out = adj @ X
    return out


def _run_cellcharter(
    adata: AnnData,
    *,
    distance: int,
    aggregation: str,
    n_components: int,
    use_rep: str | None,
    gmm_init: str,
    random_state: int,
    key: str,
) -> None:
    """Cellcharter pipeline: shell-aggregate → PCA → GMM."""
    if aggregation not in ("mean", "variance"):
        raise ValueError(
            f"aggregation={aggregation!r} not supported. Use 'mean' or 'variance'."
        )
    if not isinstance(n_components, int) or n_components < 1:
        raise ValueError(f"`n_components` must be an int >= 1, got {n_components}.")

    if use_rep is not None:
        if use_rep not in adata.obsm:
            raise KeyError(f"'{use_rep}' not found in `adata.obsm`.")
        emb = adata.obsm[use_rep]
        if emb.shape[1] < n_components:
            raise ValueError(
                f"`adata.obsm['{use_rep}']` has {emb.shape[1]} columns, "
                f"need at least n_components={n_components}."
            )
        embedding = cp.asarray(emb[:, :n_components], dtype=cp.float32)
    else:
        feat = _cellcharter_features(adata, distance, aggregation, key)
        # Deeper shells can yield all-zero columns when a gene's expression
        # never propagates into that shell across the whole dataset. rsc PCA
        # rejects zero columns; drop them so the embedding still uses the
        # informative dimensions.
        if sparse_gpu.issparse(feat):
            col_sum = cp.asarray(feat.sum(axis=0)).ravel()
            nonzero = cp.where(col_sum != 0)[0]
            if int(nonzero.size) < feat.shape[1]:
                feat = feat[:, nonzero]
        else:
            col_sum = feat.sum(axis=0)
            nonzero = col_sum != 0
            if int(nonzero.sum()) < feat.shape[1]:
                feat = feat[:, nonzero]
        inner = AnnData(X=feat, obs=pd.DataFrame(index=adata.obs_names.copy()))
        rsc.get.anndata_to_GPU(inner)
        rsc.pp.pca(inner)
        embedding = cp.asarray(inner.obsm["X_pca"], dtype=cp.float32)

    from ._gmm import gmm_fit_predict

    labels = gmm_fit_predict(
        embedding,
        n_components=n_components,
        random_state=random_state,
        init=gmm_init,
    )
    adata.obs["cellcharter_niche"] = pd.Categorical(cp.asnumpy(labels).astype(str))


def _cellcharter_features(
    adata: AnnData,
    distance: int,
    aggregation: str,
    key: str,
) -> cp.ndarray | sparse_gpu.csr_matrix:
    """Build the shell-aggregated feature matrix: ``[X | Â₁X | Â₂X | …]``.

    For each k in ``1..distance`` the kth-shell adjacency is computed by
    multiplying the previous adjacency by the base graph and subtracting the
    already-visited neighbors. Each shell is row-L1-normalized via the same
    fused ``mul_csr`` kernel used for utag, then aggregated as either:

    - ``"mean"``: ``Âₖ @ X``
    - ``"variance"``: ``Âₖ @ (X·X) - (Âₖ @ X)²``  (matches squidpy's path; densifies X)

    All layers are concatenated horizontally.
    """
    from rapids_singlecell._cuda import _norm_cuda as _nc

    adj = rsc.get.X_to_GPU(adata.obsp[key])
    if adj.dtype != cp.float32:
        adj = adj.astype(cp.float32)

    # 1-hop adjacency, no self-loops; visited tracks {self ∪ 1-hop}.
    adj_hop = adj.copy()
    adj_hop.setdiag(cp.float32(0.0))
    adj_hop.eliminate_zeros()
    adj_visited = adj.copy()
    adj_visited.setdiag(cp.float32(1.0))

    X = rsc.get.X_to_GPU(adata.X)
    if aggregation == "variance":
        # Variance needs element-wise square of X; densify once up front.
        X_dense = X.toarray() if sparse_gpu.issparse(X) else X
        X_sq = X_dense * X_dense
        aggregated: list = [X_dense]
    else:
        aggregated = [X]

    for k in range(1, distance + 1):
        if k > 1:
            # Walk one more hop, keep only newly reachable neighbors.
            adj_hop = adj_hop @ adj
            new_shell = (adj_hop > adj_visited).astype(cp.float32)
            adj_hop = new_shell
            adj_visited = adj_visited + new_shell

        # L1 row-normalize the shell adjacency in place.
        adj_norm = adj_hop.copy()
        if adj_norm.nnz > 0:
            _nc.mul_csr(
                adj_norm.indptr,
                adj_norm.data,
                nrows=adj_norm.shape[0],
                target_sum=1.0,
                stream=cp.cuda.get_current_stream().ptr,
            )

        if aggregation == "variance":
            mean = adj_norm @ X_dense
            mean_sq = adj_norm @ X_sq
            aggregated.append(mean_sq - mean * mean)
        else:
            aggregated.append(adj_norm @ X)

    if all(not sparse_gpu.issparse(m) for m in aggregated):
        return cp.concatenate(aggregated, axis=1)
    aggregated = [
        m if sparse_gpu.issparse(m) else sparse_gpu.csr_matrix(m) for m in aggregated
    ]
    return sparse_gpu.hstack(aggregated, format="csr")
