from __future__ import annotations

import inspect
import warnings
from functools import partial
from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
import pandas as pd
from anndata import AnnData
from cupyx.scipy import sparse as sparse_gpu
from scverse_misc import Deprecation, deprecated

import rapids_singlecell as rsc

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


__all__ = [
    "calculate_niche",
    "calculate_niche_cellcharter",
    "calculate_niche_neighborhood",
    "calculate_niche_utag",
]

FLAVORS = ("neighborhood", "utag", "cellcharter")
GMM_INIT = "random_from_data"
NOT_A_NICHE = "not_a_niche"


def _normalize_resolutions(
    resolutions: float | Sequence[float],
) -> tuple[float, ...]:
    values = (
        (float(resolutions),)
        if isinstance(resolutions, (int, float))
        else tuple(float(resolution) for resolution in resolutions)
    )
    if not values:
        raise ValueError("`resolutions` must contain at least one value.")
    if pd.Index(values).has_duplicates:
        raise ValueError("`resolutions` must contain unique values.")
    return values


def calculate_niche_neighborhood(
    adata: AnnData,
    *,
    groups: str,
    resolutions: float | Sequence[float],
    n_neighbors: int = 15,
    spatial_connectivities_key: str = "spatial_connectivities",
    scale: bool = True,
    distance: int = 1,
    abs_nhood: bool = False,
    n_hop_weights: Sequence[float] | None = None,
    min_niche_size: int | None = None,
    mask: pd.Series | None = None,
    library_key: str | None = None,
    inplace: bool = True,
) -> AnnData | None:
    """\
    Compute spatial niches from cell-type neighborhood profiles on the GPU.

    Each cell is described by the frequency of ``groups`` labels among its spatial
    neighbors; the profiles are then clustered with leiden. Mirrors
    ``squidpy.gr.calculate_niche_neighborhood`` :cite:p:`monkeybread`.

    Labels are written to ``adata.obs['nhood_niche_res=<res>']``.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groups
        Column in ``adata.obs`` with cell-type labels.
    resolutions
        Resolution(s) for leiden. A label column is written for each value.
    n_neighbors
        Neighbors for the post-aggregation kNN graph passed to leiden.
    spatial_connectivities_key
        Key in ``adata.obsp`` with the spatial connectivity matrix.
    scale
        Z-score the neighborhood profile before clustering.
    distance
        Number of n-hop neighborhoods to include.
    abs_nhood
        Use absolute neighbor counts instead of per-cell relative frequencies.
    n_hop_weights
        Per-hop weights when ``distance > 1``.
    min_niche_size
        Discard niches with fewer cells than this; relabel as ``"not_a_niche"``.
    mask
        Boolean :class:`~pandas.Series` indexed like ``adata.obs``. Cells that are
        ``False`` are labeled ``"not_a_niche"``.
    library_key
        Column in ``adata.obs`` identifying samples. If given, niches are computed
        per sample and labels are prefixed with ``lib=<id>_``.
    inplace
        Write the niche columns to ``adata``. If ``False``, return a modified copy.
    """
    resolutions = _normalize_resolutions(resolutions)
    _check_key(adata, spatial_connectivities_key)
    if groups is None:
        raise ValueError("`groups` is required for flavor='neighborhood'.")
    if groups not in adata.obs.columns:
        raise KeyError(f"'{groups}' not found in `adata.obs`.")
    if n_neighbors < 1:
        raise ValueError(f"`n_neighbors` must be >= 1, got {n_neighbors}.")
    if distance < 1:
        raise ValueError(f"`distance` must be >= 1, got {distance}.")

    return _calculate_niche_custom(
        adata,
        partial(
            _nhood_embedding,
            groups=groups,
            distance=distance,
            n_hop_weights=n_hop_weights,
            abs_nhood=abs_nhood,
            scale=scale,
            key=spatial_connectivities_key,
        ),
        partial(
            _leiden_cluster,
            n_neighbors=n_neighbors,
            resolutions=resolutions,
            base_colname="nhood_niche",
        ),
        min_niche_size=min_niche_size,
        mask=mask,
        library_key=library_key,
        inplace=inplace,
    )


def calculate_niche_utag(
    adata: AnnData,
    *,
    resolutions: float | Sequence[float],
    n_neighbors: int = 15,
    spatial_connectivities_key: str = "spatial_connectivities",
    min_niche_size: int | None = None,
    mask: pd.Series | None = None,
    library_key: str | None = None,
    inplace: bool = True,
) -> AnnData | None:
    """\
    Compute spatial niches from UTAG-smoothed expression on the GPU.

    Expression is propagated over the L1-normalized spatial graph, PCA-reduced and
    clustered with leiden. Mirrors ``squidpy.gr.calculate_niche_utag``
    :cite:p:`UTAG2022`.

    Labels are written to ``adata.obs['utag_niche_res=<res>']``.

    Parameters
    ----------
    adata
        Annotated data matrix.
    resolutions
        Resolution(s) for leiden. A label column is written for each value.
    n_neighbors
        Neighbors for the post-aggregation kNN graph passed to leiden.
    spatial_connectivities_key
        Key in ``adata.obsp`` with the spatial connectivity matrix.
    min_niche_size
        Discard niches with fewer cells than this; relabel as ``"not_a_niche"``.
    mask
        Boolean :class:`~pandas.Series` indexed like ``adata.obs``. Cells that are
        ``False`` are labeled ``"not_a_niche"``.
    library_key
        Column in ``adata.obs`` identifying samples. If given, niches are computed
        per sample and labels are prefixed with ``lib=<id>_``.
    inplace
        Write the niche columns to ``adata``. If ``False``, return a modified copy.
    """
    resolutions = _normalize_resolutions(resolutions)
    _check_key(adata, spatial_connectivities_key)
    if n_neighbors < 1:
        raise ValueError(f"`n_neighbors` must be >= 1, got {n_neighbors}.")

    return _calculate_niche_custom(
        adata,
        partial(_utag_embedding, key=spatial_connectivities_key),
        partial(
            _leiden_cluster,
            n_neighbors=n_neighbors,
            resolutions=resolutions,
            base_colname="utag_niche",
        ),
        min_niche_size=min_niche_size,
        mask=mask,
        library_key=library_key,
        inplace=inplace,
    )


def calculate_niche_cellcharter(
    adata: AnnData,
    *,
    distance: int = 3,
    aggregation: Literal["mean", "variance"] = "mean",
    random_state: int = 42,
    spatial_connectivities_key: str = "spatial_connectivities",
    n_components: int = 10,
    use_rep: str | None = None,
    min_niche_size: int | None = None,
    mask: pd.Series | None = None,
    library_key: str | None = None,
    inplace: bool = True,
) -> AnnData | None:
    """\
    Compute spatial niches with the CellCharter approach on the GPU.

    Expression is shell-aggregated over n-hop neighborhoods, PCA-reduced and clustered
    with a Gaussian mixture. Mirrors ``squidpy.gr.calculate_niche_cellcharter``
    :cite:p:`CellCharter2024`.

    Labels are written to ``adata.obs['cellcharter_niche']``.

    Parameters
    ----------
    adata
        Annotated data matrix.
    distance
        Number of n-hop neighborhoods to include.
    aggregation
        Per-shell aggregation. ``"mean"`` (default) or ``"variance"``.
    random_state
        Random seed for the GMM.
    spatial_connectivities_key
        Key in ``adata.obsp`` with the spatial connectivity matrix. Not used when
        ``use_rep`` is provided.
    n_components
        Number of mixture components.
    use_rep
        Key in ``adata.obsm`` to use as the embedding; if provided, the first
        ``n_components`` columns are used and the shell-aggregation + PCA step is
        skipped.
    min_niche_size
        Discard niches with fewer cells than this; relabel as ``"not_a_niche"``.
    mask
        Boolean :class:`~pandas.Series` indexed like ``adata.obs``. Cells that are
        ``False`` are labeled ``"not_a_niche"``.
    library_key
        Column in ``adata.obs`` identifying samples. If given, niches are computed
        per sample and labels are prefixed with ``lib=<id>_``.
    inplace
        Write the niche columns to ``adata``. If ``False``, return a modified copy.
    """
    if use_rep is None:
        _check_key(adata, spatial_connectivities_key)
    if distance < 0:
        raise ValueError(f"`distance` must be >= 0, got {distance}.")
    if aggregation not in ("mean", "variance"):
        raise ValueError(
            f"aggregation={aggregation!r} not supported. Use 'mean' or 'variance'."
        )
    if not isinstance(n_components, int) or n_components < 1:
        raise ValueError(f"`n_components` must be an int >= 1, got {n_components}.")
    if use_rep is not None:
        if use_rep not in adata.obsm:
            raise KeyError(f"'{use_rep}' not found in `adata.obsm`.")
        if adata.obsm[use_rep].shape[1] < n_components:
            raise ValueError(
                f"`adata.obsm['{use_rep}']` has {adata.obsm[use_rep].shape[1]} columns, "
                f"need at least n_components={n_components}."
            )

    return _calculate_niche_custom(
        adata,
        partial(
            _cellcharter_embedding,
            distance=distance,
            aggregation=aggregation,
            n_components=n_components,
            use_rep=use_rep,
            key=spatial_connectivities_key,
        ),
        partial(
            _gmm_cluster,
            n_components=n_components,
            random_state=random_state,
            base_colname="cellcharter_niche",
        ),
        min_niche_size=min_niche_size,
        mask=mask,
        library_key=library_key,
        inplace=inplace,
    )


@deprecated(
    Deprecation(
        "0.17.0",
        "Use `calculate_niche_neighborhood`, `calculate_niche_utag`, or "
        "`calculate_niche_cellcharter` instead.",
    )
)
def calculate_niche(
    adata: AnnData,
    *,
    flavor: Literal["neighborhood", "utag", "cellcharter"],
    groups: str | None = None,
    n_neighbors: int = 15,
    resolutions: float | Sequence[float] | None = None,
    distance: int | None = None,
    n_hop_weights: Sequence[float] | None = None,
    abs_nhood: bool = False,
    scale: bool = True,
    min_niche_size: int | None = None,
    mask: pd.Series | None = None,
    library_key: str | None = None,
    aggregation: Literal["mean", "variance"] = "mean",
    n_components: int = 10,
    use_rep: str | None = None,
    spatial_connectivities_key: str = "spatial_connectivities",
    random_state: int = 42,
    inplace: bool = True,
    copy: bool | None = None,
    **kwargs,
) -> AnnData | None:
    """\
    Compute spatial niches on the GPU.

    .. deprecated:: 0.17.0
        ``calculate_niche`` is deprecated and will be removed in a future release,
        following :mod:`squidpy`. Use the flavor-specific functions instead:

        - :func:`~rapids_singlecell.gr.calculate_niche_neighborhood`
        - :func:`~rapids_singlecell.gr.calculate_niche_utag`
        - :func:`~rapids_singlecell.gr.calculate_niche_cellcharter`

    The spatial graph in ``adata.obsp[spatial_connectivities_key]`` must be
    precomputed (e.g. via :func:`squidpy.gr.spatial_neighbors`), except for
    ``flavor="cellcharter"`` when ``use_rep`` is provided.

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
        Resolution(s) for leiden, defaulting to ``(0.5,)``. A label column is written
        for each value. Ignored for ``flavor="cellcharter"``.
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
    mask
        Boolean :class:`~pandas.Series` indexed like ``adata.obs``. Cells that are
        ``False`` are labeled ``"not_a_niche"``.
    library_key
        Column in ``adata.obs`` identifying samples. If given, niches are computed
        per sample and labels are prefixed with ``lib=<id>_``.
    aggregation
        Per-shell aggregation for ``flavor="cellcharter"``. ``"mean"`` (default) or ``"variance"``.
    n_components
        Number of mixture components for ``flavor="cellcharter"``.
    use_rep
        Key in ``adata.obsm`` to use as the embedding for ``flavor="cellcharter"``;
        if provided, the first ``n_components`` columns are used and the shell-aggregation
        + PCA step is skipped.
    spatial_connectivities_key
        Key in ``adata.obsp`` with the spatial connectivity matrix. Not used for
        ``flavor="cellcharter"`` when ``use_rep`` is provided.
    random_state
        Random seed for the GMM (``flavor="cellcharter"`` only).
    inplace
        Write the niche columns to ``adata``. If ``False``, return a modified copy.
    copy
        Deprecated alias for ``inplace``; ``copy=True`` is ``inplace=False``.
    kwargs
        Accepts the removed ``gmm_init`` argument, which is ignored with a warning.
    """
    if flavor not in FLAVORS:
        raise ValueError(
            f"Unknown flavor '{flavor}'. Use 'neighborhood', 'utag', or 'cellcharter'."
        )
    if kwargs.pop("gmm_init", None) is not None:
        warnings.warn(
            f"`gmm_init` is no longer supported and is ignored; the CellCharter GMM "
            f"always uses {GMM_INIT!r}.",
            FutureWarning,
            stacklevel=3,
        )
    if kwargs:
        raise TypeError(
            f"calculate_niche() got an unexpected keyword argument "
            f"{next(iter(kwargs))!r}."
        )
    if copy is not None:
        warnings.warn(
            "`copy` is deprecated, use `inplace` instead.",
            FutureWarning,
            stacklevel=3,
        )
        inplace = not copy
    _check_unnecessary_args(
        flavor,
        {
            "groups": groups,
            "n_neighbors": n_neighbors,
            "resolutions": resolutions,
            "distance": distance,
            "n_hop_weights": n_hop_weights,
            "abs_nhood": abs_nhood,
            "scale": scale,
            "aggregation": aggregation,
            "n_components": n_components,
            "use_rep": use_rep,
            "random_state": random_state,
        },
    )
    if distance is None:
        distance = 3 if flavor == "cellcharter" else 1
    if resolutions is None:
        resolutions = (0.5,)

    shared = {
        "spatial_connectivities_key": spatial_connectivities_key,
        "min_niche_size": min_niche_size,
        "mask": mask,
        "library_key": library_key,
        "inplace": inplace,
    }
    if flavor == "neighborhood":
        return calculate_niche_neighborhood(
            adata,
            groups=groups,
            resolutions=resolutions,
            n_neighbors=n_neighbors,
            scale=scale,
            distance=distance,
            abs_nhood=abs_nhood,
            n_hop_weights=n_hop_weights,
            **shared,
        )
    if flavor == "utag":
        return calculate_niche_utag(
            adata,
            resolutions=resolutions,
            n_neighbors=n_neighbors,
            **shared,
        )
    return calculate_niche_cellcharter(
        adata,
        distance=distance,
        aggregation=aggregation,
        random_state=random_state,
        n_components=n_components,
        use_rep=use_rep,
        **shared,
    )


UNUSED_ARGS = {
    "neighborhood": (
        "aggregation",
        "n_components",
        "use_rep",
        "random_state",
    ),
    "utag": (
        "groups",
        "distance",
        "n_hop_weights",
        "abs_nhood",
        "scale",
        "aggregation",
        "n_components",
        "use_rep",
        "random_state",
    ),
    "cellcharter": (
        "groups",
        "n_neighbors",
        "resolutions",
        "n_hop_weights",
        "abs_nhood",
        "scale",
    ),
}


def _check_unnecessary_args(flavor: str, params: dict[str, object]) -> None:
    """Warn about arguments the chosen flavor ignores, following :mod:`squidpy`."""
    defaults = inspect.signature(calculate_niche).parameters
    unnecessary = [
        name
        for name in UNUSED_ARGS[flavor]
        if params[name] is not None and params[name] != defaults[name].default
    ]
    if unnecessary:
        warnings.warn(
            f"Parameters {', '.join(unnecessary)} are not used for flavor '{flavor}'.",
            UserWarning,
            stacklevel=3,
        )


def _check_key(adata: AnnData, key: str) -> None:
    if key not in adata.obsp:
        raise KeyError(
            f"'{key}' not found in `adata.obsp`. "
            "Compute it first with `squidpy.gr.spatial_neighbors`."
        )


def _calculate_niche_custom(
    adata: AnnData,
    embed: Callable[[AnnData], cp.ndarray],
    cluster: Callable[..., list[str]],
    *,
    min_niche_size: int | None,
    mask: pd.Series | None,
    library_key: str | None,
    inplace: bool,
) -> AnnData | None:
    """Run embed → cluster → postprocess, optionally stratified by ``library_key``."""
    if adata.n_obs == 0:
        raise ValueError(
            "Cannot calculate niches on an AnnData object with no observations."
        )

    if library_key is not None:
        if library_key not in adata.obs.columns:
            raise KeyError(f"'{library_key}' not found in `adata.obs`.")
        n_missing = int(adata.obs[library_key].isna().sum())
        if n_missing:
            raise ValueError(
                f"`adata.obs[{library_key!r}]` contains {n_missing} missing "
                "library value(s)."
            )

    adata = adata if inplace else adata.copy()

    if library_key is None:
        embedding = None if adata.n_obs == 1 else embed(adata)
        cols = cluster(adata, embedding, library_id=None)
        _postprocess_niche_results(
            adata, cols, mask=mask, min_niche_size=min_niche_size, prefix=None
        )
        return None if inplace else adata

    columns: list[str] = []
    results: list[tuple[np.ndarray, pd.DataFrame]] = []
    for lib_id in adata.obs[library_key].unique():
        lib_mask = (adata.obs[library_key] == lib_id).to_numpy()
        lib_adata = adata[lib_mask].copy()
        embedding = None if lib_adata.n_obs == 1 else embed(lib_adata)
        cols = cluster(lib_adata, embedding, library_id=lib_id)
        _postprocess_niche_results(
            lib_adata,
            cols,
            mask=mask,
            min_niche_size=min_niche_size,
            prefix=f"lib={lib_id}_",
        )
        results.append((lib_mask, lib_adata.obs[cols].astype(str)))
        columns.extend(c for c in cols if c not in columns)

    # Build each column from scratch so cells outside any processed library are
    # 'not_a_niche' rather than stale values from an earlier call.
    for col in columns:
        values = np.full(adata.n_obs, NOT_A_NICHE, dtype=object)
        for lib_mask, lib_obs in results:
            if col in lib_obs.columns:
                values[lib_mask] = lib_obs[col].to_numpy()
        adata.obs[col] = pd.Categorical(values)

    return None if inplace else adata


def _postprocess_niche_results(
    adata: AnnData,
    result_columns: Sequence[str],
    *,
    mask: pd.Series | None,
    min_niche_size: int | None,
    prefix: str | None,
) -> None:
    """Apply ``mask``, ``min_niche_size`` and the library ``prefix`` to niche labels."""
    if mask is None and min_niche_size is None and prefix is None:
        return

    for col in result_columns:
        labels = adata.obs[col].astype(str)
        if mask is not None:
            keep = mask.reindex(adata.obs_names, fill_value=True).to_numpy(dtype=bool)
            labels = labels.where(keep, other=NOT_A_NICHE)
        if min_niche_size is not None:
            counts = labels.value_counts()
            small = counts[counts < min_niche_size].index
            labels = labels.where(~labels.isin(small), other=NOT_A_NICHE)
        if prefix is not None:
            labels = labels.where(labels == NOT_A_NICHE, other=prefix + labels)
        adata.obs[col] = pd.Categorical(labels.values)


def _leiden_cluster(
    adata: AnnData,
    embedding: cp.ndarray | None,
    *,
    n_neighbors: int,
    resolutions: Sequence[float],
    base_colname: str,
    library_id: object | None,
) -> list[str]:
    """kNN graph + leiden over the embedding; one ``<base_colname>_res=<res>`` per resolution."""
    res_list = list(resolutions)
    result_columns = [f"{base_colname}_res={res}" for res in res_list]
    if adata.n_obs == 1:
        if n_neighbors > 1:
            _warn_size_adaptation(
                parameter="n_neighbors",
                requested=n_neighbors,
                effective=1,
                n_obs=adata.n_obs,
                library_id=library_id,
            )
        for out_key in result_columns:
            adata.obs[out_key] = pd.Categorical(["0"])
        return result_columns

    effective_n_neighbors = n_neighbors
    if n_neighbors > adata.n_obs:
        effective_n_neighbors = 1 + int(0.5 * adata.n_obs)
        _warn_size_adaptation(
            parameter="n_neighbors",
            requested=n_neighbors,
            effective=effective_n_neighbors,
            n_obs=adata.n_obs,
            library_id=library_id,
        )

    inner = AnnData(X=embedding, obs=pd.DataFrame(index=adata.obs_names.copy()))
    rsc.pp.neighbors(inner, n_neighbors=effective_n_neighbors, use_rep="X")

    base = "_niche_leiden"
    rsc.tl.leiden(
        inner,
        resolution=res_list,
        key_added=base,
        dtype=np.float64,
    )

    for res, out_key in zip(res_list, result_columns, strict=True):
        src = f"{base}_{res}" if len(res_list) > 1 else base
        adata.obs[out_key] = pd.Categorical(inner.obs[src].astype(str).values)
    return result_columns


def _gmm_cluster(
    adata: AnnData,
    embedding: cp.ndarray | None,
    *,
    n_components: int,
    random_state: int,
    base_colname: str,
    library_id: object | None,
) -> list[str]:
    """Gaussian-mixture clustering of the embedding, seeded like squidpy's CellCharter."""
    effective_n_components = min(n_components, adata.n_obs)
    if effective_n_components != n_components:
        _warn_size_adaptation(
            parameter="n_components",
            requested=n_components,
            effective=effective_n_components,
            n_obs=adata.n_obs,
            library_id=library_id,
        )
    if adata.n_obs == 1:
        adata.obs[base_colname] = pd.Categorical(["0"])
        return [base_colname]

    from ._gmm import gmm_fit_predict

    labels = gmm_fit_predict(
        embedding,
        n_components=effective_n_components,
        random_state=random_state,
        init=GMM_INIT,
    )
    adata.obs[base_colname] = pd.Categorical(cp.asnumpy(labels).astype(str))
    return [base_colname]


def _warn_size_adaptation(
    *,
    parameter: str,
    requested: int,
    effective: int,
    n_obs: int,
    library_id: object | None,
) -> None:
    scope = "the dataset" if library_id is None else f"library {library_id!r}"
    warnings.warn(
        f"`{parameter}={requested}` exceeds the {n_obs} observation(s) in {scope}; "
        f"using `{parameter}={effective}`.",
        UserWarning,
        stacklevel=3,
    )


def _nhood_embedding(
    adata: AnnData,
    *,
    groups: str,
    distance: int,
    n_hop_weights: Sequence[float] | None,
    abs_nhood: bool,
    scale: bool,
    key: str,
) -> cp.ndarray:
    """Neighborhood profile, optionally z-scored."""
    profile = _neighborhood_profile(
        adata,
        groups=groups,
        distance=distance,
        weights=n_hop_weights,
        abs_nhood=abs_nhood,
        key=key,
    )
    if not scale:
        return profile
    inner = AnnData(X=profile, obs=pd.DataFrame(index=adata.obs_names.copy()))
    rsc.pp.scale(inner, zero_center=True)
    return inner.X


def _utag_embedding(adata: AnnData, *, key: str) -> cp.ndarray:
    """UTAG-smoothed expression, PCA-reduced."""
    inner = AnnData(
        X=_utag_features(adata, key), obs=pd.DataFrame(index=adata.obs_names.copy())
    )
    rsc.pp.pca(inner, key_added=None)
    return cp.asarray(inner.obsm["X_pca"])


def _cellcharter_embedding(
    adata: AnnData,
    *,
    distance: int,
    aggregation: str,
    n_components: int,
    use_rep: str | None,
    key: str,
) -> cp.ndarray:
    """Shell-aggregated expression, PCA-reduced — or the first ``n_components`` of ``use_rep``."""
    if use_rep is not None:
        return cp.asarray(adata.obsm[use_rep][:, :n_components], dtype=cp.float32)

    feat = _cellcharter_features(adata, distance, aggregation, key)
    # Deeper shells can yield all-zero columns when a gene's expression
    # never propagates into that shell across the whole dataset. rsc PCA
    # rejects zero columns; drop them so the embedding still uses the
    # informative dimensions.
    if sparse_gpu.issparse(feat):
        feat_nonzero = feat.tocsc(copy=True)
        feat_nonzero.eliminate_zeros()
        nonzero = cp.flatnonzero(cp.diff(feat_nonzero.indptr))
        if int(nonzero.size) < feat.shape[1]:
            feat = feat[:, nonzero]
    else:
        nonzero = cp.any(feat != 0, axis=0)
        if int(nonzero.sum()) < feat.shape[1]:
            feat = feat[:, nonzero]
    inner = AnnData(X=feat, obs=pd.DataFrame(index=adata.obs_names.copy()))
    rsc.get.anndata_to_GPU(inner)
    rsc.pp.pca(inner, key_added=None)
    return cp.asarray(inner.obsm["X_pca"], dtype=cp.float32)


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
    codes = cp.asarray(cats.codes, dtype=cp.int64)
    valid = codes >= 0
    rows = cp.arange(n_obs, dtype=cp.int64)[valid]
    one_hot[rows, codes[valid]] = 1.0

    adj = rsc.get.X_to_GPU(adata.obsp[key]).copy()
    if adj.dtype != cp.float32:
        adj = adj.astype(cp.float32)
    adj.eliminate_zeros()

    if distance == 1 or weights is None:
        weights = [1.0] * distance
    elif len(weights) < distance:
        weights = list(weights) + [weights[-1]] * (distance - len(weights))
    if not abs_nhood and sum(weights) == 0:
        raise ValueError("`n_hop_weights` must not sum to zero.")

    profile = cp.zeros((n_obs, n_cats), dtype=cp.float32)
    adj_k = adj
    for hop in range(distance):
        if hop > 0:
            adj_k = adj_k @ adj
        counts = adj_k @ one_hot  # (n_obs, n_cats) dense
        if not abs_nhood:
            row_sum = counts.sum(axis=1, keepdims=True)
            row_sum = cp.where(row_sum == 0, cp.float32(1.0), row_sum)
            counts = counts / row_sum
        profile += cp.float32(weights[hop]) * counts

    if not abs_nhood:
        profile /= cp.float32(sum(weights))

    return profile


def _utag_features(adata: AnnData, key: str) -> cp.ndarray | sparse_gpu.csr_matrix:
    """L1-row-normalize the spatial adjacency and propagate expression: D^-1 A @ X."""
    from rapids_singlecell._cuda import _norm_cuda as _nc

    adj = rsc.get.X_to_GPU(adata.obsp[key]).copy()
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
