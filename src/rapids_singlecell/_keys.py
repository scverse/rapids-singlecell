from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, assert_never, cast, overload

from ._settings import BasicEmbeddingPreset, Default, Preset, settings

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = [
    "_Embedding",
    "_EmbeddingKeys",
    "_PcaKeys",
    "_embedding_keys",
    "_existing_preset_keys",
    "_harmony_obsm_key",
    "_keys_for_obsm",
    "_preset_harmony_names",
    "_preset_keys",
    "_preset_obsm_names",
    "_resolve_obsm_key",
]

type _Embedding = Literal["pca", "tsne", "umap", "draw_graph", "diffmap"]


@dataclass(frozen=True)
class _EmbeddingKeys:
    """The keys an embedding is stored under."""

    uns: str
    obsm: str


@dataclass(frozen=True)
class _PcaKeys(_EmbeddingKeys):
    varm: str


@overload
def _embedding_keys(
    embedding: Literal["pca"], key_added: str | Default | Preset | None = ...
) -> _PcaKeys: ...
@overload
def _embedding_keys(
    embedding: Literal["tsne", "umap", "diffmap"],
    key_added: str | Default | Preset | None = ...,
) -> _EmbeddingKeys: ...
@overload
def _embedding_keys(
    embedding: Literal["draw_graph"],
    key_added: str | Default | Preset | None = ...,
    *,
    layout: str,
) -> _EmbeddingKeys: ...
def _embedding_keys(
    embedding: _Embedding,
    key_added: str | Default | Preset | None = Default(),
    *,
    layout: str = "",
) -> _EmbeddingKeys:
    """Return the keys `embedding` is stored under for a given `key_added`.

    `key_added=None` gives the scanpy 1 names, a string gives that name, and a
    :class:`~rapids_singlecell._settings.Preset` (or the default marker, meaning
    the active preset) resolves `key_added` from that preset first.
    """
    if isinstance(key_added, Default):
        key_added = settings.preset
    if isinstance(key_added, Preset):
        key_added = cast(
            "BasicEmbeddingPreset", getattr(key_added, embedding)
        ).key_added
    match embedding, key_added:
        case "draw_graph", k:
            return _draw_graph_keys(k, layout=layout)
        case "pca", None:
            return _PcaKeys("pca", "X_pca", "PCs")
        case "pca", str():
            return _PcaKeys(key_added, key_added, key_added)
        case "diffmap", None:
            return _EmbeddingKeys("diffmap_evals", "X_diffmap")
        case "diffmap", str():
            return _EmbeddingKeys(key_added, key_added)
        case "umap" | "tsne" as e, k:
            return _EmbeddingKeys(k or e, k or f"X_{e}")
        case _:  # pragma: no cover
            assert_never(embedding)


def _draw_graph_keys(key_added: str | None, *, layout: str) -> _EmbeddingKeys:
    if key_added is None:
        return _EmbeddingKeys("draw_graph", f"X_draw_graph_{layout}")
    formatted = key_added.format(layout=layout)
    return _EmbeddingKeys(formatted, formatted)


def _preset_keys(embedding: _Embedding, *, layout: str = "") -> list[_EmbeddingKeys]:
    """The keys `embedding` uses under each preset, oldest preset first."""
    return [_embedding_keys(embedding, p, layout=layout) for p in Preset]  # type: ignore[call-overload]


def _preset_obsm_names(embedding: _Embedding, *, layout: str = "") -> set[str]:
    """Every `.obsm` name `embedding` may use, across all presets."""
    return {keys.obsm for keys in _preset_keys(embedding, layout=layout)}


def _keys_for_obsm(
    embedding: _Embedding, obsm: str, *, layout: str = ""
) -> _EmbeddingKeys | None:
    """Reverse `_embedding_keys`: the keys whose `.obsm` name is `obsm`.

    `obsm` is a stored key name, not a `key_added` — `'X_pca'` maps back to the
    scanpy 1 triple (`varm='PCs'`), not to `varm='X_pca'`.
    """
    return next(
        (keys for keys in _preset_keys(embedding, layout=layout) if keys.obsm == obsm),
        None,
    )


def _harmony_obsm_key(basis: str) -> str:
    """The `.obsm` key harmony writes when it corrects `basis`.

    Harmony has no name of its own: it suffixes whatever basis it corrected, so
    it follows the preset through `basis` (`'X_pca'` -> `'X_pca_harmony'`,
    `'pca'` -> `'pca_harmony'`) without naming either spelling here.
    """
    return f"{basis}_harmony"


def _preset_harmony_names() -> set[str]:
    """Every `.obsm` name harmony may write for a preset PCA key."""
    return {_harmony_obsm_key(name) for name in _preset_obsm_names("pca")}


def _existing_preset_keys(
    adata: AnnData, embedding: _Embedding, *, layout: str = ""
) -> _EmbeddingKeys | None:
    """Return the keys `embedding` is actually stored under, or `None` if absent."""
    for keys in _preset_keys(embedding, layout=layout):
        if keys.obsm in adata.obsm:
            return keys
    return None


def _resolve_obsm_key(
    adata: AnnData,
    key: str | None,
    embedding: _Embedding = "pca",
    *,
    layout: str = "",
) -> str | None:
    """Resolve a requested `.obsm` key against the naming both presets may have used.

    A `key` that is not one of `embedding`'s preset names is returned unchanged, so
    explicit keys such as `'X_scVI'` always win. A preset name that is absent falls
    back to the other preset's spelling, and `key=None` auto-detects.
    """
    if key is not None and key not in _preset_obsm_names(embedding, layout=layout):
        return key
    existing = _existing_preset_keys(adata, embedding, layout=layout)
    return key if existing is None else existing.obsm
