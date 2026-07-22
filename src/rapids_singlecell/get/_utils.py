from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anndata import AnnData


def _get_obs_rep(
    adata: AnnData,
    *,
    use_raw: bool = False,
    layer: str | None = None,
    obsm: str | None = None,
    obsp: str | None = None,
) -> Any:
    """Return the selected observation-aligned representation."""
    if not isinstance(use_raw, bool):
        msg = f"use_raw expected to be bool, was {type(use_raw)}."
        raise TypeError(msg)

    choices = {
        "layer": layer,
        "use_raw": use_raw,
        "obsm": obsm,
        "obsp": obsp,
    }
    selected = [
        (key, value)
        for key, value in choices.items()
        if value is not None and value is not False
    ]

    match selected:
        case []:
            return adata.X
        case [("layer", key)]:
            return adata.layers[key]
        case [("use_raw", True)]:
            return adata.raw.X
        case [("obsm", key)]:
            return adata.obsm[key]
        case [("obsp", key)]:
            return adata.obsp[key]
        case _:
            names = [f"`{key}`" for key, _ in selected]
            names[-1] = f"or {names[-1]}"
            msg = f"Only one of {', '.join(names)} can be specified."
            raise ValueError(msg)


def _set_obs_rep(
    adata: AnnData,
    val: Any,
    *,
    use_raw: bool = False,
    layer: str | None = None,
    obsm: str | None = None,
    obsp: str | None = None,
) -> None:
    """Set the selected observation-aligned representation."""
    is_layer = layer is not None
    is_raw = use_raw is not False
    is_obsm = obsm is not None
    is_obsp = obsp is not None
    choices_made = sum((is_layer, is_raw, is_obsm, is_obsp))
    assert choices_made <= 1

    if choices_made == 0:
        adata.X = val
    elif is_layer:
        adata.layers[layer] = val
    elif use_raw:
        adata.raw.X = val
    elif is_obsm:
        adata.obsm[obsm] = val
    elif is_obsp:
        adata.obsp[obsp] = val
    else:
        msg = "Unexpected observation representation selection."
        raise AssertionError(msg)
