from __future__ import annotations

from collections.abc import Collection
from importlib.util import find_spec
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, Literal, overload

import cupy as cp
import numpy as np
import pandas as pd
from anndata import AnnData
from cupyx.scipy import sparse as cp_sparse

from rapids_singlecell._compat import DaskArray
from rapids_singlecell._settings import Preset, settings

if TYPE_CHECKING or find_spec("anndata.acc") is not None:
    from anndata.acc import A, AdRef, GraphAcc, Idx2D, LayerAcc, MultiAcc
else:
    A = None
    AdRef = type("AdRef", (), {"__module__": "anndata.acc"})
    GraphAcc = type("GraphAcc", (), {"__module__": "anndata.acc"})
    Idx2D = object
    LayerAcc = type("LayerAcc", (), {"__module__": "anndata.acc"})
    MultiAcc = type("MultiAcc", (), {"__module__": "anndata.acc"})

type ArrAcc = GraphAcc | LayerAcc | MultiAcc


def _collection_of(thing: object, typ: type | tuple[type, ...]) -> bool:
    return (
        isinstance(thing, Collection)
        and not isinstance(thing, typ)
        and len(thing) > 0
        and all(isinstance(element, typ) for element in thing)
    )


def _require_anndata_acc():
    """Return the accessor namespace or explain the optional version requirement."""
    if A is None:
        raise ImportError(
            "AnnData accessor references require a version of `anndata` that "
            "provides `anndata.acc`."
        )
    return A


def _resolve_vector_acc(spec: str, *, strict: bool):
    """Resolve a vector string across released and development AnnData APIs."""
    resolve = _require_anndata_acc().resolve
    kwargs = {"strict": strict}
    if "vec" in signature(resolve).parameters:
        kwargs["vec"] = True
    return resolve(spec, **kwargs)


def _resolve_matrix_acc(spec: str) -> ArrAcc:
    """Resolve a whole-array accessor string across AnnData accessor versions."""
    acc_api = _require_anndata_acc()
    if "vec" in signature(acc_api.resolve).parameters:
        return acc_api.resolve(spec, vec=False)
    if spec == "X":
        return acc_api.X
    if "." in spec:
        name, key = spec.split(".", 1)
        mapping = getattr(acc_api, name, None)
        if name in {"layers", "obsm", "varm", "obsp", "varp"} and mapping is not None:
            return mapping[key]
    raise ValueError(
        f"Cannot parse matrix accessor {spec!r}. Expected `X`, "
        "`layers.<key>`, `obsm.<key>`, `varm.<key>`, `obsp.<key>`, "
        "or `varp.<key>`."
    )


def _get_accessor_array(adata: AnnData, acc: ArrAcc) -> Any:
    """Retrieve a complete accessor array across AnnData accessor versions."""
    parameters = tuple(signature(acc.get).parameters.values())
    requires_index = len(parameters) > 1 and parameters[1].default is Parameter.empty
    if not requires_index:
        return acc.get(adata)
    if isinstance(acc, LayerAcc | GraphAcc):
        return acc.get(adata, (slice(None), slice(None)))
    return getattr(adata, f"{acc.dim}m")[acc.k]


@overload
def _get_arr(
    adata: AnnData,
    acc: Collection[ArrAcc | str],
    *,
    dim: Literal["obs", "var"] | None = None,
) -> list[Any]: ...


@overload
def _get_arr(
    adata: AnnData,
    acc: ArrAcc | str | None = None,
    *,
    dim: Literal["obs", "var"] | None = None,
    use_raw: bool = False,
    layer: str | None = None,
    obsm: str | None = None,
    obsp: str | None = None,
    varm: str | None = None,
    varp: str | None = None,
) -> Any: ...


def _get_arr(
    adata: AnnData,
    acc: ArrAcc | str | Collection[ArrAcc | str] | None = None,
    *,
    dim: Literal["obs", "var"] | None = None,
    use_raw: bool = False,
    layer: str | None = None,
    obsm: str | None = None,
    obsp: str | None = None,
    varm: str | None = None,
    varp: str | None = None,
) -> Any:
    """Return a 2D AnnData array aligned with ``dim``."""
    if _collection_of(acc, (GraphAcc, LayerAcc, MultiAcc, str)):
        return [
            _get_arr(
                adata,
                item,
                dim=dim,
                use_raw=use_raw,
                layer=layer,
                obsm=obsm,
                obsp=obsp,
                varm=varm,
                varp=varp,
            )
            for item in acc
        ]

    if not isinstance(use_raw, bool):
        msg = f"use_raw expected to be bool, was {type(use_raw)}."
        raise TypeError(msg)

    choices = {
        "layer": layer,
        "use_raw": use_raw,
        "obsm": obsm,
        "obsp": obsp,
        "varm": varm,
        "varp": varp,
    }
    selected = [
        (key, value)
        for key, value in choices.items()
        if value is not None and value is not False
    ]

    if acc is not None:
        if selected:
            raise TypeError(
                "`acc` cannot be combined with `layer`/`use_raw`/`obsm`/"
                "`obsp`/`varm`/`varp`."
            )
        if isinstance(acc, str):
            acc = _resolve_matrix_acc(acc)
        if not isinstance(acc, GraphAcc | LayerAcc | MultiAcc):
            raise TypeError(
                "`acc` must be a LayerAcc (for example `A.X`), a GraphAcc "
                "(for example `A.obsp[...]`), or a MultiAcc "
                f"(for example `A.obsm[...]`), got {acc!r}."
            )
        if isinstance(acc, MultiAcc | GraphAcc) and dim is not None and acc.dim != dim:
            raise ValueError(f"`dim` ({dim!r}) does not match `acc`'s ({acc.dim!r}).")
        data = _get_accessor_array(adata, acc)
        return data.T if isinstance(acc, LayerAcc) and dim == "var" else data

    if dim is None:
        dim = "var" if varm is not None or varp is not None else "obs"

    match selected:
        case []:
            return adata.X.T if dim == "var" else adata.X
        case [("layer", key)]:
            data = adata.layers[key]
            return data.T if dim == "var" else data
        case [("use_raw", True)]:
            return adata.raw.X
        case [("obsm", key)]:
            if dim == "var":
                raise ValueError("`obsm` cannot be used when `dim` is `var`.")
            return adata.obsm[key]
        case [("obsp", key)]:
            if dim == "var":
                raise ValueError("`obsp` cannot be used when `dim` is `var`.")
            return adata.obsp[key]
        case [("varm", key)]:
            if dim == "obs":
                raise ValueError("`varm` cannot be used when `dim` is `obs`.")
            return adata.varm[key]
        case [("varp", key)]:
            if dim == "obs":
                raise ValueError("`varp` cannot be used when `dim` is `obs`.")
            return adata.varp[key]
        case _:
            names = [f"`{key}`" for key, _ in selected]
            names[-1] = f"or {names[-1]}"
            raise ValueError(f"Only one of {', '.join(names)} can be specified.")


def _resolve_ref(
    ref: AdRef | str | Collection[AdRef | str],
) -> AdRef | str | list[AdRef] | list[str]:
    """Resolve reference strings under the Scanpy 2 preview preset."""
    if isinstance(ref, Collection) and not isinstance(ref, AdRef | str):
        refs = list(ref)
        if not all(isinstance(item, AdRef | str) for item in refs):
            raise TypeError("All references must be strings or AdRefs.")
        resolved = [_resolve_ref(item) for item in refs]
        n_strings = sum(isinstance(item, str) for item in resolved)
        if n_strings not in {0, len(resolved)}:
            raise TypeError(
                "All references must be either AdRefs or strings, not a mix."
            )
        return resolved

    if not isinstance(ref, AdRef | str):
        raise TypeError(f"References must be strings or AdRefs, got {ref!r}.")
    if (
        isinstance(ref, AdRef)
        or settings.preset is not Preset.ScanpyV2Preview
        or A is None
    ):
        return ref
    resolved = _resolve_vector_acc(ref, strict=False)
    return ref if resolved is None else resolved


def _ref_dim(
    ref: AdRef | str, *, dim: Literal["obs", "var"] | None
) -> Literal["obs", "var"]:
    """Derive and validate the dimension of one vector reference."""
    ref = _resolve_ref(ref)
    if isinstance(ref, str):
        return dim or "obs"
    ref_dims = tuple(ref.dims)
    if len(ref_dims) != 1:
        raise ValueError(f"Reference `{ref}` is not one-dimensional.")
    ref_dim = ref_dims[0]
    if dim is not None and dim != ref_dim:
        raise ValueError(f"Dimension of `{ref}` ({ref_dim}) does not match `{dim}`.")
    return ref_dim


def _refs_dim(
    refs: Collection[AdRef | str], *, dim: Literal["obs", "var"] | None = None
) -> Literal["obs", "var"]:
    """Derive the single dimension shared by a collection of references."""
    if not refs:
        raise ValueError("At least one reference is required.")
    dims = {_ref_dim(ref, dim=dim) for ref in refs}
    if len(dims) != 1:
        raise ValueError(
            "All references must refer to the same single axis "
            f"(`obs` or `var`), got {dims}."
        )
    return next(iter(dims))


def _fetch_vec(adata: AnnData, ref: AdRef | str, *, dim: Literal["obs", "var"]) -> Any:
    """Retrieve the vector pointed to by an already dimension-resolved reference."""
    ref = _resolve_ref(ref)
    if isinstance(ref, str):
        return getattr(adata, dim)[ref]
    return ref.acc.get(adata, ref.idx)


@overload
def _get_vec(
    adata: AnnData,
    ref: Collection[AdRef] | Collection[str],
    *,
    dim: Literal["obs", "var"] | None = None,
) -> list[Any]: ...


@overload
def _get_vec(
    adata: AnnData,
    ref: AdRef | str,
    *,
    dim: Literal["obs", "var"] | None = None,
) -> Any: ...


def _get_vec(
    adata: AnnData,
    ref: AdRef | str | Collection[AdRef] | Collection[str],
    *,
    dim: Literal["obs", "var"] | None = None,
) -> Any:
    """Return the vector or vectors selected by AnnData references."""
    if isinstance(ref, Collection) and not isinstance(ref, AdRef | str):
        refs = _resolve_ref(ref)
        resolved_dim = _refs_dim(refs, dim=dim)
        return [_fetch_vec(adata, item, dim=resolved_dim) for item in refs]

    resolved = _resolve_ref(ref)
    resolved_dim = _ref_dim(resolved, dim=dim)
    return _fetch_vec(adata, resolved, dim=resolved_dim)


def _to_numpy_1d(values: Any) -> Any:
    """Materialize a device or lazy vector on the host for metadata handling."""
    if isinstance(values, DaskArray):
        return _to_numpy_1d(values.compute())
    if isinstance(values, cp.ndarray):
        return cp.asnumpy(values).reshape(-1)
    if cp_sparse.issparse(values):
        return values.toarray().get().reshape(-1)
    if isinstance(values, np.ndarray):
        return values.reshape(-1)
    return values


def _check_mask(
    data: AnnData | Any,
    mask: str | AdRef[Idx2D | int, AnnData] | Any,
    dim: Literal["obs", "var"],
    *,
    allow_probabilities: bool = False,
) -> Any:
    """Resolve and validate a boolean mask or probability vector."""
    if mask is None:
        return None
    description = "mask/probabilities" if allow_probabilities else "mask"

    if isinstance(mask, AdRef | str):
        mask = _resolve_ref(mask)
        if not isinstance(data, AnnData):
            raise ValueError(
                f"Cannot use a reference for {description} without providing "
                "an AnnData object."
            )
        try:
            mask_array = _get_vec(data, mask, dim=dim)
        except KeyError:
            if isinstance(mask, AdRef):
                message = (
                    f"Did not find `{mask}` in `adata`. Either add the "
                    f"{description} first to `adata.{dim}` or pass an array."
                )
            else:
                message = f"Did not find `adata.{dim}[{mask!r}]`."
            raise ValueError(message) from None
    else:
        mask_array = mask

    expected = data.shape[0 if dim == "obs" else 1]
    if mask_array.shape != (expected,):
        raise ValueError(f"The shape of the {description} does not match the data.")

    is_bool = pd.api.types.is_bool_dtype(mask_array.dtype)
    if not allow_probabilities and not is_bool:
        raise ValueError("Mask array must be boolean.")
    if allow_probabilities and not (
        is_bool or pd.api.types.is_float_dtype(mask_array.dtype)
    ):
        raise ValueError(
            f"{description.capitalize()} array must be boolean or floating point."
        )
    return mask_array


def _get_obs_rep(
    adata: AnnData,
    *,
    use_raw: bool = False,
    layer: str | None = None,
    obsm: str | None = None,
    obsp: str | None = None,
) -> Any:
    """Return the selected observation-aligned representation."""
    return _get_arr(
        adata,
        dim="obs",
        use_raw=use_raw,
        layer=layer,
        obsm=obsm,
        obsp=obsp,
    )


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
