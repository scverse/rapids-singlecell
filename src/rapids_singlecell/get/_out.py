"""The `use`/`out` parameter contract of scanpy 1.13 (scverse/scanpy#4359).

`use` says where a function reads from, `out` where it writes to:
`False` (default places), `True` (return instead of write), an accessor,
or for several outputs a mapping from output names to accessors.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from functools import wraps
from inspect import signature
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from anndata import AnnData
from scverse_misc import Deprecation, deprecated_arg

from rapids_singlecell._settings import Default, Preset, settings

from ._utils import AdRef, GraphAcc, LayerAcc, MultiAcc, _require_anndata_acc, _resolve_matrix_acc, _to_numpy_1d

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from ._utils import ArrAcc

type OutKind = Literal["layer", "obsm", "varm", "obsp", "varp", "obs", "var", "uns"]
"""Kind of place an output is written to; `'layer'` includes `.X`."""

type Dest = dict[str, Any]
"""Old-style destination, e.g. `{'obsm': 'X_pca'}`; `{}` is `.X`."""

DEPR_USE = Deprecation("0.19.0", "Use `use` instead.")
DEPR_COPY = Deprecation("0.19.0", "Copy `adata` before calling this function instead.")
DEPR_INPLACE = Deprecation("0.19.0", "Use `out=True` instead of `inplace=False`.")
DEPR_KEY_ADDED = Deprecation("0.19.0", "Use `out` instead.")
DEPR_RAW = Deprecation("0.19.0", ":attr:`~anndata.AnnData.raw` is deprecated. Keep the data in a layer instead.")
DEPR_BATCH_KEY = Deprecation("0.19.0", "Use `batch` instead, e.g. `batch=A.obs['b']` for `batch_key='b'`.")
DEPR_NEIGHBORS_KEY = Deprecation(
    "0.19.0", "Use `use` instead, e.g. `use=A.obsp['k_connectivities']` for `neighbors_key='k'`."
)
DEPR_OBSP = Deprecation("0.19.0", "Use `use=A.obsp[...]` instead.")


@contextmanager
def own_deprecations() -> Generator[None]:
    """Silence rsc's own deprecation warnings during an internal call."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"The argument \w+ is deprecated", FutureWarning)
        yield


def _resolve(acc: ArrAcc | AdRef | str, *, vec: bool = False) -> ArrAcc | AdRef:
    if not isinstance(acc, str):
        return acc
    return _require_anndata_acc().resolve(acc, vec=True) if vec else _resolve_matrix_acc(acc)


def _resolve_obs(use: Any, *, kinds: tuple[type, ...] = (LayerAcc, MultiAcc)) -> Any:
    """Resolve a `use` string and require an `obs`-aligned accessor of one of `kinds`."""
    if use is None:
        return None
    use = _resolve(use)
    if isinstance(use, MultiAcc | GraphAcc) and use.dim != "obs":
        raise ValueError(f"Input must be aligned to `obs`, but {use!r} is aligned to `var`.")
    if not isinstance(use, kinds):
        names: dict[type, str] = {
            LayerAcc: "`A.X`/`A.layers[...]`",
            MultiAcc: "`A.obsm[...]`",
            GraphAcc: "`A.obsp[...]`",
        }
        raise TypeError(f"`use` must be one of {', '.join(names[k] for k in kinds)}, was {use!r}.")
    return use


def _use_args(use: ArrAcc | str | None, *, kinds: tuple[type, ...] = (LayerAcc, MultiAcc), **legacy: Any) -> Dest:
    """Merge `use` and the deprecated `layer`/`obsm`/`obsp` into old-style choices."""
    passed = {k: v for k, v in legacy.items() if v is not None}
    if use is None:
        return passed
    if passed:
        raise TypeError(f"Pass either `use` or `{'`/`'.join(passed)}`, not both.")
    return _dest(_resolve_obs(use, kinds=kinds))


def _dest(acc: Any) -> Dest:
    """Translate an accessor into old-style choices, e.g. `A.obsm['x']` -> `{'obsm': 'x'}`."""
    from anndata.acc import MetaAcc

    if isinstance(acc, str):
        acc = _resolve(acc, vec="." in acc and acc.split(".", 1)[0] in {"obs", "var"})
    match acc:
        case LayerAcc(k=None):
            return {}
        case LayerAcc(k=key):
            return {"layer": key}
        case MultiAcc(dim=dim, k=key):
            return {f"{dim}m": key}
        case GraphAcc(dim=dim, k=key):
            return {f"{dim}p": key}
        case AdRef(acc=MetaAcc(dim=dim), idx=str(col)):
            return {dim: col}
    raise TypeError(f"Cannot write to {acc!r}.")


def _kind(dest: Dest) -> OutKind:
    return cast("OutKind", next(iter(dest), "layer"))


def _key(dest: Dest | None) -> str | None:
    return None if dest is None else next(iter(dest.values()), None)


def _set(adata: AnnData, val: Any, dest: Dest) -> None:
    if not dest:
        adata.X = val
    elif "layer" in dest:
        adata.layers[dest["layer"]] = val
    else:
        ((kind, key),) = dest.items()
        getattr(adata, kind)[key] = _to_numpy_1d(val) if kind in {"obs", "var"} else val


def _check_legacy(out: object, name: str, *, default: object = False) -> None:
    if out is not default:
        raise TypeError(f"Pass either `out` or `{name}`, not both.")


def _legacy_out[O](out: O, name: str, value: O | None, *, default: Any = False) -> O:
    """Merge a deprecated parameter into `out`, e.g. `inplace=False` as `out=True`."""
    if value is None:
        return out
    _check_legacy(out, name, default=default)
    return value


def _write_out[V](adata: AnnData, val: V, *, out: object, dest: Dest) -> V | None:
    """Honor the `out` of a single-output function: return `val`, or write it to `out` or `dest`."""
    if out is True:
        return val
    if out is not False and not isinstance(out, LayerAcc | MultiAcc | GraphAcc | AdRef | str):
        raise TypeError(f"`out` must be a `bool` or an accessor, was {out!r}.")
    _set(adata, val, dest if out is False else _dest(out))
    return None


def _out_key(out: object, key_added: str | None, *, kind: OutKind, default: Any = None) -> Any:
    """Translate the `out` of a single-output function into a key of `kind` (or `True`)."""
    if key_added is not None and out is not True:
        _check_legacy(out, "key_added")
        return key_added
    if out is True or out is False:
        return True if out else default
    if kind == "uns":
        raise TypeError(f"`.uns` accessors aren't supported yet, `out` must be a `bool`, was {out!r}.")
    dest = _dest(out)
    if _kind(dest) != kind or (key := _key(dest)) is None:
        raise TypeError(f"`out` must point to `.{kind}`, was {out!r}.")
    return key


def _out_dests[D: tuple](
    out: object,
    key_added: str | None,
    *,
    dests: Callable[[Any], D],
    kinds: tuple,
    uns: bool = False,
    default: Any = None,
    primary: bool = True,
) -> tuple[Any, D] | Literal[True]:
    """Translate the `out` of a function with several outputs into one destination each.

    `kinds` (an instance of the result `NamedTuple`) holds the allowed `OutKind`s per output,
    `dests(key)` where the default/`key_added` puts each one (`None` to skip).
    Returns `True` (return the outputs), or the key for the `.uns` entry and the destinations.
    A single accessor names the primary output `k`; the others are named `k` in kinds of places
    no other output goes to, else `k_{name}`. A mapping writes only the outputs it names.
    """
    if key_added is not None and out is not True:
        _check_legacy(out, "key_added")
        return key_added, dests(key_added)
    names: tuple[str, ...] = getattr(kinds, "_fields")  # noqa: B009
    defaults = dests(default)
    if out is True:
        return True
    if out is False:
        return default, defaults
    if primary and isinstance(out, LayerAcc | MultiAcc | GraphAcc | AdRef | str):
        result = _follow_primary(_check_kind(kinds, names[0], out), defaults)
        key = _key(result[0])
    elif isinstance(out, Mapping):
        if unknown := out.keys() - set(names):
            raise TypeError(f"Unknown output(s) {sorted(unknown)}, valid are {names}.")
        resolved = {name: _check_kind(kinds, name, acc) for name, acc in out.items()}
        places = Counter(tuple(d.items()) for d in resolved.values())
        if dupes := [n for n, d in resolved.items() if places[tuple(d.items())] > 1]:
            raise ValueError(f"Outputs {dupes} can't be written to the same place, was {out!r}.")
        cls: Any = type(defaults)
        result = cls(*(resolved.get(name) for name in names))
        key = None if result[0] == defaults[0] else _key(result[0])
    else:
        what = "a `bool`, an accessor for the primary output, or" if primary else "a `bool` or"
        raise TypeError(f"`out` must be {what} a mapping with keys {names}, was {out!r}.")
    if uns and result[0] is None:
        raise TypeError(f"`out` must include {names[0]!r}, after which the `.uns` entry is named, was {out!r}.")
    return default if key is None else key, result


def _check_kind(kinds: tuple, name: str, acc: object) -> Dest:
    if not isinstance(acc, LayerAcc | MultiAcc | GraphAcc | AdRef | str):
        raise TypeError(f"Output {name!r} must be an accessor, was {acc!r}.")
    dest = _dest(acc)
    if _kind(dest) not in (valid := getattr(kinds, name)):
        raise TypeError(f"Output {name!r} must be written to one of {sorted(valid)}, was {acc!r}.")
    return dest


def _follow_primary[D: tuple](first: Dest, defaults: D) -> D:
    cls: Any = type(defaults)
    if (key := _key(first)) is None:  # `.X`
        return cls(first, *defaults[1:])
    counts = Counter(_kind(d) for d in (first, *defaults[1:]) if d is not None)
    return cls(
        first,
        *(
            None if d is None else {(k := _kind(d)): key if counts[k] == 1 else f"{key}_{name}"}
            for name, d in zip(getattr(defaults, "_fields")[1:], defaults[1:], strict=True)  # noqa: B009
        ),
    )


def _write_dests(adata: AnnData, vals: tuple, dests: tuple) -> None:
    for val, dest in zip(vals, dests, strict=True):
        if dest is not None:
            _set(adata, val, dest)


def _use_to_uns(use: ArrAcc | str | None, *, rep: bool) -> dict[str, Any]:
    """Store `use` in `.uns` params the way `use_rep`/`layer`/`obsm` were stored."""
    if use is None:
        return {}
    match _resolve_obs(use):
        case LayerAcc(k=None):
            return {"use_rep": "X"} if rep else {}
        case LayerAcc(k=key):
            return {"layer": key}
        case MultiAcc(k=key):
            return {"use_rep" if rep else "obsm": key}
    raise AssertionError


def deprecated_copy[**P, R](func: Callable[P, R]) -> Callable[P, R | AnnData]:
    """Deprecate `copy`, running `func` on a copy of its first argument and returning that."""
    sig = signature(func)
    first = next(iter(sig.parameters))

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | AnnData:
        bound = sig.bind(*args, **kwargs)
        if not bound.arguments.get("copy") or not isinstance(data := bound.arguments[first], AnnData):
            return func(*args, **kwargs)
        if bound.arguments.get("out") is True or bound.arguments.get("inplace") is False:
            raise ValueError("`copy=True` cannot be used with `out=True`.")
        bound.arguments[first] = data = data.copy()
        bound.arguments["copy"] = False
        res = func(*bound.args, **bound.kwargs)
        return data if res is None else res

    return deprecated_arg("copy", DEPR_COPY)(wrapper)


def _obs_col(ref: AdRef | str | None, *, param: str) -> str | None:
    """Name of the `.obs` column a grouping `ref` points to (strings stay names unless the v2 preset)."""
    from ._utils import _resolve_ref

    resolved: Any = None if ref is None else _resolve_ref(ref)
    if resolved is None or isinstance(resolved, str):
        return resolved
    if (dest := _dest(resolved)).keys() != {"obs"}:
        raise TypeError(f"`{param}` must point to a column of `.obs`, was {ref!r}.")
    return dest["obs"]


def _mask_arg(mask: Any, legacy: Any, *, dim: Literal["obs", "var"]) -> Any:
    """Merge `mask` with its deprecated `mask_{dim}`, whose strings name a column of `.{dim}`."""
    if legacy is None:
        return _resolve(mask, vec=True) if isinstance(mask, str) else mask
    if mask is not None and not isinstance(mask, Default):
        raise TypeError(f"Pass either `mask` or `mask_{dim}`, not both.")
    if isinstance(legacy, str) and settings.preset is Preset.ScanpyV2Preview:
        return getattr(_require_anndata_acc(), dim)[legacy]
    return legacy


def _ref_to_json(ref: Any) -> str | list[str] | None:
    """Store a mask in `.uns` params like scanpy: an `AdRef` as JSON, a column name as is, arrays as `None`."""
    if isinstance(ref, AdRef):
        return [json.dumps(_require_anndata_acc().to_json(ref))]
    return ref if isinstance(ref, str) else None


def _ref_from_json(ref: Any) -> Any:
    """Parse a mask stored by `_ref_to_json` or scanpy."""
    if isinstance(ref, list | tuple | np.ndarray) and len(ref) == 1 and isinstance(ref[0], str):
        return _require_anndata_acc().from_json(json.loads(ref[0]))
    return ref
