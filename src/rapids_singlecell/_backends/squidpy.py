"""Squidpy backend using native defaults and small niche argument bridges."""

from __future__ import annotations

import inspect
import warnings
from functools import wraps

from rapids_singlecell._utils._random import _seed_from_rng
from rapids_singlecell.squidpy_gpu import calculate_niche as _calculate_niche
from rapids_singlecell.squidpy_gpu import calculate_niche_cellcharter as _cellcharter
from rapids_singlecell.squidpy_gpu import calculate_niche_neighborhood as _neighborhood
from rapids_singlecell.squidpy_gpu import calculate_niche_utag as _utag
from rapids_singlecell.squidpy_gpu import co_occurrence, spatial_autocorr
from rapids_singlecell.squidpy_gpu import ligrec as _ligrec

name = "rapids-singlecell"
aliases = ["cuda", "rapids", "rapids_singlecell"]

SQUIDPY_LIGREC_CPU_ONLY = {
    "seed": None,
    "n_jobs": None,
    "show_progress_bar": True,
    "numba_parallel": None,
}


@wraps(_ligrec)
def ligrec(*args, **kwargs):
    ignored = []
    for key, default in SQUIDPY_LIGREC_CPU_ONLY.items():
        if key in kwargs and kwargs.pop(key) != default:
            ignored.append(key)
    if ignored:
        warnings.warn(
            f"Parameters {', '.join(ignored)} have no effect on the "
            "rapids-singlecell backend.",
            UserWarning,
            stacklevel=2,
        )
    return _ligrec(*args, **kwargs)


def _niche_adapter(func, *, flavor_api: bool = False):
    """Expose native niche signatures with Squidpy's argument names."""
    names = {"adata": "data"}
    defaults = {}
    if flavor_api:
        names.update(inplace="copy", random_state="rng")
        defaults.update(copy=False, rng=None)

    @wraps(func)
    def adapted(data, **kwargs):
        if kwargs.get("flavor") == "spatialleiden":
            raise NotImplementedError("Use `backend='cpu'` for flavor='spatialleiden'.")
        if flavor_api:
            if "copy" in kwargs:
                kwargs["inplace"] = not kwargs.pop("copy")
            if (rng := kwargs.pop("rng", None)) is not None:
                kwargs["random_state"] = _seed_from_rng(rng, allow_none=False)
        return func(data, **kwargs)

    signature = inspect.signature(func)
    parameters = []
    for parameter in signature.parameters.values():
        name = names.get(parameter.name, parameter.name)
        parameters.append(
            parameter.replace(name=name, default=defaults.get(name, parameter.default))
        )
    adapted.__signature__ = signature.replace(parameters=parameters)
    return adapted


calculate_niche = _niche_adapter(_calculate_niche)
calculate_niche_neighborhood = _niche_adapter(_neighborhood, flavor_api=True)
calculate_niche_utag = _niche_adapter(_utag, flavor_api=True)
calculate_niche_cellcharter = _niche_adapter(_cellcharter, flavor_api=True)

__all__ = [
    "calculate_niche",
    "calculate_niche_cellcharter",
    "calculate_niche_neighborhood",
    "calculate_niche_utag",
    "co_occurrence",
    "ligrec",
    "spatial_autocorr",
]
