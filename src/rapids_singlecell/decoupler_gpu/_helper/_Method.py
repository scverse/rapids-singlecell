from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING

import pandas as pd

from rapids_singlecell.decoupler_gpu._helper._run import _run

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from rapids_singlecell.decoupler_gpu._helper._data import DataType


class MethodMeta:
    def __init__(
        self,
        name: str,
        desc: str,
        func: Callable,
        *,
        stype: str,
        adj: bool,
        weight: bool,
        test: bool,
        limits: tuple,
        reference: str,
    ):
        self.name = name
        self.desc = desc
        self.func = func
        self.stype = stype
        self.adj = adj
        self.weight = weight
        self.test = test
        self.limits = limits
        self.reference = reference

    def meta(self) -> pd.DataFrame:
        meta = pd.DataFrame(
            [
                {
                    "name": self.name,
                    "desc": self.desc,
                    "stype": self.stype,
                    "weight": self.weight,
                    "test": self.test,
                    "limits": self.limits,
                    "reference": self.reference,
                }
            ]
        )
        return meta


class Method(MethodMeta):
    def __init__(
        self,
        _method: MethodMeta,
        *,
        default_bsize: int = 5000,
        batch_on: Literal["rows", "backed"] = "rows",
        dense: bool = True,
        prepare: Callable[[dict], None] | None = None,
    ):
        """Keep execution defaults separate from the method's scientific metadata.

        ``batch_on="backed"`` processes in-memory and Dask input together, but
        still honors ``bsize`` for backed input. ``dense=False`` preserves CSR.
        ``prepare`` adjusts call arguments once, before extraction and batching.
        """
        super().__init__(
            name=_method.name,
            desc=_method.desc,
            func=_method.func,
            stype=_method.stype,
            adj=_method.adj,
            weight=_method.weight,
            test=_method.test,
            limits=_method.limits,
            reference=_method.reference,
        )
        self._method = _method
        self._default_bsize = default_bsize
        self._batch_on = batch_on
        self._dense = dense
        self._prepare = prepare
        self.__doc__ = self.func.__doc__
        # Expose each method's effective default while sharing one call signature.
        call_signature = signature(self.__call__)
        self.__signature__ = call_signature.replace(
            parameters=[
                p.replace(default=default_bsize) if p.name == "bsize" else p
                for p in call_signature.parameters.values()
            ]
        )

    def __call__(  # noqa: PLR0917 - Preserve decoupler's positional API.
        self,
        data: DataType,
        net: pd.DataFrame,
        tmin: int | float = 5,
        raw: bool = False,  # noqa: FBT001, FBT002
        empty: bool = True,  # noqa: FBT001, FBT002
        bsize: int | float | None = None,
        verbose: bool = False,  # noqa: FBT001, FBT002
        *,
        pre_load: bool = False,
        adj_pv_gpu: bool = False,
        **kwargs,
    ):
        kwargs.update(
            data=data,
            net=net,
            tmin=tmin,
            raw=raw,
            empty=empty,
            bsize=self._default_bsize if bsize is None else bsize,
            verbose=verbose,
            pre_load=pre_load,
            adj_pv_gpu=adj_pv_gpu,
        )
        if self._prepare is not None:
            self._prepare(kwargs)
        return _run(
            name=self.name,
            func=self.func,
            adj=self.adj,
            test=self.test,
            batch_on=self._batch_on,
            dense=self._dense,
            **kwargs,
        )
