from __future__ import annotations

import warnings
from dataclasses import dataclass
from inspect import signature
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class _Cutoffs:
    min_disp: float
    max_disp: float
    min_mean: float
    max_mean: float

    @classmethod
    def validate(
        cls,
        *,
        n_top_genes: int | None,
        min_disp: float,
        max_disp: float,
        min_mean: float,
        max_mean: float,
    ) -> _Cutoffs | int:
        if n_top_genes is None:
            return cls(min_disp, max_disp, min_mean, max_mean)

        # Import here to avoid circular import
        from . import highly_variable_genes

        cutoffs = {"min_disp", "max_disp", "min_mean", "max_mean"}
        defaults = {
            p.name: p.default
            for p in signature(highly_variable_genes).parameters.values()
            if p.name in cutoffs
        }
        if {k: v for k, v in locals().items() if k in cutoffs} != defaults:
            msg = "If you pass `n_top_genes`, all cutoffs are ignored."
            warnings.warn(msg, UserWarning)
        return n_top_genes

    def in_bounds(
        self,
        mean: NDArray,
        dispersion_norm: NDArray,
    ) -> NDArray:
        return (
            (mean > self.min_mean)
            & (mean < self.max_mean)
            & (dispersion_norm > self.min_disp)
            & (dispersion_norm < self.max_disp)
        )
