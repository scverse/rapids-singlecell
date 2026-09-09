from __future__ import annotations

from typing import Any

__backend__: str

def penalty(
    E: Any,
    *,
    O: Any,
    theta: Any,
    penalty: Any,
    n_batches: int,
    n_clusters: int,
    stabilized: bool,
    stream: int = 0,
) -> None: ...
def fused_pen_norm_int(
    similarities: Any,
    *,
    penalty: Any,
    cats: Any,
    idx_in: Any,
    R_out: Any,
    term: float,
    n_rows: int,
    n_cols: int,
    n_covariates: int = 1,
    stream: int = 0,
) -> None: ...
