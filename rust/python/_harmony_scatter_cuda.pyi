from __future__ import annotations

from typing import Any

__backend__: str

def scatter_add(
    v: Any,
    *,
    cats: Any,
    n_cells: int,
    n_pcs: int,
    n_covariates: int = 1,
    switcher: int,
    a: Any,
    stream: int = 0,
) -> None: ...
def scatter_add_shared(
    v: Any,
    *,
    cats: Any,
    n_cells: int,
    n_pcs: int,
    n_batches: int,
    n_covariates: int = 1,
    switcher: int,
    a: Any,
    n_blocks: int,
    stream: int = 0,
) -> None: ...
def gather_rows(
    src: Any,
    *,
    idx: Any,
    dst: Any,
    n_rows: int,
    n_cols: int,
    stream: int = 0,
) -> None: ...
def scatter_rows(
    src: Any,
    *,
    idx: Any,
    dst: Any,
    n_rows: int,
    n_cols: int,
    stream: int = 0,
) -> None: ...
def gather_int(
    src: Any,
    *,
    idx: Any,
    dst: Any,
    n: int,
    stream: int = 0,
) -> None: ...
