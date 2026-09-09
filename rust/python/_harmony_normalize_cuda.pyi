from __future__ import annotations

from typing import Any

__backend__: str

def normalize(
    X: Any,
    *,
    rows: int,
    cols: int,
    stream: int = 0,
) -> None: ...
def l2_row_normalize(
    src: Any,
    *,
    dst: Any,
    n_rows: int,
    n_cols: int,
    stream: int = 0,
) -> None: ...
