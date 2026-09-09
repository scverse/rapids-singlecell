from __future__ import annotations

from typing import Any

__backend__: str

def colsum(
    A: Any,
    *,
    out: Any,
    rows: int,
    cols: int,
    stream: int = 0,
) -> None: ...
def colsum_atomic(
    A: Any,
    *,
    out: Any,
    rows: int,
    cols: int,
    stream: int = 0,
) -> None: ...
