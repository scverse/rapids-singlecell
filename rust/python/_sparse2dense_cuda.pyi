from __future__ import annotations

from typing import Any

__backend__: str

def sparse2dense(
    indptr: Any,
    index: Any,
    data: Any,
    *,
    out: Any,
    major: int,
    minor: int,
    c_switch: bool,
    max_nnz: int,
    stream: int = 0,
) -> None: ...
