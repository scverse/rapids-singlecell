from __future__ import annotations

from typing import Any

__backend__: str

def outer(
    E: Any,
    *,
    Pr_b: Any,
    R_sum: Any,
    n_cats: int,
    n_pcs: int,
    switcher: int,
    stream: int = 0,
) -> None: ...
