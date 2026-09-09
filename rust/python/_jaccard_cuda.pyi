from __future__ import annotations

from typing import Any

__backend__: str

def jaccard_shared_counts(
    knn: Any,
    *,
    n_obs: int,
    k: int,
    jaccard_vals: Any,
    stream: int = 0,
) -> None: ...
