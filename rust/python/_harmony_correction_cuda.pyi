from __future__ import annotations

from typing import Any

__backend__: str

def correction_fast(
    X: Any,
    *,
    R: Any,
    O: Any,
    cats: Any,
    cat_offsets: Any,
    cell_indices: Any,
    lambda_kb: Any,
    n_cells: int,
    n_pcs: int,
    n_clusters: int,
    n_batches: int,
    Z: Any,
    inv_mat: Any,
    R_col: Any,
    Phi_t_diag_R_X: Any,
    W: Any,
    g_factor: Any,
    g_P_row0: Any,
    stream: int = 0,
    handle: int,
) -> None: ...
def compute_inv_mat(
    O: Any,
    *,
    lambda_kb: Any,
    n_batches: int,
    n_clusters: int,
    cluster_k: int,
    inv_mat: Any,
    g_factor: Any,
    g_P_row0: Any,
    stream: int = 0,
) -> None: ...
