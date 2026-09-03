from __future__ import annotations

from typing import TYPE_CHECKING

from ._autocorr_core import _autocorr_cupy

if TYPE_CHECKING:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix


def _gearys_C_cupy(
    data: cp.ndarray | csr_matrix,
    adj_matrix_cupy: csr_matrix,
    n_permutations: int | None = 100,
    *,
    multi_gpu: bool | list[int] | str | None = None,
) -> tuple[cp.ndarray, cp.ndarray | None]:
    return _autocorr_cupy(
        data, adj_matrix_cupy, n_permutations, geary=True, multi_gpu=multi_gpu
    )
