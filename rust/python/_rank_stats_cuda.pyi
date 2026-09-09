from typing import Annotated, overload

import numpy as np
from numpy.typing import NDArray

@overload
def csr_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda_managed"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda_managed"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda_managed"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda_managed"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda_managed"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda_managed"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda_managed"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda_managed"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda_managed"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda_managed"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda_managed"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda_managed"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda_managed"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda_managed"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda_managed"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_tile_to_dense(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    out: Annotated[NDArray[np.float64], {"order": "F", "device": "cuda_managed"}],
    *,
    col_lb: int,
    col_ub: int,
    stream: int = 0,
) -> None: ...
@overload
def fdr_bh_reverse_cummin(
    values: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    stream: int = 0,
) -> None: ...
@overload
def fdr_bh_reverse_cummin(
    values: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    stream: int = 0,
) -> None: ...
@overload
def group_chunk_stats(
    block: Annotated[
        NDArray[np.float64], {"order": "F", "device": "cuda", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    group_sum_sq: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    compute_nnz: bool,
    stream: int = 0,
) -> None: ...
@overload
def group_chunk_stats(
    block: Annotated[
        NDArray[np.float64], {"order": "F", "device": "cuda_managed", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    group_sum_sq: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    compute_nnz: bool,
    stream: int = 0,
) -> None: ...
