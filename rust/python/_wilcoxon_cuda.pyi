"""CUDA kernels for Wilcoxon rank-sum test"""

from typing import Annotated, overload

import numpy as np
from numpy.typing import NDArray

@overload
def ovr_rank_dense_host_streaming(
    X: Annotated[
        NDArray[np.float32], {"shape": (None, None), "order": "C", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    total_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    total_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool,
    compute_totals: bool,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_rank_dense_host_streaming(
    X: Annotated[
        NDArray[np.float32], {"shape": (None, None), "order": "F", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    total_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    total_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool,
    compute_totals: bool,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_rank_dense_host_streaming(
    X: Annotated[
        NDArray[np.float64], {"shape": (None, None), "order": "C", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    total_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    total_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool,
    compute_totals: bool,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_rank_dense_host_streaming(
    X: Annotated[
        NDArray[np.float64], {"shape": (None, None), "order": "F", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    total_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    total_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool,
    compute_totals: bool,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_rank_dense_host_streaming(
    X: Annotated[
        NDArray[np.float32], {"shape": (None, None), "order": "C", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    total_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    total_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool,
    compute_totals: bool,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_rank_dense_host_streaming(
    X: Annotated[
        NDArray[np.float32], {"shape": (None, None), "order": "F", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    total_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    total_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool,
    compute_totals: bool,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_rank_dense_host_streaming(
    X: Annotated[
        NDArray[np.float64], {"shape": (None, None), "order": "C", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    total_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    total_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool,
    compute_totals: bool,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_rank_dense_host_streaming(
    X: Annotated[
        NDArray[np.float64], {"shape": (None, None), "order": "F", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    total_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    total_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool,
    compute_totals: bool,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovo_rank_dense_host_streaming(
    X: Annotated[
        NDArray[np.float32], {"shape": (None, None), "order": "C", "writable": False}
    ],
    ref_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    grp_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovo_rank_dense_host_streaming(
    X: Annotated[
        NDArray[np.float32], {"shape": (None, None), "order": "F", "writable": False}
    ],
    ref_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    grp_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovo_rank_dense_host_streaming(
    X: Annotated[
        NDArray[np.float64], {"shape": (None, None), "order": "C", "writable": False}
    ],
    ref_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    grp_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovo_rank_dense_host_streaming(
    X: Annotated[
        NDArray[np.float64], {"shape": (None, None), "order": "F", "writable": False}
    ],
    ref_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    grp_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovo_rank_dense_host_streaming(
    X: Annotated[
        NDArray[np.float32], {"shape": (None, None), "order": "C", "writable": False}
    ],
    ref_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    grp_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovo_rank_dense_host_streaming(
    X: Annotated[
        NDArray[np.float32], {"shape": (None, None), "order": "F", "writable": False}
    ],
    ref_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    grp_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovo_rank_dense_host_streaming(
    X: Annotated[
        NDArray[np.float64], {"shape": (None, None), "order": "C", "writable": False}
    ],
    ref_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    grp_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovo_rank_dense_host_streaming(
    X: Annotated[
        NDArray[np.float64], {"shape": (None, None), "order": "F", "writable": False}
    ],
    ref_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    grp_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovo_rank_dense_tiered_unsorted_ref(
    ref_data: Annotated[
        NDArray[np.float32], {"order": "F", "device": "cuda", "writable": False}
    ],
    grp_data: Annotated[
        NDArray[np.float32], {"order": "F", "device": "cuda", "writable": False}
    ],
    grp_offsets: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
    stream: int = 0,
) -> None: ...
@overload
def ovo_rank_dense_tiered_unsorted_ref(
    ref_data: Annotated[
        NDArray[np.float32], {"order": "F", "device": "cuda_managed", "writable": False}
    ],
    grp_data: Annotated[
        NDArray[np.float32], {"order": "F", "device": "cuda_managed", "writable": False}
    ],
    grp_offsets: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
    stream: int = 0,
) -> None: ...
@overload
def ovr_rank_dense_streaming(
    block: Annotated[
        NDArray[np.float32], {"order": "F", "device": "cuda", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
    stream: int = 0,
) -> None: ...
@overload
def ovr_rank_dense_streaming(
    block: Annotated[
        NDArray[np.float32], {"order": "F", "device": "cuda_managed", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
    stream: int = 0,
) -> None: ...
def _set_host_worker_limit(limit: int) -> int: ...
