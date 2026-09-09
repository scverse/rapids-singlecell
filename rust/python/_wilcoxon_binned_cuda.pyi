from typing import Annotated, overload

import numpy as np
from numpy.typing import NDArray

@overload
def dense_hist(
    X: Annotated[
        NDArray[np.float32], {"order": "F", "device": "cuda", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    stream: int = 0,
) -> None: ...
@overload
def dense_hist(
    X: Annotated[
        NDArray[np.float64], {"order": "F", "device": "cuda", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    stream: int = 0,
) -> None: ...
@overload
def dense_hist(
    X: Annotated[
        NDArray[np.float32], {"order": "F", "device": "cuda_managed", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda_managed"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    stream: int = 0,
) -> None: ...
@overload
def dense_hist(
    X: Annotated[
        NDArray[np.float64], {"order": "F", "device": "cuda_managed", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda_managed"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    stream: int = 0,
) -> None: ...
@overload
def csr_hist(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    gene_start: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_hist(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    gene_start: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_hist(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    gene_start: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_hist(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    gene_start: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_hist(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda_managed"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    gene_start: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_hist(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda_managed"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    gene_start: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_hist(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda_managed"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    gene_start: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_hist(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda_managed"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    gene_start: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_hist(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    gene_start: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_hist(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    gene_start: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_hist(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    gene_start: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_hist(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    gene_start: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_hist(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda_managed"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    gene_start: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_hist(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda_managed"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    gene_start: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_hist(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda_managed"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    gene_start: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_hist(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    gcodes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    hist: Annotated[NDArray[np.uint32], {"order": "C", "device": "cuda_managed"}],
    *,
    n_cells: int,
    n_genes: int,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    gene_start: int,
    stream: int = 0,
) -> None: ...
