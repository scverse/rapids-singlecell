from typing import Annotated, overload

import numpy as np
from numpy.typing import NDArray

@overload
def csc_scale_diff(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    std: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    ncols: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_scale_diff(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    std: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    ncols: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_scale_diff(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    std: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    ncols: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_scale_diff(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    std: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    ncols: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_scale_diff(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    std: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    ncols: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_scale_diff(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    std: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    ncols: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_scale_diff(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    std: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    ncols: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_scale_diff(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    std: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    ncols: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_scale_diff(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    std: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    mask: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    clipper: float,
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_scale_diff(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    std: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    mask: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    clipper: float,
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_scale_diff(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    std: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    mask: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    clipper: float,
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_scale_diff(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    std: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    mask: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    clipper: float,
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_scale_diff(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    std: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    mask: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    clipper: float,
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_scale_diff(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    std: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    mask: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    clipper: float,
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_scale_diff(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    std: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    mask: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    clipper: float,
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_scale_diff(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    std: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    mask: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    clipper: float,
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def dense_scale_center_diff(
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    mean: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    std: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    mask: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    clipper: float,
    nrows: int,
    ncols: int,
    stream: int = 0,
) -> None: ...
@overload
def dense_scale_center_diff(
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    mean: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    std: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    mask: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    clipper: float,
    nrows: int,
    ncols: int,
    stream: int = 0,
) -> None: ...
@overload
def dense_scale_center_diff(
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    mean: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    std: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    mask: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    clipper: float,
    nrows: int,
    ncols: int,
    stream: int = 0,
) -> None: ...
@overload
def dense_scale_center_diff(
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    mean: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    std: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    mask: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    clipper: float,
    nrows: int,
    ncols: int,
    stream: int = 0,
) -> None: ...
@overload
def dense_scale_diff(
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    std: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    mask: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    clipper: float,
    nrows: int,
    ncols: int,
    stream: int = 0,
) -> None: ...
@overload
def dense_scale_diff(
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    std: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    mask: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    clipper: float,
    nrows: int,
    ncols: int,
    stream: int = 0,
) -> None: ...
@overload
def dense_scale_diff(
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    std: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    mask: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    clipper: float,
    nrows: int,
    ncols: int,
    stream: int = 0,
) -> None: ...
@overload
def dense_scale_diff(
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    std: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    mask: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    clipper: float,
    nrows: int,
    ncols: int,
    stream: int = 0,
) -> None: ...
