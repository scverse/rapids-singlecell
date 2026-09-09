from typing import Annotated, overload

import numpy as np
from numpy.typing import NDArray

@overload
def nan_mean_minor(
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    nans: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def nan_mean_minor(
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    nans: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def nan_mean_minor(
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    nans: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def nan_mean_minor(
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    nans: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def nan_mean_minor(
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    nans: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def nan_mean_minor(
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    nans: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def nan_mean_minor(
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    nans: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def nan_mean_minor(
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    nans: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def nan_mean_major(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    nans: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    major: int,
    minor: int,
    stream: int = 0,
) -> None: ...
@overload
def nan_mean_major(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    nans: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    major: int,
    minor: int,
    stream: int = 0,
) -> None: ...
@overload
def nan_mean_major(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    nans: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    major: int,
    minor: int,
    stream: int = 0,
) -> None: ...
@overload
def nan_mean_major(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    nans: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    major: int,
    minor: int,
    stream: int = 0,
) -> None: ...
@overload
def nan_mean_major(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    nans: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    major: int,
    minor: int,
    stream: int = 0,
) -> None: ...
@overload
def nan_mean_major(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    nans: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    major: int,
    minor: int,
    stream: int = 0,
) -> None: ...
@overload
def nan_mean_major(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    nans: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    major: int,
    minor: int,
    stream: int = 0,
) -> None: ...
@overload
def nan_mean_major(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    nans: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    major: int,
    minor: int,
    stream: int = 0,
) -> None: ...
