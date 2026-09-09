from typing import Annotated, overload

import numpy as np
from numpy.typing import NDArray

@overload
def mean_var_major(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    vars: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    major: int,
    minor: int,
    stream: int = 0,
) -> None: ...
@overload
def mean_var_major(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    vars: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    major: int,
    minor: int,
    stream: int = 0,
) -> None: ...
@overload
def mean_var_major(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    vars: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    major: int,
    minor: int,
    stream: int = 0,
) -> None: ...
@overload
def mean_var_major(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    vars: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    major: int,
    minor: int,
    stream: int = 0,
) -> None: ...
@overload
def mean_var_major(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    vars: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    major: int,
    minor: int,
    stream: int = 0,
) -> None: ...
@overload
def mean_var_major(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    vars: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    major: int,
    minor: int,
    stream: int = 0,
) -> None: ...
@overload
def mean_var_major(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    vars: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    major: int,
    minor: int,
    stream: int = 0,
) -> None: ...
@overload
def mean_var_major(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    vars: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    major: int,
    minor: int,
    stream: int = 0,
) -> None: ...
@overload
def mean_var_minor(
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    vars: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def mean_var_minor(
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    vars: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def mean_var_minor(
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    vars: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def mean_var_minor(
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    vars: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def mean_var_minor(
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    vars: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def mean_var_minor(
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    vars: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def mean_var_minor(
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    vars: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def mean_var_minor(
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    means: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    vars: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    nnz: int,
    stream: int = 0,
) -> None: ...
