from typing import Annotated, overload

import numpy as np
from numpy.typing import NDArray

@overload
def expected_zeros(
    scaled_means: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    total_counts: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    expected: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    n_genes: int,
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def expected_zeros(
    scaled_means: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    total_counts: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    expected: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    n_genes: int,
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def expected_zeros(
    scaled_means: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    total_counts: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    expected: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    n_genes: int,
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def expected_zeros(
    scaled_means: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    total_counts: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    expected: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    n_genes: int,
    n_cells: int,
    stream: int = 0,
) -> None: ...
