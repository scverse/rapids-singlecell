"""Native array primitives shared by GPU algorithms."""

from cupy import ndarray

def axpy(alpha: ndarray, y: ndarray, x: ndarray, *, stream: int = 0) -> None: ...
def dense_sum(
    data: ndarray, output: ndarray, *, axis: int, square: bool, stream: int = 0
) -> None: ...
def clip_sums(
    data: ndarray,
    indices: ndarray,
    clip: ndarray,
    squares: ndarray,
    sums: ndarray,
    *,
    stream: int = 0,
) -> None: ...
def count_indices(indices: ndarray, counts: ndarray, *, stream: int = 0) -> None: ...
def duplicates_diff(
    rows: ndarray, cols: ndarray, output: ndarray, *, stream: int = 0
) -> None: ...
def duplicates_assign(
    src_rows: ndarray,
    src_cols: ndarray,
    indices: ndarray,
    rows: ndarray,
    cols: ndarray,
    *,
    stream: int = 0,
) -> None: ...
def scatter(
    data: ndarray,
    indices: ndarray,
    output: ndarray,
    *,
    squares: ndarray | None = None,
    count: bool = False,
    stream: int = 0,
) -> None: ...
