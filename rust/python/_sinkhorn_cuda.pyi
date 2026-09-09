"""Rust/cuda-oxide GPU entry points; streams and allocations are borrowed."""

from cupy import ndarray

def auto_eps(
    cost: ndarray,
    cost_off: ndarray,
    n: ndarray,
    m: ndarray,
    scale: float,
    floor: float,
    eps: ndarray,
    stream: int = 0,
) -> None: ...
def update_g(
    cost: ndarray,
    cost_off: ndarray,
    n: ndarray,
    m: ndarray,
    f: ndarray,
    f_off: ndarray,
    g: ndarray,
    g_off: ndarray,
    col2pair: ndarray,
    eps: ndarray,
    log_b: ndarray,
    conv: ndarray,
    omega: float = 1.0,
    stream: int = 0,
) -> None: ...
def update_f(
    cost: ndarray,
    cost_off: ndarray,
    m: ndarray,
    g: ndarray,
    g_off: ndarray,
    f: ndarray,
    f_off: ndarray,
    row2pair: ndarray,
    eps: ndarray,
    log_a: ndarray,
    conv: ndarray,
    omega: float = 1.0,
    stream: int = 0,
) -> None: ...
def check_convergence(
    f: ndarray,
    f_prev: ndarray,
    f_off: ndarray,
    n: ndarray,
    g: ndarray,
    g_prev: ndarray,
    g_off: ndarray,
    m: ndarray,
    tol: float,
    conv: ndarray,
    stream: int = 0,
) -> None: ...
def build_cost(
    emb: ndarray,
    cidx_l: ndarray,
    f_off: ndarray,
    cidx_r: ndarray,
    g_off: ndarray,
    n: ndarray,
    m: ndarray,
    cost_off: ndarray,
    tile_pair: ndarray,
    tile_i0: ndarray,
    tile_j0: ndarray,
    cost: ndarray,
    stream: int = 0,
) -> None: ...
