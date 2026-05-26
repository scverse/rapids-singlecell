from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp

from rapids_singlecell._cuda import _pseudobulk_cuda as _pb

if TYPE_CHECKING:
    from collections.abc import Callable


def _to_f64_contig(X: cp.ndarray) -> cp.ndarray:
    return cp.ascontiguousarray(X.astype(cp.float64, copy=False))


def _check_paired(X: cp.ndarray, Y: cp.ndarray) -> None:
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(
            f"paired inputs must be 2D, got X.ndim={X.ndim}, Y.ndim={Y.ndim}"
        )
    if X.shape != Y.shape:
        raise ValueError(
            f"paired inputs must have identical shape, got {X.shape} vs {Y.shape}"
        )


def _check_pairwise(X: cp.ndarray, Y: cp.ndarray) -> None:
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(
            f"pairwise inputs must be 2D, got X.ndim={X.ndim}, Y.ndim={Y.ndim}"
        )
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            "pairwise inputs must have matching feature count, "
            f"got X.shape[1]={X.shape[1]} vs Y.shape[1]={Y.shape[1]}"
        )


def _paired_impl(
    X: cp.ndarray, Y: cp.ndarray, kernel: Callable[..., None]
) -> cp.ndarray:
    _check_paired(X, Y)
    X = _to_f64_contig(X)
    Y = _to_f64_contig(Y)
    n_pairs, n_features = X.shape
    out = cp.empty(n_pairs, dtype=cp.float64)
    if out.size == 0:
        return out
    kernel(
        X,
        Y,
        out=out,
        n_pairs=int(n_pairs),
        n_features=int(n_features),
        stream=cp.cuda.get_current_stream().ptr,
    )
    return out


def _pairwise_impl(
    X: cp.ndarray, Y: cp.ndarray, kernel: Callable[..., None]
) -> cp.ndarray:
    _check_pairwise(X, Y)
    X = _to_f64_contig(X)
    Y = _to_f64_contig(Y)
    n_x, n_features = X.shape
    n_y = Y.shape[0]
    out = cp.empty((n_x, n_y), dtype=cp.float64)
    if out.size == 0:
        return out
    kernel(
        X,
        Y,
        out=out,
        n_x=int(n_x),
        n_y=int(n_y),
        n_features=int(n_features),
        stream=cp.cuda.get_current_stream().ptr,
    )
    return out


def paired_squared(X: cp.ndarray, Y: cp.ndarray) -> cp.ndarray:
    return _paired_impl(X, Y, _pb.paired_squared)


def paired_abs_mean(X: cp.ndarray, Y: cp.ndarray) -> cp.ndarray:
    return _paired_impl(X, Y, _pb.paired_abs_mean)


def pairwise_squared(X: cp.ndarray, Y: cp.ndarray) -> cp.ndarray:
    return _pairwise_impl(X, Y, _pb.pairwise_squared)


def pairwise_abs_mean(X: cp.ndarray, Y: cp.ndarray) -> cp.ndarray:
    return _pairwise_impl(X, Y, _pb.pairwise_abs_mean)
