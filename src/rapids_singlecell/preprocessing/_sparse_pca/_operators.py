"""Sparse products for SVD with optional implicit mean centering."""

from __future__ import annotations

from functools import cache

import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import LinearOperator

from ._csr_adjoint import _CSRAdjoint

_accumulator = """
template <typename T> struct SvdAccumulator { using type = T; };
template <> struct SvdAccumulator<float> { using type = double; };
"""

_sum_squares = cp.ReductionKernel(
    "T x",
    "T out",
    "x * x",
    "a + b",
    "out = a",
    "0",
    "svd_sum_squares",
    reduce_type="typename SvdAccumulator<T>::type",
    preamble=_accumulator,
)

# Each row uses a warp, or a full block for matrices with long rows.
_csr_matvec = cp.ElementwiseKernel(
    "raw I indptr, raw J indices, raw X data, raw V rhs, int32 threads",
    "raw T out",
    """
    const ptrdiff_t row = i / threads;
    const int lane = threadIdx.x % 32;
    double sum = 0;
    for (ptrdiff_t j = ptrdiff_t(indptr[row]) + (i % threads); j < indptr[row + 1]; j += threads)
        sum += double(data[j]) * double(rhs[indices[j]]);
    for (int delta = 16; delta; delta /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (threads == 32) {
        if (!lane) out[row] = sum;
    } else {
        __shared__ double partial[8];
        if (!lane) partial[threadIdx.x / 32] = sum;
        __syncthreads();
        if (!threadIdx.x) {
            double total = 0;
            for (int w = 0; w < 8; ++w) total += partial[w];
            out[row] = total;
        }
        __syncthreads();
    }
    """,
    "svd_csr_matvec",
)


# Lanes read adjacent RHS columns. With two columns per lane, a single CSR
# index and value serve up to 64 RHS columns.
@cache
def _csr_matmat(columns_per_lane):
    return cp.ElementwiseKernel(
        "raw I indptr, raw J indices, raw X data, raw V rhs, int64 width, int64 tiles",
        "raw T out",
        """
        using Acc = typename SvdAccumulator<T>::type;
        const ptrdiff_t tile = i / blockDim.x, row = tile / tiles;
        const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32;
        const ptrdiff_t col = (tile % tiles) * (32 * COLS) + lane;
        const ptrdiff_t stop = indptr[row + 1];
        Acc sums[COLS] = {};
        if (col < width) {
            for (ptrdiff_t base = ptrdiff_t(indptr[row]) + warp; base < stop; base += 8 * warps) {
                T group[COLS] = {};
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const ptrdiff_t j = base + k * warps;
                    if (j < stop) {
                        const ptrdiff_t off = ptrdiff_t(indices[j]) * width + col;
                        const T value = T(data[j]);
                        #pragma unroll
                        for (int c = 0; c < COLS; ++c)
                            if (col + 32 * c < width) group[c] += value * T(rhs[off + 32 * c]);
                    }
                }
                #pragma unroll
                for (int c = 0; c < COLS; ++c) sums[c] += group[c];
            }
        }
        __shared__ Acc partial[COLS][16][32];
        #pragma unroll
        for (int c = 0; c < COLS; ++c) partial[c][warp][lane] = sums[c];
        __syncthreads();
        if (warp == 0) {
            #pragma unroll
            for (int c = 0; c < COLS; ++c) {
                if (col + 32 * c < width) {
                    Acc total = 0;
                    for (int w = 0; w < warps; ++w) total += partial[c][w][lane];
                    out[row * width + col + 32 * c] = total;
                }
            }
        }
        __syncthreads();
        """,
        f"svd_csr_matmat_{columns_per_lane}",
        preamble=_accumulator + f"#define COLS {columns_per_lane}\n",
    )


_sum_runs = cp.ElementwiseKernel(
    "raw T data, I start, I stop",
    "T value",
    "value = data[start]; for (I j = start + 1; j < stop; ++j) value += data[j];",
    "svd_sum_duplicates",
)


def _canonical_csr(X):
    """Coalesce duplicate coordinates in input order, including before computing moments."""
    if X.format != "coo":
        X = X.tocsr()
        if X.has_canonical_format:
            return X
    coo = X.tocoo()
    if not coo.nnz:
        return coo.tocsr()
    order = cp.lexsort(cp.stack((coo.col, coo.row)))
    rows, cols = coo.row[order], coo.col[order]
    boundaries = cp.flatnonzero((rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])) + 1
    starts = cp.r_[0, boundaries]
    values = _sum_runs(coo.data[order], starts, cp.r_[boundaries, coo.nnz])
    indptr = cp.searchsorted(rows[starts], cp.arange(X.shape[0] + 1))
    canonical = csr_matrix((values, cols[starts], indptr), shape=X.shape)
    canonical.has_canonical_format = True
    return canonical


def _csr_dot(X, V):
    """Compute a bitwise reproducible CSR vector or block product."""
    if V.ndim not in (1, 2) or V.shape[0] != X.shape[1]:
        raise ValueError("dimension mismatch")
    dtype = cp.result_type(X.dtype, V.dtype)
    out = cp.empty((X.shape[0], *V.shape[1:]), dtype=dtype)
    width = 1 if V.ndim == 1 else V.shape[1]
    V = cp.ascontiguousarray(V)
    long_rows = X.nnz >= 512 * X.shape[0]
    if width == 1 and dtype in (cp.float32, cp.float64):
        threads = 256 if long_rows else 32
        _csr_matvec(X.indptr, X.indices, X.data, V, threads, out, size=X.shape[0] * threads, block_size=256)
    else:
        columns_per_lane = 2 if width > 32 else 1
        tile_width = 32 * columns_per_lane
        tiles = (width + tile_width - 1) // tile_width
        threads = 512 if long_rows else 128
        _csr_matmat(columns_per_lane)(
            X.indptr, X.indices, X.data, V, width, tiles, out, size=X.shape[0] * tiles * threads, block_size=threads
        )
    return out


def mean_centered_operator(X, mean: cp.ndarray | None = None, *, adjoint=None) -> LinearOperator:
    """Apply X and its adjoint from the same CSR data, optionally subtracting the column means."""
    X = _canonical_csr(X)
    if adjoint is None:
        adjoint = _CSRAdjoint(X)
    adjoint_mean = mean.conj() if mean is not None else None

    def matmat(V):
        result = _csr_dot(X, V)
        if mean is not None:
            result = result - (mean @ V)
        return result

    def rmatmat(V):
        result = adjoint.dot(V)
        if adjoint_mean is not None:
            offset = adjoint_mean[:, None] if V.ndim == 2 else adjoint_mean
            sum_dtype = cp.float64 if V.dtype == cp.float32 else V.dtype
            sums = V.sum(axis=0, dtype=sum_dtype).astype(V.dtype, copy=False)
            result = result - offset * sums
        return result

    return LinearOperator(
        shape=X.shape,
        matvec=matmat,
        rmatvec=rmatmat,
        matmat=matmat,
        rmatmat=rmatmat,
        dtype=X.dtype,
    )
