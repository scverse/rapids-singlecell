"""Deterministic adjoint products directly from canonical CSR rows."""

from __future__ import annotations

from functools import cache

import cupy as cp

from ._csr_adjoint_block import _block_adjoint

_TILE = 256
_THREADS = 64
_PARTIAL_BYTES = 128 << 20

_lower_bound = (
    f"#define BOUNDS_TILE {_TILE}\n"
    + """
template <typename Columns, typename I>
__device__ I gene_lower_bound(const Columns& cols, I lo, I hi, ptrdiff_t key) {
    while (lo < hi) {
        const I mid = lo + (hi - lo) / 2;
        if (cols[mid] < key) lo = mid + 1; else hi = mid;
    }
    return lo;
}
"""
)

_span_lookup = (
    _lower_bound
    + """
// With exact=false the caller stops scanning at col1 within the enclosing interval.
template <bool packed, bool exact, typename Rows, typename Columns, typename Bounds, typename I>
__device__ void gene_tile_span(const Rows& rows, const Columns& cols, const Bounds& bounds,
        ptrdiff_t r, ptrdiff_t nrows, ptrdiff_t ncols, ptrdiff_t col0, ptrdiff_t col1,
        ptrdiff_t interval, I& a, I& b) {
    const I start = rows[r], stop = rows[r + 1];
    if (packed) {
        const ptrdiff_t tile = col0 / BOUNDS_TILE;
        const unsigned word = bounds[(tile / 4) * nrows + r];
        const I base = start + (word & 4095);
        const int k = tile % 4;
        const int shift = 12 + 5*k;
        const unsigned lo = k ? (word >> (shift - 5)) & 31 : 0;
        const unsigned hi = (word >> shift) & 31;
        a = base + lo;
        b = base + hi;
        if (hi == 31) {
            a = gene_lower_bound(cols, base, stop, col0);
            b = exact ? gene_lower_bound(cols, a, stop, col1) : stop;
            return;
        }
        interval = BOUNDS_TILE;
    } else {
        const ptrdiff_t bucket = col0 / interval;
        a = bucket ? start + bounds[(bucket - 1)*nrows + r] : start;
        b = (bucket + 1)*interval < ncols ? start + bounds[bucket*nrows + r] : stop;
    }
    if (col0 % interval) a = gene_lower_bound(cols, a, b, col0);
    if (exact && col1 % interval && col1 < ncols) b = gene_lower_bound(cols, a, b, col1);
}
"""
)


# The search table contains row-relative positions, never matrix values.
_bounds_kernel = cp.ElementwiseKernel(  # type: ignore[attr-defined]  # cupy-stubs
    "raw I rows, raw J cols, int64 nrows, int64 interval, bool packed",
    "B bound",
    """
    const ptrdiff_t row = i % nrows, col = (i / nrows) * interval;
    const I start = rows[row], stop = rows[row + 1];
    if (packed) {
        const I base = gene_lower_bound(cols, start, stop, col);
        unsigned word = base - start;
        I pos = base;
        for (int k = 0; k < 4; ++k) {
            pos = gene_lower_bound(cols, pos, stop, col + (k + 1)*BOUNDS_TILE);
            word |= unsigned(min(I(31), pos - base)) << (12 + 5*k);
        }
        bound = word;
    } else {
        bound = gene_lower_bound(cols, start, stop, col + interval) - start;
    }
    """,
    "svd_csr_gene_bounds",
    preamble=_lower_bound,
)


_first_row = cp.ElementwiseKernel(  # type: ignore[attr-defined]  # cupy-stubs
    "raw I rows, raw J cols, raw X data",
    "X reference",
    """
    const I stop = rows[1], j = gene_lower_bound(cols, I(0), stop, i);
    reference = j < stop && cols[j] == i ? data[j] : X(0);
    """,
    "svd_csr_first_row",
    preamble=_lower_bound,
)

_column_mean = cp.ElementwiseKernel(  # type: ignore[attr-defined]  # cupy-stubs
    "float64 shifted_sum, float64 count, X reference, int64 nrows",
    "float64 mean, float64 offset, float64 low, float64 zero_energy, int32 sensitive",
    """
    const double missing = nrows - count;
    const double shifted = fma(-missing, double(reference), shifted_sum);
    offset = shifted / nrows;
    if (missing > count) {
        const double total = fma(count, double(reference), shifted_sum);
        mean = total / nrows;
        low = fma(-double(nrows), mean, total) / nrows;
    } else {
        mean = double(reference) + offset;
        const double y = mean - double(reference);
        // Recover the rounding errors in both the addition and the division.
        low = (double(reference) - (mean - y)) + (offset - y) + fma(-double(nrows), offset, shifted) / nrows;
    }
    zero_energy = (missing * mean) * mean;
    // Missing zeros provide an ordinary-scale deviation for sparse columns.
    // A fully populated column can lose half its input precision when centering.
    const double tol = sizeof(X) == 4 ? sqrt(0x1p-23) : 0x1p-26;
    sensitive = count == nrows && reference != 0 && fabs(offset) <= tol * fabs(double(reference));
    """,
    "svd_csr_column_mean",
)

_centered_energy = cp.ElementwiseKernel(  # type: ignore[attr-defined]  # cupy-stubs
    "raw X data, raw I cols, raw X reference, raw float64 offset, raw float64 zeros, "
    "int64 nnz, int64 ncols, int64 stride",
    "float64 partial, raw int32 sensitive",
    """
    double energy = 0;
    for (ptrdiff_t col = i; col < ncols; col += stride) energy += zeros[col];
    for (ptrdiff_t j = i; j < nnz; j += stride) {
        const I col = cols[j];
        const double delta = (double(data[j]) - double(reference[col])) - offset[col];
        // Implicitly subtracting two products loses at least half of double
        // precision when this entry nearly equals its nonzero column mean.
        if (data[j] != 0 && fabs(delta) <= 0x1p-26 * fabs(double(data[j])))
            atomicExch(&sensitive[col], 1);
        energy += delta * delta;
    }
    partial = energy;
    """,
    "svd_csr_centered_energy",
)

_centered_panel = cp.ElementwiseKernel(  # type: ignore[attr-defined]  # cupy-stubs
    "raw I rows, raw J cols, raw X data, raw K selected, raw float64 mean, raw float64 low, int64 row0, int64 width",
    "T value",
    """
    const ptrdiff_t row = row0 + i / width, k = i % width;
    const K col = selected[k];
    const I end = rows[row + 1], j = gene_lower_bound(cols, rows[row], end, col);
    const T x = j < end && cols[j] == col ? T(data[j]) : T(0);
    value = (x - T(mean[k])) - T(low[k]);
    """,
    "svd_centered_panel",
    preamble=_lower_bound,
)

_screen_centered = cp.ElementwiseKernel(  # type: ignore[attr-defined]  # cupy-stubs
    "X x, raw I cols, raw float64 mean, raw float64 low, raw M rounded, int64 dense_width",
    "raw int32 sensitive",
    """
    const ptrdiff_t col = dense_width ? i % dense_width : ptrdiff_t(cols[i]);
    const double delta = (double(x) - mean[col]) - low[col];
    const double mean_error = (mean[col] - double(rounded[col])) + low[col];
    const double tol = sizeof(M) == 4 ? sqrt(0x1p-23) : 0x1p-26;
    if (x != 0 && (fabs(delta) <= 0x1p-26 * fabs(double(x)) || fabs(mean_error) > tol * fabs(delta)))
        atomicExch(&sensitive[col], 1);
    """,
    "svd_screen_centered",
)


def _select_center(sensitive, mean, low):
    selected = cp.flatnonzero(sensitive)
    if selected.size:
        return selected, mean[selected], low[selected]
    return None


# Each warp owns its counters. When lanes handle different rows, they take
# turns updating a shared column in ascending lane order.
@cache
def _adjoint_kernel(width, lanes, *, moments, packed, wide_block):
    return cp.ElementwiseKernel(
        "raw I rows, raw J cols, raw X data, raw B bounds, raw V rhs, int64 nrows, int64 ncols, "
        "int64 ntiles, int64 interval, int64 groups, int64 rhs_width, int64 rhs_start",
        "raw A out",
        _block_adjoint
        if wide_block
        else """
        const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
        const ptrdiff_t block = i / blockDim.x, group = block / ntiles, tile = block % ntiles;
        const ptrdiff_t col0 = tile * TILE, col1 = min(col0 + TILE, ptrdiff_t(ncols));
        const ptrdiff_t start = nrows * group / groups, stop = nrows * (group + 1) / groups;
        __shared__ A sums[WARPS][TILE][WIDTH];
        for (int f = lane; f < TILE; f += 32)
            for (int k = 0; k < WIDTH; ++k) sums[warp][f][k] = A(0);
        __syncwarp();
        for (ptrdiff_t first = start + warp * 32; first < stop; first += WARPS * 32) {
            const ptrdiff_t r = first + lane;
            I a = 0, b = 0;
            if (r < stop) {
                gene_tile_span<PACKED, true>(rows, cols, bounds, r, nrows, ncols, col0, col1, interval, a, b);
            }
            for (int t = 0; t < 32 && first + t < stop; t += 32 / LANES) {
                const int slot = t + lane / LANES;
                const ptrdiff_t begin = __shfl_sync(0xffffffff, a, slot);
                const ptrdiff_t end = __shfl_sync(0xffffffff, b, slot);
                A weights[WIDTH];
                if (LANES < 32 && !MOMENTS && first + slot < stop)
                    for (int k = 0; k < WIDTH; ++k)
                        weights[k] = rhs_start+k < rhs_width ? A(rhs[(first+slot)*rhs_width+rhs_start+k]) : A(0);
                auto add = [&](int f, A value) {
                    #if MOMENTS
                    sums[warp][f][0] += value - A(rhs[col0 + f]);
                    sums[warp][f][1] += A(1);
                    #else
                    #pragma unroll
                    for (int k = 0; k < WIDTH; ++k)
                        if (rhs_start+k < rhs_width)
                            sums[warp][f][k] += value * (LANES < 32 ? weights[k] :
                                                          A(rhs[(first+slot)*rhs_width+rhs_start+k]));
                    #endif
                };
                if (LANES == 32) {
                    for (ptrdiff_t j = begin + lane; j < end; j += 32)
                        add(cols[j] - col0, A(data[j]));
                } else {
                    for (ptrdiff_t j = begin + lane % LANES; __any_sync(0xffffffff, j < end); j += LANES) {
                        const int f = j < end ? int(cols[j] - col0) : TILE;
                        const unsigned peers = __match_any_sync(0xffffffff, f);
                        if (j < end) {
                            const A value = A(data[j]);
                            for (unsigned rest = peers; rest; rest &= rest - 1) {
                                if (lane == __ffs(rest) - 1) add(f, value);
                                __syncwarp(peers);
                            }
                        }
                        __syncwarp();
                    }
                }
                __syncwarp();
            }
        }
        __syncthreads();
        for (int f = threadIdx.x; f < col1 - col0; f += blockDim.x)
            for (int k = 0; k < WIDTH; ++k) {
                A sum = 0;
                for (int w=0; w<WARPS; ++w) sum += sums[w][f][k];
                out[(group*ncols+col0+f)*WIDTH+k] = sum;
            }
        __syncthreads();
        """,
        f"svd_csr_adjoint_{width}_{lanes}_{int(moments)}_{int(packed)}_{int(wide_block)}",
        preamble=_span_lookup
        + f"#define TILE {_TILE}\n#define WIDTH {width}\n#define LANES {lanes}\n"
        + f"#define WARPS {_THREADS // 32}\n#define MOMENTS {int(moments)}\n#define PACKED {int(packed)}",
    )


class _CSRAdjoint:
    """Apply the conjugate transpose and compute column moments from canonical CSR."""

    def __init__(self, X):
        self.X = X
        nrows, ncols = X.shape
        self.interval = _TILE
        # Widen the search intervals if a fine table would exceed 1/16 of the
        # CSR storage or 1 GiB. A binary search handles subdivisions on demand.
        max_offset = int(cp.diff(X.indptr).max()) if ncols > 65535 and nrows else max(ncols, 1)
        offset_dtype = cp.min_scalar_type(max_offset)
        limit = min(1 << 30, (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) // 16)
        n_bounds = (ncols + 4 * _TILE - 1) // (4 * _TILE)
        # Four adjacent tiles share a 12-bit row offset and four 5-bit prefixes.
        # A saturated prefix uses binary search for that tile.
        full_bytes = nrows * max(0, (ncols - 1) // _TILE) * offset_dtype.itemsize
        self.packed = max_offset < 4096 and nrows * n_bounds * 4 <= limit < full_bytes
        if self.packed:
            self.interval = 4 * _TILE
            offset_dtype = cp.uint32
        else:
            while nrows * ((ncols - 1) // self.interval) * offset_dtype.itemsize > limit:
                self.interval *= 2
            n_bounds = max(0, (ncols - 1) // self.interval)
        self.bounds = cp.empty((n_bounds, nrows), dtype=offset_dtype)
        if self.bounds.size:
            _bounds_kernel(X.indptr, X.indices, nrows, self.interval, self.packed, self.bounds)

    def _run(self, rhs, *, moments):
        X = self.X
        nrows, ncols = X.shape
        cols = 2 if moments else (1 if rhs.ndim == 1 else rhs.shape[1])
        dtype = cp.dtype(cp.float64) if moments else cp.result_type(X.dtype, rhs.dtype)
        acc_dtype = cp.dtype(cp.complex128 if dtype.kind == "c" else cp.float64)
        out = cp.empty((ncols, cols), dtype=dtype)
        if not out.size or not nrows:
            out.fill(0)
            return out
        max_width = _PARTIAL_BYTES // (ncols * acc_dtype.itemsize)
        row_tiles = nrows * ((ncols + _TILE - 1) // _TILE)
        sparse_tiles = X.nnz <= 4 * row_tiles
        # Stage the tile when at least three quarters of a warp has work.
        # Dense feature panels amortize staging with as few as 16 columns.
        wide_block = (
            dtype.kind == "f" and max_width >= 32 and (cols >= 24 or (cols >= 16 and 8 * X.nnz >= nrows * ncols))
        )
        if wide_block:
            width = 64 if cols > 32 and max_width >= 64 else 32
            tile, threads = 64, 256
        else:
            width = min(2 if moments else 4, cols)
            tile, threads = _TILE, _THREADS
        groups = min((nrows + 63) // 64, max(1, max_width // width))
        ntiles = (ncols + tile - 1) // tile
        parts = cp.empty((groups, ncols, width), dtype=acc_dtype)
        rhs = cp.ascontiguousarray(rhs)
        # Share a warp across rows when a tile has few entries per row.
        lanes = 32
        if sparse_tiles:
            lanes = 1 if width > 1 else (4 if X.nnz > 2 * row_tiles else 2)
        kernel = _adjoint_kernel(width, lanes, moments=moments, packed=self.packed, wide_block=wide_block)
        for start in range(0, cols, width):
            kernel(
                X.indptr,
                X.indices,
                X.data,
                self.bounds,
                rhs,
                nrows,
                ncols,
                ntiles,
                self.interval,
                groups,
                cols,
                start,
                parts,
                size=groups * ntiles * threads,
                block_size=threads,
            )
            count = min(width, cols - start)
            if cols == width:
                parts.sum(axis=0, dtype=acc_dtype, out=out)
            else:
                out[:, start : start + count] = parts.sum(axis=0)[:, :count]
        return out

    def dot(self, V):
        if V.ndim not in (1, 2) or V.shape[0] != self.X.shape[0]:
            raise ValueError("dimension mismatch")
        if self.X.dtype.kind == "c":
            out = self._run(V.conj(), moments=False).conj()
        else:
            out = self._run(V, moments=False)
        return out[:, 0] if V.ndim == 1 else out

    def centered_stats(self):
        X = self.X
        nrows, ncols = X.shape
        self.precise_center = None
        self.center = None
        if not nrows:
            return cp.full(ncols, cp.nan, dtype=cp.float64), cp.asarray(0, dtype=cp.float64)
        # Center relative to an observed row to retain small differences even
        # when a column's mean lies between two representable values.
        reference = _first_row(X.indptr, X.indices, X.data, size=ncols)
        sums, counts = self._run(reference, moments=True).T
        mean, offset, low, zeros, sensitive = _column_mean(sums, counts, reference, nrows)
        # Coalesced partial sums bound the reduction workspace to 2 MiB.
        stride = min(1 << 18, (max(X.nnz, ncols) + 15) // 16)
        energy = cp.empty(stride, dtype=cp.float64)
        _centered_energy(X.data, X.indices, reference, offset, zeros, X.nnz, ncols, stride, energy, sensitive)
        self.center = mean, low, sensitive
        self.precise_center = _select_center(sensitive, mean, low)
        return mean, energy.sum()
