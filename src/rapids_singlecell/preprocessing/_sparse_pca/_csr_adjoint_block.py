"""CSR adjoint kernel for a block of 32 or 64 real right-hand sides."""

from __future__ import annotations

# A CTA stages a 64-cell by 64-feature CSR tile. Each bit records one
# cell for a feature, so warps visit only occupied entries in fixed order.
_block_adjoint = """
    const ptrdiff_t block = i / blockDim.x, tile = block % ntiles, group = block / ntiles;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int local_row = threadIdx.x / 4, sublane = threadIdx.x % 4;
    const ptrdiff_t first = nrows * group / groups, last = nrows * (group + 1) / groups;
    const ptrdiff_t col0 = tile * 64, col1 = min(col0 + 64, ptrdiff_t(ncols));
    __shared__ unsigned int active[2][64];
    __shared__ X values[64][64];
    double sums[8][(WIDTH / 32)] = {};
    for (ptrdiff_t row0 = first; row0 < last; row0 += 64) {
        if (threadIdx.x < 128) active[threadIdx.x / 64][threadIdx.x % 64] = 0;
        __syncthreads();
        const ptrdiff_t row = row0 + local_row;
        I begin = 0, end = 0;
        if (row < last && sublane == 0) {
            gene_tile_span<PACKED, false>(rows, cols, bounds, row, nrows, ncols, col0, col1, interval, begin, end);
        }
        begin = __shfl_sync(0xffffffff, begin, lane - sublane);
        end = __shfl_sync(0xffffffff, end, lane - sublane);
        for (ptrdiff_t j = ptrdiff_t(begin) + sublane; j < end; j += 4) {
            const ptrdiff_t f = cols[j] - col0;
            if (f >= col1 - col0) break;
            values[f][local_row] = data[j];
            atomicOr(&active[local_row / 32][f], 1u << (local_row % 32));
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int f = 8 * k + warp;
            if (f >= col1 - col0) continue;
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                for (unsigned mask = active[half][f]; mask; mask &= mask - 1) {
                    const int r = 32 * half + __ffs(mask) - 1;
                    const double value = values[f][r];
                    #pragma unroll
                    for (int q = 0; q < (WIDTH / 32); ++q) {
                        const ptrdiff_t col = rhs_start + 32 * q + lane;
                        if (col < rhs_width)
                            sums[k][q] += value * double(rhs[(row0 + r) * rhs_width + col]);
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const ptrdiff_t f = col0 + 8 * k + warp;
        if (f < ncols) {
            #pragma unroll
            for (int q = 0; q < (WIDTH / 32); ++q)
                out[(group * ncols + f) * WIDTH + 32 * q + lane] = sums[k][q];
        }
    }
    __syncthreads();
"""
