"""Lossless (expression, group) records for a feature-major Wilcoxon panel."""

from __future__ import annotations

import cupy as cp
import numpy as np

_tile = cp.RawKernel(  # type: ignore[attr-defined]
    r"""
extern "C" __global__ void transpose(const int* ptr, const int* cols, const unsigned* data,
        const unsigned short* labels, int rows, int tiles_per, int* prefix, unsigned mask, int* failed,
        int first, int width, int offset, unsigned* words, unsigned short* groups, unsigned* table, int bits) {
    bool write = bits >= 0;  // Negative counts entries, zero writes raw values, positive packs a dictionary.
    __shared__ unsigned tile[128][33];
    __shared__ unsigned char present[128][33];
    __shared__ int cursor[128];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, col0 = first + 128 * blockIdx.y;
    int c = threadIdx.x, k = (col0 + c) * gridDim.x + blockIdx.x;
    if (c < 128 && col0 + c < width) cursor[c] = write && k ? prefix[k - 1] - offset : 0;
    for (int t = 0; t < tiles_per; ++t) {
        long long row0 = ((long long)blockIdx.x * tiles_per + t) * 32;
        if (row0 >= rows) break;
        for (int i = threadIdx.x; i < 4096; i += blockDim.x) present[i / 32][i % 32] = 0;
        __syncthreads();
        for (int r = warp; r < 32 && row0 + r < rows; r += blockDim.x / 32) {
            int a = ptr[row0 + r], end = ptr[row0 + r + 1], b = end;
            if (!lane) while (a < b) {
                int m = a + (b - a) / 2;
                if (cols[m] < col0) a = m + 1; else b = m;
            }
            a = __shfl_sync(0xffffffff, a, 0);
            for (long long p = (long long)a + lane; p < end; p += 32) {
                int j = cols[p] - col0;
                if (j >= 128) break;
                present[j][r] = 1;
                if (write) tile[j][r] = data[p];
            }
        }
        __syncthreads();
        for (int j = warp; j < 128 && col0 + j < width; j += blockDim.x / 32) {
            unsigned present_lanes = __ballot_sync(0xffffffff, present[j][lane]);
            int base = __shfl_sync(0xffffffff, lane ? 0 : cursor[j], 0);
            if (!lane) cursor[j] = base + __popc(present_lanes);
            if (write && present[j][lane]) {
                int pos = base + __popc(present_lanes & ((1u << lane) - 1));
                unsigned key = tile[j][lane], g = labels[row0 + lane];
                if (!bits) { words[pos] = key; groups[pos] = g; continue; }
                unsigned h = (key ^ (key >> 16)) * 0x7feb352du, slot = 0;
                h ^= h >> 15;
                for (int probe = 0; key && probe < 128; ++probe) {
                    unsigned s = (h + probe) & mask;
                    if (!s) continue;  // Reserve slot zero for +0.
                    unsigned old = table[s];
                    if (!old) old = atomicCAS(table + s, 0u, key);
                    if (!old || old == key) { slot = s; break; }
                }
                if (key && !slot) atomicExch(failed, 1);
                words[pos] = (slot << bits) | g;
            }
        }
        __syncthreads();
    }
    if (!write && c < 128 && col0 + c < width) prefix[k] = cursor[c];
}""",
    "transpose",
)

_gather = cp.RawKernel(  # type: ignore[attr-defined]
    r"""
    extern "C" __global__ void gather(const unsigned* words, const unsigned* table,
            const unsigned short* codes, unsigned bits, const int* src, const int* dst,
            float* values, unsigned short* groups) {
        int col = blockIdx.y, start = src[col], stop = src[col + 1];
        for (long long i = start + (long long)blockIdx.x * blockDim.x + threadIdx.x;
             i < stop; i += gridDim.x * blockDim.x) {
            unsigned w = words[i];
            int target = dst[col] + (i - start);
            values[target] = __uint_as_float(bits ? table[w >> bits] : w);
            groups[target] = bits ? w & ((1u << bits) - 1) : codes[i];
        }
    }""",
    "gather",
)


def pack_dictionary(csr, labels, bounds):
    """Pack canonical CSR directly into independent gene shards with bounded scratch."""
    rows, cols = csr.shape
    if csr.dtype != cp.float32 or max(csr.nnz, rows, cols) > np.iinfo(np.int32).max or cols > 128 * 65535:
        return None
    if not csr.has_canonical_format:
        csr = csr.copy()
        csr.sum_duplicates()
    shards = list(zip(bounds[:-1], bounds[1:], strict=True))
    empty = cp.empty(0, cp.uint32)
    if not csr.nnz:
        return [(np.zeros(hi - lo + 1, np.int32), empty, empty.view(cp.uint16), 0) for lo, hi in shards]
    tiles = (rows + 31) // 32
    budget = max(1, min(2**25, csr.nnz) // cols)  # Cap the stripe prefix at 128 MiB.
    tiles_per = (tiles + budget - 1) // budget
    stripes = (tiles + tiles_per - 1) // tiles_per
    prefix = cp.empty((cols, stripes), cp.int32)
    bits = max(1, int(labels.max(initial=0)).bit_length())
    slots = 1 << min(32 - bits, 22, max(1, (csr.nnz // 64).bit_length()))
    indices = (cp.ascontiguousarray(v, dtype=cp.int32) for v in (csr.indptr, csr.indices))
    groups = cp.asarray(labels, dtype=cp.uint16, order="C")
    failed = cp.zeros(1, cp.int32)
    args = (*indices, cp.ascontiguousarray(csr.data), groups, np.int32(rows), np.int32(tiles_per), prefix)

    def transpose(lo, hi, offset, buffers, mode):
        grid = (stripes, (hi - lo + 127) // 128)
        span = np.array((lo, hi, offset), np.int32)
        _tile(grid, (256,), (*args, np.uint32(slots - 1), failed, *span, *buffers, np.int32(mode)))

    transpose(0, cols, 0, (empty, empty, empty), -1)
    cp.cumsum(prefix.ravel(), dtype=cp.int32, out=prefix.ravel())
    offsets = np.r_[np.int32(0), prefix[:, -1].get()]
    for mode in (bits, 0) if len(shards) * slots * 4 < csr.nnz * 2 else (0,):
        table = cp.zeros(slots, cp.uint32) if mode else empty
        pieces = []
        for lo, hi in shards:
            ptr = offsets[lo : hi + 1] - offsets[lo]
            words = cp.empty(int(ptr[-1]), cp.uint32)
            aux = table if mode else cp.empty(words.size, cp.uint16)
            transpose(lo, hi, offsets[lo], (words, aux, table), mode)
            pieces.append((ptr, words, aux, mode))
        if not mode or not failed.item():
            return pieces
        del pieces, words, aux, table  # Release the failed dictionary before allocating raw shards.


def assemble_records(blocks, lo, hi):
    source = np.stack([b[0][lo : hi + 1] for b in blocks])
    cumulative = np.cumsum(np.diff(source), axis=0, dtype=np.int64)
    ptr = np.r_[0, cumulative[-1].cumsum()]
    if ptr[-1] > np.iinfo(np.int32).max:
        raise MemoryError("Wilcoxon panel exceeds the native index range")
    dest = cp.asarray(ptr[:-1] + cumulative - np.diff(source), dtype=cp.int32)
    starts = source[:, :1].copy()
    offsets = cp.asarray(source - starts)
    values = cp.empty(int(ptr[-1]), cp.float32)
    codes = cp.empty(values.size, cp.uint16)
    stream = cp.cuda.get_current_stream()
    for i, (_, words, aux, bits) in enumerate(blocks):
        span = slice(starts[i, 0], source[i, -1])
        data = cp.asarray(words[span])
        lookup = cp.asarray(aux if bits else aux[span])
        _gather((32, hi - lo), (256,), (data, lookup, lookup, np.uint32(bits), offsets[i], dest[i], values, codes))
        if not isinstance(aux, cp.ndarray):
            stream.synchronize()  # Release each spilled lookup before staging the next.
    return ptr.astype(np.int32), values, codes
