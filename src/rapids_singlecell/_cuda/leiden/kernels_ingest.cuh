#pragma once

// Ingest. Before the driver, Python runs I1 ingest_check (error flags, row
// order, nnz_c, wmax, order-free symmetry fingerprints) and, for non-canonical
// input, I1c canonicalize (rows stably sorted, duplicates summed in fp64 in
// stored order). The driver runs per scale I2 quantize_degrees (level-0 device
// copies, W0q where needed, k_hat, 2m_hat, degree classes) and I4 (the replica
// union). An entry (v, u) counts iff u != v and its value (the weight, or 1.0f
// for every entry != 0 without use_weights) is > 0.

#include <mutex>
#include <type_traits>

#include "arena.cuh"

namespace leiden {

using namespace nb::literals;

constexpr u64 kUnitBits = 0x3F800000ull;  // IEEE bits of 1.0f

template <typename WI>
struct EntryValue {
    const WI* data;
    bool weighted;  // use_weights (else every entry != 0 has the value 1.0f)
    __device__ __forceinline__ bool positive(i64 j) const {
        return weighted ? data[j] > WI(0) : data[j] != WI(0);
    }
    __device__ __forceinline__ u64 bits(i64 j) const {
        return weighted ? weight_bits(data[j]) : kUnitBits;
    }
    // max(1, rint(w 2^s)), exact; 0 if the entry is not positive
    __device__ __forceinline__ i64 quantized(i64 j, int s) const {
        if (weighted) return quantize_scaled(static_cast<double>(data[j]), s);
        return positive(j) ? quantize_scaled(1.0, s) : 0;
    }
};

struct ScanPartial {
    u32 flags = 0;
    u64 wmax = 0;
    u64 h_fwd = 0;
    u64 h_rev = 0;
    i64 counted = 0;
};

__device__ __forceinline__ void flush_scan_partial(ScanPartial p,
                                                   Control* ctl) {
    __shared__ u32 s_flags[kWarpsPerBlock];
    __shared__ u64 s_wmax[kWarpsPerBlock], s_hf[kWarpsPerBlock],
        s_hr[kWarpsPerBlock];
    __shared__ i64 s_cnt[kWarpsPerBlock];
    p.flags = warp_or(p.flags);
    p.wmax = warp_max(p.wmax);
    p.h_fwd = warp_sum(p.h_fwd);
    p.h_rev = warp_sum(p.h_rev);
    p.counted = warp_sum(p.counted);
    const int lane = threadIdx.x & (kWarp - 1), w = threadIdx.x / kWarp;
    if (lane == 0) {
        s_flags[w] = p.flags;
        s_wmax[w] = p.wmax;
        s_hf[w] = p.h_fwd;
        s_hr[w] = p.h_rev;
        s_cnt[w] = p.counted;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        ScanPartial b;
        for (int i = 0; i < kWarpsPerBlock; ++i) {
            b.flags |= s_flags[i];
            b.wmax = s_wmax[i] > b.wmax ? s_wmax[i] : b.wmax;
            b.h_fwd += s_hf[i];
            b.h_rev += s_hr[i];
            b.counted += s_cnt[i];
        }
        if (b.flags) atomicOr(&ctl->flags, b.flags);
        if (b.wmax) atomicMax(&ctl->wmax_bits, b.wmax);
        atomicAdd(&ctl->h_fwd, b.h_fwd);
        atomicAdd(&ctl->h_rev, b.h_rev);
        atomicAdd(reinterpret_cast<u64*>(&ctl->n_counted),
                  static_cast<u64>(b.counted));
    }
}

template <typename IP, typename IX, typename WI>
__global__ void ingest_scan_kernel(const IP* __restrict__ indptr,
                                   const IX* __restrict__ indices,
                                   EntryValue<WI> val, i64 n, i64 nnz,
                                   Control* ctl) {
    const int lane = threadIdx.x & (kWarp - 1);
    const i64 warp0 =
        static_cast<i64>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
    const i64 nwarps = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
    ScanPartial p;
    for (i64 v = warp0; v < n; v += nwarps) {
        const i64 b = static_cast<i64>(indptr[v]);
        const i64 e = static_cast<i64>(indptr[v + 1]);
        if (lane == 0) {
            if (e < b || (v == 0 && b != 0) || (v == n - 1 && e != nnz))
                p.flags |= kFlagBadIndptr;
            if (e - b >= (1ll << 31)) p.flags |= kFlagRowTooLong;
        }
        if (e < b || b < 0 || e > nnz) continue;  // warp-uniform guard
        for (i64 j = b + lane; j < e; j += kWarp) {
            const i64 u = static_cast<i64>(indices[j]);
            if (u < 0 || u >= n) {
                p.flags |= kFlagBadIndex;
                continue;
            }
            if (j > b && static_cast<i64>(indices[j - 1]) >= u)
                p.flags |= kFlagNonCanonical;
            if (val.weighted) {
                const WI w = val.data[j];
                if (!isfinite(w))
                    p.flags |= kFlagNonFinite;
                else if (w < WI(0))
                    p.flags |= kFlagNegative;
            }
            if (u == v) continue;
            if (!val.positive(j)) {
                if (!val.weighted) p.flags |= kFlagUnitZero;
                continue;
            }
            const u64 bits = val.bits(j);
            if (bits > p.wmax) p.wmax = bits;
            fingerprint_terms(v, u, bits, p.h_fwd, p.h_rev);
            ++p.counted;
        }
    }
    flush_scan_partial(p, ctl);
}

struct IngestInfo {
    u32 flags = 0;
    u64 wmax_bits = 0;
    double wmax = 0.0;  // max counted value (1.0 for unit weights)
    i64 n_counted = 0;  // nnz_c
    u64 h_fwd = 0, h_rev = 0;
};

inline double value_from_bits(u64 bits, bool fp32) {
    if (fp32) {
        const u32 b = static_cast<u32>(bits);
        float f;
        std::memcpy(&f, &b, sizeof(f));
        return static_cast<double>(f);
    }
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}

inline void throw_structure_errors(u32 flags) {
    if (flags & kFlagBadIndptr)
        throw std::invalid_argument(
            "leiden: indptr must start at 0, be non-decreasing and end at nnz");
    if (flags & kFlagRowTooLong)
        throw std::invalid_argument(
            "leiden: rows with >= 2^31 stored entries are not supported");
    if (flags & kFlagBadIndex)
        throw std::invalid_argument("leiden: column index out of range [0, n)");
}

// I1 (one sync): structural errors raise; weight errors and the fix-up
// conditions come back as flags.
template <typename IP, typename IX, typename WI>
IngestInfo run_ingest_check(const IP* indptr, const IX* indices, const WI* data,
                            i64 n, i64 nnz, bool use_weights, Control* ctl,
                            void* pinned, cudaStream_t s) {
    if (n < 0 || n >= kMaxVertices)
        throw std::invalid_argument("leiden: n must be in [0, 2^30)");
    if (n == 0 && nnz != 0)
        throw std::invalid_argument("leiden: indptr must end at nnz");
    const EntryValue<WI> val{data, use_weights};
    clear_control(ctl, s);
    if (n > 0) {
        ingest_scan_kernel<IP, IX, WI>
            <<<grid_occ(ingest_scan_kernel<IP, IX, WI>, n, kWarpsPerBlock),
               kBlock, 0, s>>>(indptr, indices, val, n, nnz, ctl);
        CUDA_CHECK_LAST_ERROR(ingest_scan_kernel);
    }
    const Control h = read_control(ctl, s, pinned);
    IngestInfo info;
    info.flags = h.flags;
    info.wmax_bits = h.wmax_bits;
    info.n_counted = h.n_counted;
    info.h_fwd = h.h_fwd;
    info.h_rev = h.h_rev;
    throw_structure_errors(info.flags);
    if (info.n_counted >= kMaxCountedEntries)
        throw std::invalid_argument(
            "leiden: graphs with >= 2^40 counted entries are not supported");
    if (info.h_fwd != info.h_rev) info.flags |= kFlagAsymmetric;
    info.wmax = value_from_bits(info.wmax_bits,
                                !val.weighted || std::is_same_v<WI, float>);
    return info;
}

// I1c: chunks of whole rows (<= kCanonChunk entries, or one longer row) keep
// CUB calls below 2^31 items. A stable radix sort of (local row << 30 | column)
// keeps duplicates in stored order; they are summed in fp64 in that order.
constexpr i64 kCanonChunk = 1ll << 26;
constexpr int kColumnBits = 30;  // n < 2^30

inline i64 canon_chunk_cap(i64 nnz, i64 max_row_nnz, i64 chunk) {
    return std::max<i64>(std::min<i64>(nnz, chunk), max_row_nnz);
}

struct CanonBufs {
    u64* keys_a = nullptr;  // [cap]
    u64* keys_b = nullptr;
    int* vals_a = nullptr;  // [cap] entry index within the chunk
    int* vals_b = nullptr;
    i64* loff = nullptr;   // [n + 2] distinct columns per row of a chunk
    i64* loff2 = nullptr;  // [n + 2] chunk-local output offsets
    i64* base = nullptr;   // [1] output entries written so far
    void* cub = nullptr;
    std::size_t cub_bytes = 0;

    static std::size_t cub_temp(i64 n, i64 cap) {
        std::size_t b = 0, best = 0;
        if (cap > 0) {
            u64* k = nullptr;
            int* v = nullptr;
            cub::DoubleBuffer<u64> keys(k, k);
            cub::DoubleBuffer<int> vals(v, v);
            cuda_check(cub::DeviceRadixSort::SortPairs(
                           nullptr, b, keys, vals,
                           cub_items(cap, "canonical sort"), 0, 64),
                       "leiden: cub canonical sort query");
            best = b;
        }
        b = 0;
        i64* off = nullptr;
        cuda_check(
            cub::DeviceScan::ExclusiveSum(nullptr, b, off, off,
                                          cub_items(n + 2, "canonical scan")),
            "leiden: cub canonical scan query");
        return std::max(best, b);
    }

    void carve(Carver& c, i64 n, i64 cap) {
        keys_a = c.take<u64>(cap);
        keys_b = c.take<u64>(cap);
        vals_a = c.take<int>(cap);
        vals_b = c.take<int>(cap);
        loff = c.take<i64>(n + 2);
        loff2 = c.take<i64>(n + 2);
        base = c.take<i64>(1);
        cub_bytes = cub_temp(n, cap);
        cub = c.bytes(cub_bytes);
    }
};

inline std::size_t canonicalize_scratch_bytes(i64 n, i64 nnz, i64 max_row_nnz,
                                              i64 chunk = kCanonChunk) {
    if (n < 0 || nnz < 0 || max_row_nnz < 0)
        throw std::invalid_argument("n, nnz and max_row_nnz must be >= 0");
    if (max_row_nnz >= (1ll << 31))
        throw std::invalid_argument(
            "leiden: rows with >= 2^31 stored entries are not supported");
    Carver c;
    CanonBufs b;
    if (chunk < 1 || chunk > kCanonChunk)
        throw std::invalid_argument("chunk_entries must be in [1, 2^26]");
    b.carve(c, n, canon_chunk_cap(nnz, max_row_nnz, chunk));
    return c.used();
}

template <typename IP, typename IX>
__global__ void canon_keys_kernel(const IP* __restrict__ indptr,
                                  const IX* __restrict__ indices, i64 r0,
                                  i64 r1, i64 e0, u64* __restrict__ keys,
                                  int* __restrict__ vals) {
    const int lane = threadIdx.x & (kWarp - 1);
    const i64 warp0 =
        static_cast<i64>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
    const i64 nwarps = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
    constexpr u64 kColMask = (1ull << kColumnBits) - 1ull;
    for (i64 v = r0 + warp0; v < r1; v += nwarps) {
        const i64 b = static_cast<i64>(indptr[v]);
        const i64 e = static_cast<i64>(indptr[v + 1]);
        for (i64 j = b + lane; j < e; j += kWarp) {
            keys[j - e0] = (static_cast<u64>(v - r0) << kColumnBits) |
                           (static_cast<u64>(indices[j]) & kColMask);
            vals[j - e0] = static_cast<int>(j - e0);
        }
    }
}

template <typename IP>
__global__ void canon_count_kernel(const IP* __restrict__ indptr, i64 r0,
                                   i64 rows, i64 e0,
                                   const u64* __restrict__ keys,
                                   i64* __restrict__ loff) {
    const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
    for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < rows; i += stride) {
        const i64 b = static_cast<i64>(indptr[r0 + i]) - e0;
        const i64 e = static_cast<i64>(indptr[r0 + i + 1]) - e0;
        i64 k = 0;
        for (i64 j = b; j < e; ++j) k += (j == b || keys[j] != keys[j - 1]);
        loff[i] = k;
        if (i == rows - 1) loff[rows] = 0;
    }
}

// Thread per row: the canonical row at base + loff2[i].
template <typename IP, typename WI>
__global__ void canon_write_kernel(
    const IP* __restrict__ indptr, i64 r0, i64 rows, i64 e0,
    const u64* __restrict__ keys, const int* __restrict__ vals,
    const WI* __restrict__ data, const i64* __restrict__ loff2,
    const i64* __restrict__ base, i64* __restrict__ indptr_out,
    int* __restrict__ idx_out, double* __restrict__ data_out) {
    constexpr u64 kColMask = (1ull << kColumnBits) - 1ull;
    const i64 b0 = base[0];
    const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
    for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < rows; i += stride) {
        const i64 b = static_cast<i64>(indptr[r0 + i]) - e0;
        const i64 e = static_cast<i64>(indptr[r0 + i + 1]) - e0;
        i64 out = b0 + loff2[i];
        indptr_out[r0 + i] = out;
        i64 j = b;
        while (j < e) {
            const u64 key = keys[j];
            const i64 src = e0 + vals[j];
            double sum = data ? static_cast<double>(data[src]) : 1.0;
            for (++j; j < e && keys[j] == key; ++j) {
                const i64 sj = e0 + vals[j];
                sum =
                    __dadd_rn(sum, data ? static_cast<double>(data[sj]) : 1.0);
            }
            idx_out[out] = static_cast<int>(key & kColMask);
            data_out[out] = sum;
            ++out;
        }
    }
}

__global__ void canon_advance_kernel(const i64* __restrict__ loff2, i64 rows,
                                     i64* base, i64* indptr_out, i64 n,
                                     int last) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        base[0] += loff2[rows];
        if (last) indptr_out[n] = base[0];
    }
}

inline int bit_length(u64 x) {
    int b = 0;
    while (x) {
        ++b;
        x >>= 1;
    }
    return b;
}

// Returns the canonical nnz (`h_indptr`: the input indptr on the host).
template <typename IP, typename IX, typename WI>
i64 run_canonicalize(const IP* indptr, const std::vector<i64>& h_indptr,
                     const IX* indices, const WI* data, i64 n, i64* indptr_out,
                     int* idx_out, double* data_out, void* scratch,
                     std::size_t scratch_bytes, i64 chunk, cudaStream_t s) {
    const i64 nnz = n > 0 ? h_indptr[n] : 0;
    i64 max_row = 0;
    for (i64 v = 0; v < n; ++v)
        max_row = std::max(max_row, h_indptr[v + 1] - h_indptr[v]);
    if (chunk < 1 || chunk > kCanonChunk)
        throw std::invalid_argument("chunk_entries must be in [1, 2^26]");
    const i64 cap = canon_chunk_cap(nnz, max_row, chunk);
    Carver c{static_cast<char*>(scratch), 0};
    CanonBufs b;
    b.carve(c, n, cap);
    if (c.used() > scratch_bytes)
        throw std::invalid_argument(
            "leiden: canonicalize scratch too small; need " +
            std::to_string(c.used()) + " bytes");
    cuda_check(cudaMemsetAsync(b.base, 0, sizeof(i64), s),
               "leiden: canonicalize base");
    if (n == 0) {
        cuda_check(cudaMemsetAsync(indptr_out, 0, sizeof(i64), s),
                   "leiden: canonicalize empty");
        cuda_check(cudaStreamSynchronize(s), "leiden: canonicalize sync");
        return 0;
    }
    i64 r0 = 0;
    while (r0 < n) {
        i64 r1 = r0 + 1;
        while (r1 < n && h_indptr[r1 + 1] - h_indptr[r0] <= cap) ++r1;
        const i64 rows = r1 - r0, e0 = h_indptr[r0], m = h_indptr[r1] - e0;
        if (m > 0) {
            canon_keys_kernel<IP, IX><<<grid_rows(rows), kBlock, 0, s>>>(
                indptr, indices, r0, r1, e0, b.keys_a, b.vals_a);
            CUDA_CHECK_LAST_ERROR(canon_keys_kernel);
        }
        cub::DoubleBuffer<u64> keys(b.keys_a, b.keys_b);
        cub::DoubleBuffer<int> vals(b.vals_a, b.vals_b);
        if (m > 1) {
            const int end_bit =
                kColumnBits + bit_length(static_cast<u64>(rows - 1));
            std::size_t tb = b.cub_bytes;
            cuda_check(cub::DeviceRadixSort::SortPairs(
                           b.cub, tb, keys, vals,
                           cub_items(m, "canonical sort"), 0, end_bit, s),
                       "leiden: canonical row sort");
        }
        const unsigned g = grid_items(rows);
        canon_count_kernel<IP>
            <<<g, kBlock, 0, s>>>(indptr, r0, rows, e0, keys.Current(), b.loff);
        CUDA_CHECK_LAST_ERROR(canon_count_kernel);
        std::size_t tb = b.cub_bytes;
        cuda_check(cub::DeviceScan::ExclusiveSum(
                       b.cub, tb, b.loff, b.loff2,
                       cub_items(rows + 1, "canonical offsets"), s),
                   "leiden: canonical offsets");
        canon_write_kernel<IP, WI><<<g, kBlock, 0, s>>>(
            indptr, r0, rows, e0, keys.Current(), vals.Current(), data, b.loff2,
            b.base, indptr_out, idx_out, data_out);
        CUDA_CHECK_LAST_ERROR(canon_write_kernel);
        canon_advance_kernel<<<1, 1, 0, s>>>(b.loff2, rows, b.base, indptr_out,
                                             n, r1 == n ? 1 : 0);
        CUDA_CHECK_LAST_ERROR(canon_advance_kernel);
        r0 = r1;
    }
    i64 out = 0;
    cuda_check(
        cudaMemcpyAsync(&out, b.base, sizeof(i64), cudaMemcpyDeviceToHost, s),
        "leiden: canonical nnz");
    cuda_check(cudaStreamSynchronize(s), "leiden: canonicalize sync");
    return out;
}

// Sources of level-0 quantised weights; `off` = (u != v).
struct QSrcF32 {  // fp32 on the fly (-126 <= s <= 127)
    const float* w;
    float scale;
    __device__ __forceinline__ i64 operator()(i64 j, bool off) const {
        return off ? quantize_f32(w[j], scale) : 0;
    }
};
// fp32 host input: also writes the value to the level-0 device copy
struct QSrcF32Copy {
    const float* w;
    float scale;
    float* out;
    __device__ __forceinline__ i64 operator()(i64 j, bool off) const {
        const float x = w[j];
        out[j] = x;
        return off ? quantize_f32(x, scale) : 0;
    }
};
template <typename WI>
struct QSrcMat {  // materialises W0q (0 for entries that are not counted)
    EntryValue<WI> val;
    int s;
    i64* out;
    __device__ __forceinline__ i64 operator()(i64 j, bool off) const {
        const i64 q = off ? val.quantized(j, s) : 0;
        out[j] = q;
        return q;
    }
};
struct QSrcUnit {
    i64 q;
    __device__ __forceinline__ i64 operator()(i64, bool off) const {
        return off ? q : 0;
    }
};

// I2, warp per row; also writes the int64 offsets and int32 indices copies.
template <typename IP, typename IX, typename Src>
__global__ void quantize_degrees_kernel(const IP* __restrict__ indptr,
                                        const IX* __restrict__ indices, Src src,
                                        i64 n, ClassThresholds th,
                                        i64* __restrict__ indptr64,
                                        int* __restrict__ idx32,
                                        i64* __restrict__ khat, Control* ctl) {
    __shared__ i64 s_two_m[kWarpsPerBlock];
    __shared__ i64 s_cls[kWarpsPerBlock][kNumClasses];
    const int lane = threadIdx.x & (kWarp - 1), wib = threadIdx.x / kWarp;
    const i64 warp0 = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + wib;
    const i64 nwarps = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
    i64 two_m = 0;
    i64 cls[kNumClasses] = {};
    for (i64 v = warp0; v < n; v += nwarps) {
        const i64 b = static_cast<i64>(indptr[v]);
        const i64 e = static_cast<i64>(indptr[v + 1]);
        i64 k = 0;
        for (i64 j = b + lane; j < e; j += kWarp) {
            const i64 u = static_cast<i64>(indices[j]);
            if (idx32) idx32[j] = static_cast<int>(u);
            k += src(j, u != v);
        }
        k = warp_sum(k);
        if (lane == 0) {
            const int c = degree_class(e - b, th);
            indptr64[v] = b;
            if (v == n - 1) indptr64[n] = e;
            khat[v] = k;
            two_m += k;
            ++cls[c];
        }
    }
    if (lane == 0) {
        s_two_m[wib] = two_m;
        for (int c = 0; c < kNumClasses; ++c) s_cls[wib][c] = cls[c];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        i64 t = 0, cc[kNumClasses] = {};
        for (int i = 0; i < kWarpsPerBlock; ++i) {
            t += s_two_m[i];
            for (int c = 0; c < kNumClasses; ++c) cc[c] += s_cls[i][c];
        }
        atomicAdd(reinterpret_cast<u64*>(&ctl->two_m_hat), static_cast<u64>(t));
        for (int c = 0; c < kNumClasses; ++c)
            if (cc[c])
                atomicAdd(reinterpret_cast<u64*>(&ctl->class_count[c]),
                          static_cast<u64>(cc[c]));
    }
}

__global__ void replicate_union_kernel(
    const i64* __restrict__ indptr0, const int* __restrict__ idx0,
    const float* __restrict__ wf0, const i64* __restrict__ wq0,
    const i64* __restrict__ khat0, i64 n, int R, i64* __restrict__ indptr_u,
    int* __restrict__ idx_u, float* __restrict__ wf_u, i64* __restrict__ wq_u,
    i64* __restrict__ khat_u) {
    const int lane = threadIdx.x & (kWarp - 1);
    const i64 warp0 =
        static_cast<i64>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
    const i64 nwarps = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
    const i64 nnz0 = indptr0[n];
    const i64 rows = static_cast<i64>(R) * n;
    for (i64 x = warp0; x < rows; x += nwarps) {
        const i64 r = x / n, v = x - r * n;
        const i64 b = indptr0[v], e = indptr0[v + 1], shift = r * nnz0;
        if (lane == 0) {
            indptr_u[x] = shift + b;
            if (x == rows - 1) indptr_u[rows] = static_cast<i64>(R) * nnz0;
            khat_u[x] = khat0[v];
        }
        for (i64 j = b + lane; j < e; j += kWarp) {
            idx_u[shift + j] = static_cast<int>(r * n) + idx0[j];
            if (wf_u) wf_u[shift + j] = wf0[j];
            if (wq_u) wq_u[shift + j] = wq0[j];
        }
    }
}

inline WKind level0_kind(u32 flags, bool weighted, bool data_is_f32, int s) {
    if (!weighted) return (flags & kFlagUnitZero) ? WKind::I64 : WKind::UNIT;
    if (data_is_f32 && s >= kF32ScaleMin && s <= kF32ScaleMax)
        return WKind::F32;
    return WKind::I64;
}

struct QuantizeResult {
    int s = 0;
    i64 unit_q = 0;
    i64 two_m_hat = 0;
    i64 class_count[kNumClasses] = {};
    WKind wkind = WKind::F32;
    const i64* indptr = nullptr;
    const int* indices = nullptr;
    const float* wf32 = nullptr;
    const i64* wq = nullptr;
};

inline void check_gamma(double gamma) {
    if (!(gamma >= 0.0 && gamma <= kGammaMax))
        throw std::invalid_argument(
            "leiden: resolution must be finite and in [0, 2**20]");
}

// I2 (+ I4) for one scale; the input passed I1 without a pending fix-up.
template <typename IP, typename IX, typename WI>
QuantizeResult run_quantize(const IP* indptr, const IX* indices, const WI* data,
                            i64 n, i64 nnz, bool use_weights, double gamma,
                            const IngestInfo& info, Layout& L,
                            const ClassThresholds& th, cudaStream_t stream) {
    check_gamma(gamma);
    if (info.flags & (kFlagNonCanonical | kFlagAsymmetric))
        throw std::invalid_argument(
            "leiden: input must be canonical and symmetric (run canonicalize / "
            "A + A.T first)");
    if (info.flags & (kFlagNegative | kFlagNonFinite))
        throw std::invalid_argument(
            "leiden: adjacency weights must be finite and non-negative");
    if (info.n_counted < 0 || info.n_counted >= kMaxCountedEntries)
        throw std::invalid_argument("leiden: nnz_counted out of range");
    if (n != L.p.n || nnz != L.p.nnz)
        throw std::invalid_argument("leiden: layout does not match the graph");
    QuantizeResult res;
    constexpr bool idx_is_32 = std::is_same_v<IX, int>;
    const bool host = L.p.host_input;
    const bool weighted = use_weights;
    const EntryValue<WI> val{data, weighted};
    res.s = scale_for_gamma(scale_s0(info.n_counted, info.wmax), gamma);
    res.unit_q = weighted ? 0 : unit_weight(res.s);
    res.wkind =
        level0_kind(info.flags, weighted, std::is_same_v<WI, float>, res.s);
    // fp32 host input with resolutions of both kinds has an I64 layout without
    // an fp32 copy: its F32 resolutions read the identical W0q
    if (host && res.wkind == WKind::F32 && L.l0.wf32 == nullptr &&
        L.l0.wq != nullptr)
        res.wkind = WKind::I64;
    const bool ok = (res.wkind != WKind::I64 || L.l0.wq != nullptr) &&
                    ((idx_is_32 && !host) || L.l0.indices != nullptr) &&
                    (res.wkind != WKind::F32 || !host || L.l0.wf32 != nullptr);
    if (!ok) throw std::logic_error("leiden: the layout lacks a level-0 array");
    Control* ctl = L.ctl;
    clear_control(ctl, stream);
    int* idx32 = idx_is_32 && !host ? nullptr : L.l0.indices;
    float* wf32_copy = host ? L.l0.wf32 : nullptr;
    res.indptr = L.l0.indptr;
    res.indices =
        idx32 != nullptr ? idx32 : reinterpret_cast<const int*>(indices);
    res.wf32 = res.wkind != WKind::F32 ? nullptr
               : wf32_copy != nullptr  ? wf32_copy
                                       : reinterpret_cast<const float*>(data);
    res.wq = res.wkind == WKind::I64 ? L.l0.wq : nullptr;

    if (n > 0) {
        auto launch = [&](auto src) {
            using Src = decltype(src);
            const unsigned g = grid_occ(quantize_degrees_kernel<IP, IX, Src>, n,
                                        kWarpsPerBlock);
            quantize_degrees_kernel<IP, IX, Src>
                <<<g, kBlock, 0, stream>>>(indptr, indices, src, n, th,
                                           L.l0.indptr, idx32, L.l0.khat, ctl);
            CUDA_CHECK_LAST_ERROR(quantize_degrees_kernel);
        };
        if (res.wkind == WKind::UNIT) {
            launch(QSrcUnit{res.unit_q});
        } else if (res.wkind == WKind::I64) {
            launch(QSrcMat<WI>{val, res.s, L.l0.wq});
        } else if constexpr (std::is_same_v<WI, float>) {
            if (wf32_copy != nullptr)
                launch(QSrcF32Copy{data, std::ldexp(1.0f, res.s), wf32_copy});
            else
                launch(QSrcF32{data, std::ldexp(1.0f, res.s)});
        }
    } else {
        cuda_check(cudaMemsetAsync(L.l0.indptr, 0, sizeof(i64), stream),
                   "leiden: empty indptr");
    }

    // I4: replica union (iteration 1; rebuilt when s changes)
    if (L.p.replicas > 1 && n > 0) {
        replicate_union_kernel<<<grid_rows(L.p.n_union()), kBlock, 0, stream>>>(
            L.l0.indptr, res.indices, res.wf32, res.wq, L.l0.khat, n,
            L.p.replicas, L.uni.indptr, L.uni.indices,
            res.wkind == WKind::F32 ? L.uni.wf32 : nullptr,
            res.wkind == WKind::I64 ? L.uni.wq : nullptr, L.uni.khat);
        CUDA_CHECK_LAST_ERROR(replicate_union_kernel);
    }

    const Control h = read_control(ctl, stream, L.pinned);  // sync S2
    res.two_m_hat = h.two_m_hat;
    for (int c = 0; c < kNumClasses; ++c) res.class_count[c] = h.class_count[c];
    return res;
}

inline nb::dict info_dict(const IngestInfo& r, bool idx64) {
    nb::dict d;
    d["neg"] = (r.flags & kFlagNegative) != 0;
    d["nonfinite"] = (r.flags & kFlagNonFinite) != 0;
    d["canonical"] = (r.flags & kFlagNonCanonical) == 0;
    d["symmetric"] = (r.flags & kFlagAsymmetric) == 0;
    d["nnz_counted"] = r.n_counted;
    d["wmax_bits"] = r.wmax_bits;
    d["idx64"] = idx64;
    d["flags"] = r.flags;
    d["unit_zero"] = (r.flags & kFlagUnitZero) != 0;
    d["wmax"] = r.wmax;
    d["h_fwd"] = r.h_fwd;
    d["h_rev"] = r.h_rev;
    return d;
}

inline IngestInfo info_from_dict(const nb::dict& d) {
    IngestInfo r;
    r.flags = nb::cast<u32>(d["flags"]);
    r.n_counted = nb::cast<i64>(d["nnz_counted"]);
    r.wmax = nb::cast<double>(d["wmax"]);
    return r;
}

// Device accumulators of ingest_check, serialised by a host mutex.
__device__ Control g_ingest_check_ctl;

inline std::mutex& ingest_check_mutex() {
    static std::mutex m;
    return m;
}

template <typename T, typename InDevice>
using input_array = nb::ndarray<T, InDevice, nb::c_contig>;

template <typename InDevice>
constexpr bool host_input_v = std::is_same_v<InDevice, nb::device::cpu>;

// Host arrays are read in place, which needs pageable memory access.
inline void require_pageable_access() {
    if (!device_info().pageable_access)
        throw std::invalid_argument(
            "leiden: host (NumPy) input arrays need a GPU that reads "
            "pageable host memory; copy the arrays to the device instead");
}

template <typename IP, typename IX, typename WI, typename InDevice,
          typename Device>
void register_ingest_typed(nb::module_& m) {
    m.def(
        "ingest_check",
        [](input_array<const IP, InDevice> indptr,
           input_array<const IX, InDevice> indices,
           input_array<const WI, InDevice> data, bool use_weights,
           std::uintptr_t pinned, std::uintptr_t stream) {
            if (indptr.ndim() != 1 || indptr.size() < 1)
                throw std::invalid_argument(
                    "indptr must be 1-D with n + 1 entries");
            const i64 n = static_cast<i64>(indptr.size()) - 1;
            const i64 nnz = static_cast<i64>(indices.size());
            if (static_cast<i64>(data.size()) != nnz)
                throw std::invalid_argument(
                    "data and indices must have equal length");
            if constexpr (host_input_v<InDevice>) require_pageable_access();
            IngestInfo r;
            {
                nb::gil_scoped_release release;
                const auto s = (cudaStream_t)stream;
                std::lock_guard<std::mutex> lock(ingest_check_mutex());
                void* ctl = nullptr;
                cuda_check(cudaGetSymbolAddress(&ctl, g_ingest_check_ctl),
                           "leiden: ingest_check accumulators");
                r = run_synced(s, [&] {
                    return run_ingest_check<IP, IX, WI>(
                        indptr.data(), indices.data(), data.data(), n, nnz,
                        use_weights, static_cast<Control*>(ctl),
                        reinterpret_cast<void*>(pinned), s);
                });
            }
            return info_dict(r, !std::is_same_v<IX, int>);
        },
        "indptr"_a.noconvert(), nb::kw_only(), "indices"_a.noconvert(),
        "data"_a.noconvert(), "use_weights"_a, "pinned"_a = 0, "stream"_a = 0);

    if constexpr (!host_input_v<InDevice>) {
        m.def(
            "canonicalize",
            [](gpu_array_c<const IP, Device> indptr,
               gpu_array_c<const IX, Device> indices,
               gpu_array_c<const WI, Device> data,
               gpu_array_c<i64, Device> out_indptr,
               gpu_array_c<int, Device> out_indices,
               gpu_array_c<double, Device> out_data,
               gpu_array_c<std::uint8_t, Device> scratch,
               long long chunk_entries, std::uintptr_t stream) {
                if (indptr.ndim() != 1 || indptr.size() < 1)
                    throw std::invalid_argument(
                        "indptr must be 1-D with n + 1 entries");
                const i64 n = static_cast<i64>(indptr.size()) - 1;
                const i64 nnz = static_cast<i64>(indices.size());
                if (n >= kMaxVertices)
                    throw std::invalid_argument("leiden: n must be < 2^30");
                if (static_cast<i64>(data.size()) != nnz)
                    throw std::invalid_argument(
                        "data and indices must have equal length");
                if (static_cast<i64>(out_indptr.size()) != n + 1 ||
                    static_cast<i64>(out_indices.size()) < nnz ||
                    static_cast<i64>(out_data.size()) < nnz)
                    throw std::invalid_argument(
                        "outputs must hold n + 1 offsets and nnz entries");
                i64 out = 0;
                {
                    nb::gil_scoped_release release;
                    const auto s = (cudaStream_t)stream;
                    out = run_synced(s, [&] {
                        std::vector<IP> raw(n + 1);
                        cuda_check(cudaMemcpyAsync(raw.data(), indptr.data(),
                                                   (n + 1) * sizeof(IP),
                                                   cudaMemcpyDeviceToHost, s),
                                   "leiden: canonicalize indptr");
                        cuda_check(cudaStreamSynchronize(s),
                                   "leiden: canonicalize sync");
                        std::vector<i64> h(raw.begin(), raw.end());
                        if (h[0] != 0 || h[n] != nnz)
                            throw std::invalid_argument(
                                "leiden: indptr must start at 0 and end at "
                                "nnz");
                        for (i64 v = 0; v < n; ++v)
                            if (h[v + 1] < h[v])
                                throw std::invalid_argument(
                                    "leiden: indptr must be non-decreasing");
                        return run_canonicalize<IP, IX, WI>(
                            indptr.data(), h, indices.data(), data.data(), n,
                            out_indptr.data(), out_indices.data(),
                            out_data.data(), scratch.data(), scratch.size(),
                            chunk_entries, s);
                    });
                }
                return out;
            },
            "indptr"_a.noconvert(), nb::kw_only(), "indices"_a.noconvert(),
            "data"_a.noconvert(), "out_indptr"_a.noconvert(),
            "out_indices"_a.noconvert(), "out_data"_a.noconvert(),
            "scratch"_a.noconvert(), "chunk_entries"_a = kCanonChunk,
            "stream"_a = 0);
    }
}

template <typename InDevice, typename Device>
void register_ingest_inputs(nb::module_& m) {
    register_ingest_typed<int, int, float, InDevice, Device>(m);
    register_ingest_typed<int, int, double, InDevice, Device>(m);
    register_ingest_typed<long long, int, float, InDevice, Device>(m);
    register_ingest_typed<long long, int, double, InDevice, Device>(m);
    register_ingest_typed<int, long long, float, InDevice, Device>(m);
    register_ingest_typed<int, long long, double, InDevice, Device>(m);
    register_ingest_typed<long long, long long, float, InDevice, Device>(m);
    register_ingest_typed<long long, long long, double, InDevice, Device>(m);
}

// Device input first (overloads are tried in order), then host input.
template <typename Device>
void register_ingest_bindings(nb::module_& m) {
    register_ingest_inputs<Device, Device>(m);
    register_ingest_inputs<nb::device::cpu, Device>(m);
}

// int64 column indices -> int32; out-of-range values -> -1, which ingest_check
// rejects (Python's chunked copy on GPUs without pageable access).
__global__ void narrow_indices_kernel(const i64* __restrict__ src, i64 m, i64 n,
                                      int* __restrict__ dst) {
    const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
    for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; i < m;
         i += stride) {
        const i64 u = src[i];
        dst[i] = (u >= 0 && u < n) ? static_cast<int>(u) : -1;
    }
}

template <typename Device>
void register_narrow_indices(nb::module_& m) {
    m.def(
        "narrow_indices",
        [](gpu_array_c<const i64, Device> src, gpu_array_c<int, Device> dst,
           long long n, std::uintptr_t stream) {
            const i64 count = static_cast<i64>(src.size());
            if (static_cast<i64>(dst.size()) != count)
                throw std::invalid_argument("src and dst must match");
            if (n < 0 || n >= kMaxVertices)
                throw std::invalid_argument("leiden: n must be in [0, 2^30)");
            if (count > 0) {
                narrow_indices_kernel<<<grid_items(count), kBlock, 0,
                                        (cudaStream_t)stream>>>(
                    src.data(), count, n, dst.data());
                CUDA_CHECK_LAST_ERROR(narrow_indices_kernel);
            }
        },
        "src"_a.noconvert(), nb::kw_only(), "dst"_a.noconvert(), "n"_a,
        "stream"_a = 0);
}

inline void register_ingest(nb::module_& m) {
    REGISTER_GPU_BINDINGS(register_ingest_bindings, m);
    REGISTER_GPU_BINDINGS(register_narrow_indices, m);

    m.def(
        "canonicalize_scratch_bytes",
        [](long long n, long long nnz, long long max_row_nnz,
           long long chunk_entries) {
            return canonicalize_scratch_bytes(n, nnz, max_row_nnz,
                                              chunk_entries);
        },
        "n"_a, "nnz"_a, "max_row_nnz"_a = 0, "chunk_entries"_a = kCanonChunk);
}

}  // namespace leiden
