#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <type_traits>

// Warp size (threads per warp on NVIDIA GPUs)
static constexpr int WARP_SIZE = 32;

// ---------------------------------------------------------------------------
// Energy-distance pair sums: one kernel for every precision and both layouts.
// A pair-major 1-D grid (a block belongs to one pair): A = the larger group,
// split across the pair's blocks and held in shared memory once per A tile
// when that costs no blocks per SM; B tiles stream through shared memory one
// window at a time. Within-group pairs skip the lower triangle. Sums are Kahan
// per thread with one float64 atomic per block. Distances come from the Gram
// form |a|^2 + |b|^2 - 2 a.b; the arithmetic on a window is a policy (math):
//   EdFma<double>  DFMA                 precision float64
//   EdFma<float>   FFMA                 precision float32
//   EdMma16        fp16 tensor cores    precision float16
// Rows are centered and scaled by a power of 2 as they are loaded:
// x -> M::cvt(x, sc, nm[col]) with nm = -mu * sc. Their norms come from the
// same loader and policy on the same rows (the diagonal of A A^T), so
// duplicate cells are exactly 0 apart and dense input matches CSR bit for bit.
// ---------------------------------------------------------------------------
#ifdef __CUDA_ARCH__
#define ED_ARCH __CUDA_ARCH__
#else
#define ED_ARCH 0  // host pass
#endif
// threads; B rows per tile; bytes per row and feature window; smem row pitch:
// 8 rows 144 B apart hit distinct 16-byte bank groups (ldmatrix, LDS.128)
static constexpr int ED_THREADS = 256, ED_TB = 128, ED_WB = 128,
                     ED_PITCH = ED_WB + 16;

// Larger group becomes A (split across the pair's blocks); smaller becomes B
// (tiled into shared memory and kept hot in L2 across A tiles).
struct EDistancePairTiles {
    int start_a, end_a, start_b, end_b;
};

__device__ inline EDistancePairTiles edistance_pair_tiles(
    int pair_a, int pair_b, const int* __restrict__ cat_offsets) {
    const int start_pa = cat_offsets[pair_a];
    const int end_pa = cat_offsets[pair_a + 1];
    const int start_pb = cat_offsets[pair_b];
    const int end_pb = cat_offsets[pair_b + 1];
    const bool swap = (end_pa - start_pa) < (end_pb - start_pb);
    EDistancePairTiles t;
    t.start_a = swap ? start_pb : start_pa;
    t.end_a = swap ? end_pb : end_pa;
    t.start_b = swap ? start_pa : start_pb;
    t.end_b = swap ? end_pa : end_pb;
    return t;
}

// Warp-shuffle + block reduction of local_sum, then atomicAdd into
// pairwise_sums[pair_id]. Runs once per block (not hot).
template <typename T>
__device__ inline void edistance_block_reduce_add(T local_sum,
                                                  T* __restrict__ pairwise_sums,
                                                  int pair_id, int thread_id,
                                                  int block_size) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    static __shared__ T warp_sums[WARP_SIZE];
    if ((thread_id & (WARP_SIZE - 1)) == 0)
        warp_sums[thread_id / WARP_SIZE] = local_sum;
    __syncthreads();

    if (thread_id < WARP_SIZE) {
        T val = (thread_id < (block_size / WARP_SIZE)) ? warp_sums[thread_id]
                                                       : T(0.0);
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);

        if (thread_id == 0) {
            atomicAdd(&pairwise_sums[pair_id], val);
        }
    }
}

template <typename T>
struct EDistanceKahan {
    T s = T(0), c = T(0);
    __device__ void add(T x) {
        const T y = x - c, t = s + y;
        c = (t - s) - y;
        s = t;
    }
};

// ---------------------------------------------------------------------------
// Math policies. window() accumulates kn columns of one window: A rows at a
// (pitch pa), B rows at b (pitch ED_PITCH); each() visits the accumulators as
// (A row, B row, slot, value), slot a compile-time index.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float edistance_sqrt_approx(float x) {
    float y;
    asm("sqrt.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

// FFMA / DFMA register tile, 8 x 8 floats or 4 x 8 doubles per thread: thread
// (tm, tn) of a 16 x 16 grid owns A rows tm + 16 i and B rows tn + 16 j and
// reads k 16 bytes at a time. float64 is bound by the FP64 rate: of a tile with
// ma x mb real rows a thread skips the products of its padding rows, whose
// sums the kernels drop.
template <typename T>
struct EdFma {
    using E = T;
    using Acc = T;
    using V = std::conditional_t<sizeof(T) == 4, float4, double2>;
    static constexpr int RI = sizeof(T) == 4 ? 8 : 4, TA = 16 * RI;
    static constexpr bool ok = true;
    static constexpr int min_blocks = 1;
    struct Frag {
        T c[RI][8];
    };
    template <typename In, typename W = std::common_type_t<In, T>>
    __device__ static E cvt(In x, W sc, W nm) {
        return E(fma(W(x), sc, nm));
    }
    __device__ static T dist(T v, T nr, T nc) {
        const T d2 = fma(T(-2), v, nr + nc);
        const T c = d2 < T(0) ? T(0) : d2;  // keeps NaN
        if constexpr (sizeof(T) == 4) {
            return edistance_sqrt_approx(c);
        } else {
            return sqrt(c);
        }
    }
    __device__ static void dot(T& c, const float4& a, const float4& b) {
        c = fma(a.x, b.x, c), c = fma(a.y, b.y, c), c = fma(a.z, b.z, c),
        c = fma(a.w, b.w, c);
    }
    __device__ static void dot(T& c, const double2& a, const double2& b) {
        c = fma(a.x, b.x, c), c = fma(a.y, b.y, c);
    }
    __device__ static void zero(Frag& f) {
#pragma unroll
        for (int i = 0; i < RI; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) f.c[i][j] = T(0);
    }
    __device__ static int tm() {
        return 4 * (threadIdx.x >> 5 & 3) + (threadIdx.x >> 3 & 3);
    }
    __device__ static int tn() {
        return 8 * (threadIdx.x >> 7) + (threadIdx.x & 7);
    }
    __device__ static void window(Frag& f, const unsigned char* a, int pa,
                                  const unsigned char* b, int kn, int ma,
                                  int mb) {
        a += tm() * pa, b += tn() * ED_PITCH;
#pragma unroll 2
        for (int kb = 0; kb < kn * (int)sizeof(T); kb += 16) {
            V av[RI];
#pragma unroll
            for (int i = 0; i < RI; ++i)
                av[i] = *reinterpret_cast<const V*>(a + 16 * i * pa + kb);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (sizeof(T) == 8 && tn() + 16 * j >= mb) break;
                const V bv =
                    *reinterpret_cast<const V*>(b + 16 * j * ED_PITCH + kb);
#pragma unroll
                for (int i = 0; i < RI; ++i) {
                    if (sizeof(T) == 8 && tm() + 16 * i >= ma) break;
                    dot(f.c[i][j], av[i], bv);
                }
            }
        }
    }
    template <class F>
    __device__ static void each(const Frag& f, F&& fn) {
#pragma unroll
        for (int i = 0; i < RI; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j)
                fn(tm() + 16 * i, tn() + 16 * j, 8 * i + j, f.c[i][j]);
    }
};

__device__ __forceinline__ void edistance_ldsm4(uint32_t (&r)[4],
                                                const unsigned char* p) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"((uint32_t)__cvta_generic_to_shared(p)));
}
// f16 in, f32 accumulate: m16n8k16 on sm_80+, two m16n8k8 (k halves) on sm_75
__device__ __forceinline__ void edistance_mma(float (&c)[4],
                                              const uint32_t (&a)[4],
                                              const uint32_t (&b)[2]) {
#if ED_ARCH >= 800 || ED_ARCH == 0
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#elif ED_ARCH >= 750
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
        "{%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
        "{%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#endif
}

// Tensor-core Gram on fp16 rows, 8 warps as 2 x 4 of 64 x 32: accumulator
// (i, j, q) holds A row wm + 16 i + g + 8 (q >> 1) and B row
// wn + 8 j + 2 t + (q & 1) (g = lane / 4, t = lane % 4); a k-step is 32 bytes
// (16 halves) of every row. Below sm_75: no kernels (ok = false).
struct EdMma16 {
    using E = __half;
    using Acc = float;
    static constexpr int TA = 128;
    static constexpr bool ok = ED_ARCH == 0 || ED_ARCH >= 750;
    // <= 128 registers for 2 blocks / SM (sm_75 fits 1: no cap)
    static constexpr int min_blocks = ED_ARCH >= 800 || ED_ARCH == 0 ? 2 : 1;
    struct Frag {
        float c[4][4][4];
    };
    // Centered in the input's type, then rounded to fp16
    __device__ static E cvt(float x, float sc, float nm) {
        return __float2half_rn(fmaf(x, sc, nm));
    }
    __device__ static E cvt(double x, double sc, double nm) {
        return __double2half(fma(x, sc, nm));
    }
    __device__ static float dist(float v, float nr, float nc) {
        const float d2 = fmaf(-2.f, v, nr + nc);
        return edistance_sqrt_approx(d2 < 0.f ? 0.f : d2);
    }
    __device__ static void zero(Frag& f) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int q = 0; q < 4; ++q) f.c[i][j][q] = 0.f;
    }
    __device__ static int wm() {
        return (threadIdx.x >> 7) * 64;
    }
    __device__ static int wn() {
        return (threadIdx.x >> 5 & 3) * 32;
    }
    __device__ static void window(Frag& f, const unsigned char* a, int pa,
                                  const unsigned char* b, int kn, int, int) {
        const int lane = threadIdx.x & 31, nb = kn * (int)sizeof(E);
        a += wm() * pa + (lane & 15) * pa + (lane >> 4) * 16;
        b += wn() * ED_PITCH + ((lane & 7) + (lane >> 4 << 3)) * ED_PITCH +
             (lane >> 3 & 1) * 16;
#pragma unroll
        for (int kb = 0; kb < ED_WB; kb += 32) {
            if (kb < nb) {
                uint32_t af[4][4], bf[4][2], r[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    edistance_ldsm4(af[i], a + 16 * i * pa + kb);
#pragma unroll
                for (int j = 0; j < 4; j += 2) {
                    edistance_ldsm4(r, b + 8 * j * ED_PITCH + kb);
                    bf[j][0] = r[0], bf[j][1] = r[1], bf[j + 1][0] = r[2],
                    bf[j + 1][1] = r[3];
                }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        edistance_mma(f.c[i][j], af[i], bf[j]);
            }
        }
    }
    template <class F>
    __device__ static void each(const Frag& f, F&& fn) {
        const int lane = threadIdx.x & 31;
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    fn(wm() + 16 * i + (lane >> 2) + 8 * (q >> 1),
                       wn() + 8 * j + 2 * (lane & 3) + (q & 1),
                       16 * i + 4 * j + q, f.c[i][j][q]);
    }
};

// ---------------------------------------------------------------------------
// Row loaders. load() fills window w of tile rows [r0, r0 + nr) (source rows
// rows[r]) into s, row r - r0 at s + (r - r0) * p, as the policy's element
// with zeros past d. Dense and CSR produce the same bytes.
// ---------------------------------------------------------------------------
// n valid elements of a row at x, 16 / 8 / 4-byte aligned loads where possible
template <typename In, int N>
__device__ __forceinline__ void edistance_ld(In (&v)[N], const In* x, int n,
                                             int align) {
    constexpr int B = N * (int)sizeof(In);
    if (n >= N && align >= 16 && B % 16 == 0) {
#pragma unroll
        for (int i = 0; i < B / 16; ++i)
            reinterpret_cast<uint4*>(v)[i] =
                __ldg(reinterpret_cast<const uint4*>(x) + i);
    } else if (n >= N && align >= 8 && B % 8 == 0) {
#pragma unroll
        for (int i = 0; i < B / 8; ++i)
            reinterpret_cast<uint2*>(v)[i] =
                __ldg(reinterpret_cast<const uint2*>(x) + i);
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) v[i] = i < n ? __ldg(x + i) : In(0);
    }
}

template <class M, typename In>
struct EdDenseRows {
    using E = typename M::E;
    const In* X;    // rows of ldx elements
    long long ldx;  // row stride (elements)
    int d, align;   // columns; alignment (bytes) shared by X and its row stride
    using W = std::common_type_t<In, typename M::Acc>;
    const W* nm;  // -mu * sc per column
    W sc;
    __device__ void load(unsigned char* s, int p, const int* rows, int r0,
                         int nr, int w) {
        constexpr int NE = 16 / sizeof(E), CH = ED_WB / 16;
        for (int v = threadIdx.x; v < nr * CH; v += ED_THREADS) {
            const int r = v / CH,
                      c = w * (ED_WB / (int)sizeof(E)) + v % CH * NE;
            In x[NE];
            edistance_ld(x, X + (size_t)rows[r0 + r] * ldx + c, d - c, align);
            union {
                uint4 u;
                E e[NE];
            } ch;
#pragma unroll
            for (int i = 0; i < NE; ++i)
                ch.e[i] =
                    c + i < d ? M::cvt(x[i], sc, __ldg(nm + c + i)) : E(0);
            *reinterpret_cast<uint4*>(s + r * p + v % CH * 16) = ch.u;
        }
    }
};

// The entries of window w of a CSR row whose columns may repeat: the (adjacent)
// entries of a column are summed before the conversion. Returns the cursor.
// Not inlined, so the kernels keep the registers of the canonical loop.
template <class M, typename In, typename W, typename IndptrT>
__device__ __noinline__ IndptrT edistance_load_sum(uint4* row, const In* data,
                                                   const int* indices,
                                                   const W* nm, W sc,
                                                   IndptrT cur, IndptrT end,
                                                   int w) {
    using E = typename M::E;
    constexpr int KW = ED_WB / sizeof(E);
    In x = 0;
    int prev = -1;
    for (int col; cur < end && (col = __ldg(indices + cur)) < (w + 1) * KW;
         ++cur) {
        const In v = __ldg(data + cur);
        x = col == prev ? x + v : v;
        prev = col;
        if (col >= w * KW)
            reinterpret_cast<E*>(row)[col - w * KW] =
                M::cvt(x, sc, __ldg(nm + col));
    }
    return cur;
}

// CSR with sorted columns. Thread t owns tile row t and keeps a forward cursor
// through its columns across the windows. Per window: copy the window's
// converted zeros, then scatter the nonzeros converted as the dense loader
// does. Only a matrix with duplicates takes edistance_load_sum.
template <class M, typename In, typename IndptrT>
struct EdCsrRows {
    using E = typename M::E;
    const In* data;
    const int* indices;
    const IndptrT* indptr;
    using W = std::common_type_t<In, typename M::Acc>;
    const W* nm;
    W sc;
    const uint4* zero;  // M::cvt(0, sc, nm[col]) per column, zeros past d
    bool duplicates;    // not canonical: a column may have several entries
    IndptrT cur, end;
    __device__ void load(unsigned char* s, int p, const int* rows, int r0,
                         int nr, int w) {
        constexpr int KW = ED_WB / sizeof(E);
        const int t = (int)threadIdx.x - r0;
        if (t < 0 || t >= nr) return;
        if (w == 0)
            cur = __ldg(indptr + rows[threadIdx.x]),
            end = __ldg(indptr + rows[threadIdx.x] + 1);
        uint4* row = reinterpret_cast<uint4*>(s + t * p);
#pragma unroll
        for (int c = 0; c < ED_WB / 16; ++c)
            row[c] = __ldg(zero + w * (ED_WB / 16) + c);
        if (!duplicates) {
            for (int col;
                 cur < end && (col = __ldg(indices + cur)) < (w + 1) * KW;
                 ++cur)
                // true for sorted input; keeps unsorted columns inside the row
                if (col >= w * KW)
                    reinterpret_cast<E*>(row)[col - w * KW] =
                        M::cvt(__ldg(data + cur), sc, __ldg(nm + col));
            return;
        }
        cur = edistance_load_sum<M>(row, data, indices, nm, sc, cur, end, w);
    }
};

// ---------------------------------------------------------------------------
// One (A tile, B tile): the feature-window loop. With a_loaded the A rows
// of every window already sit in sA (pitch pa); otherwise they stream with B.
// ma x mb of the tile's rows are real, the rest padding.
// ---------------------------------------------------------------------------
template <class M, class L>
__device__ __forceinline__ void edistance_tile(typename M::Frag& f, L& l,
                                               unsigned char* sA, int pa,
                                               unsigned char* sB,
                                               const int* rows, int kp,
                                               bool a_loaded, int ma, int mb) {
    constexpr int KW = ED_WB / sizeof(typename M::E);
    M::zero(f);
    for (int w = 0; w * KW < kp; ++w) {
        if (!a_loaded) l.load(sA, ED_PITCH, rows, 0, M::TA, w);
        l.load(sB, ED_PITCH, rows, M::TA, ED_TB, w);
        __syncthreads();
        M::window(f, a_loaded ? sA + w * ED_WB : sA, pa, sB,
                  min(KW, kp - w * KW), ma, mb);
        __syncthreads();
    }
}

// Pair sums: pairwise_sums[p] += the distances of pair p (cross: all i, j;
// within: i < j), in the units of the scaled rows. block_offsets[p] is the
// first block of pair p (prefix sums of per-pair block counts).
template <class M, class L>
__global__ void __launch_bounds__(ED_THREADS, M::min_blocks)
    edistance_kernel(const L ld, const typename M::Acc* __restrict__ norms,
                     const int* __restrict__ cat_offsets,
                     const int* __restrict__ cell_indices,
                     const int* __restrict__ pair_left,
                     const int* __restrict__ pair_right,
                     const int* __restrict__ block_offsets, int num_pairs,
                     int kp, bool a_resident,
                     double* __restrict__ pairwise_sums) {
    if constexpr (M::ok) {
        using T = typename M::Acc;
        constexpr int TA = M::TA, KW = ED_WB / sizeof(typename M::E);
        extern __shared__ __align__(16) unsigned char smem[];
        __shared__ int s_rows[TA + ED_TB];
        __shared__ T s_norm[TA + ED_TB];

        const int b = blockIdx.x, extra = block_offsets[num_pairs] - num_pairs;
        // the last pair whose first block is <= b
        int pair_id = max(0, b - extra);
        for (int hi = min(b, num_pairs - 1); pair_id < hi;) {
            const int mid = (pair_id + hi + 1) >> 1;
            if (block_offsets[mid] <= (int)blockIdx.x) {
                pair_id = mid;
            } else {
                hi = mid - 1;
            }
        }
        const int block_in_pair = blockIdx.x - block_offsets[pair_id];
        const int blocks_per_pair =
            block_offsets[pair_id + 1] - block_offsets[pair_id];
        const bool diag = pair_left[pair_id] == pair_right[pair_id];
        const EDistancePairTiles t = edistance_pair_tiles(
            pair_left[pair_id], pair_right[pair_id], cat_offsets);
        const int n_a = t.end_a - t.start_a, n_b = t.end_b - t.start_b;
        const int pa = a_resident ? (kp + KW - 1) / KW * ED_WB + 16 : ED_PITCH;
        unsigned char* sA = smem;
        unsigned char* sB = smem + TA * pa;
        L l = ld;
        typename M::Frag f;
        EDistanceKahan<T> sum[4];  // a tile adds one partial sum per chain

        for (int ia = block_in_pair * TA; ia < n_a;
             ia += blocks_per_pair * TA) {
            const int ma = min(TA, n_a - ia);
            // Within-group pairs: B tiles holding only j <= the tile's first A
            // cell add nothing.
            const int jb_start = diag ? (ia + 1) / ED_TB * ED_TB : 0;
            for (int jb = jb_start; jb < n_b; jb += ED_TB) {
                const int mb = min(ED_TB, n_b - jb);
                // the previous tile's epilogue is done with s_rows / s_norm
                __syncthreads();
                if (threadIdx.x < TA + ED_TB) {
                    const bool is_a = threadIdx.x < TA;
                    const int r = is_a ? threadIdx.x : threadIdx.x - TA;
                    const int pos = (is_a ? t.start_a + ia : t.start_b + jb) +
                                    min(r, (is_a ? ma : mb) - 1);
                    const int row = cell_indices[pos];
                    s_rows[threadIdx.x] = row;
                    s_norm[threadIdx.x] = norms[row];
                }
                __syncthreads();
                if (a_resident && jb == jb_start) {
                    for (int w = 0; w * KW < kp; ++w)
                        l.load(sA + w * ED_WB, pa, s_rows, 0, TA, w);
                }
                edistance_tile<M>(f, l, sA, pa, sB, s_rows, kp, a_resident, ma,
                                  mb);
                T part[4] = {};
                M::each(f, [&](int i, int j, int k, T v) {
                    const bool keep =
                        i < ma && j < mb && (!diag || ia + i < jb + j);
                    part[k & 3] +=
                        keep ? M::dist(v, s_norm[i], s_norm[TA + j]) : T(0);
                });
#pragma unroll
                for (int q = 0; q < 4; ++q) sum[q].add(part[q]);
            }
        }
        double total = 0.0;
#pragma unroll
        for (int q = 0; q < 4; ++q)
            total += double(sum[q].s) - double(sum[q].c);
        edistance_block_reduce_add(total, pairwise_sums, pair_id, threadIdx.x,
                                   ED_THREADS);
    }
}

// Norms of the n rows: block b loads rows [TA b, TA b + TA) as A rows and
// again as B rows and keeps the diagonal of A A^T.
template <class M, class L>
__global__ void __launch_bounds__(ED_THREADS, 1)
    edistance_norms_kernel(const L ld, int n, int kp,
                           typename M::Acc* __restrict__ norms) {
    if constexpr (M::ok) {
        constexpr int TA = M::TA;
        extern __shared__ __align__(16) unsigned char smem[];
        __shared__ int s_rows[TA + ED_TB];
        const int p0 = blockIdx.x * TA, m = min(TA, n - p0);
        if (threadIdx.x < TA + ED_TB)
            s_rows[threadIdx.x] = p0 + min((int)threadIdx.x % TA, m - 1);
        __syncthreads();
        L l = ld;
        typename M::Frag f;
        edistance_tile<M>(f, l, smem, ED_PITCH, smem + TA * ED_PITCH, s_rows,
                          kp, false, m, m);
        M::each(f, [&](int i, int j, int, typename M::Acc v) {
            if (i == j && i < m) norms[p0 + i] = v;
        });
    }
}

// ---------------------------------------------------------------------------
// Center and scale (_center in _edistance.py) from every step-th row, read in
// place: a thread reduces one window of cw = R::CW columns of sampled row i.
//   pass 0: out[i] = the row's largest |x|, compared as bit patterns (a NaN
//           of either sign ranks last)
//   pass 1: over the kept rows (finite values): sums[c] += rint(ldexp(x,
//           shift)), integers, exact in float64 in any order; ext[c] and
//           ext[d + c] = the largest key of x and of -x in column c
// The largest |x - mu[c]| of the kept rows is pass 0 over EdCenterReach.
// ---------------------------------------------------------------------------

__device__ __forceinline__ unsigned int edistance_bits(float x) {
    return __float_as_uint(x);
}
__device__ __forceinline__ unsigned long long edistance_bits(double x) {
    return __double_as_longlong(x);
}

// A key orders the non-NaN values as unsigned integers above 0 (-0.0 below
// +0.0), and ~key(x) is key(-x); edistance_unkey gives the value back.
template <typename In>
__device__ __forceinline__ auto edistance_key(In x) {
    const auto b = edistance_bits(x);
    const auto sign = decltype(b)(1) << (8 * sizeof(b) - 1);
    return b & sign ? ~b : b | sign;
}
__device__ __forceinline__ float edistance_unkey(unsigned int k) {
    return __uint_as_float(k >> 31 ? k ^ 1u << 31 : ~k);
}
__device__ __forceinline__ double edistance_unkey(unsigned long long k) {
    return __longlong_as_double(k >> 63 ? k ^ 1ull << 63 : ~k);
}

template <typename In>
struct EdCenterDense {
    static constexpr int CW = 64;
    const In* X;
    long long ldx;    // row stride (elements)
    const int* rows;  // read cell r is row rows[r] of X; null: row r
    __device__ void fill(In* win, long long row, int c0, int nw) const {
        if (rows) row = rows[row];
        for (int c = 0; c < nw; ++c) win[c] = X[row * ldx + c0 + c];
    }
};

// CSR with sorted columns: a row is searched for the window's first entry,
// and the stored values of a column are summed.
template <typename In, typename P, typename I>
struct EdCenterCsr {
    static constexpr int CW = 64;
    const In* data;
    const I* indices;
    const P* indptr;
    __device__ void fill(In* win, long long row, int c0, int nw) const {
        for (int c = 0; c < nw; ++c) win[c] = In(0);
        P j = indptr[row];
        const P end = indptr[row + 1];
        for (P hi = end; j < hi;) {
            const P mid = j + (hi - j) / 2;
            if (indices[mid] < c0) {
                j = mid + 1;
            } else {
                hi = mid;
            }
        }
        for (; j < end && indices[j] - c0 < nw; ++j)
            // true for sorted input; keeps unsorted columns inside the window
            if (indices[j] >= c0) win[indices[j] - c0] += data[j];
    }
};

// The largest (row 0) and smallest (row 1) kept value of each column from the
// keys of pass 1, less mu: rounding is monotone, so a column's largest
// |x - mu[c]| is at one of the two.
template <typename In>
struct EdCenterReach {
    static constexpr int CW = 64;
    const In* ext;
    const In* mu;
    long long d;
    __device__ void fill(In* win, long long row, int c0, int nw) const {
        for (int c = 0; c < nw; ++c) {
            const In x = edistance_unkey(edistance_bits(ext[row * d + c0 + c]));
            win[c] = (row ? -x : x) - mu[c0 + c];
        }
    }
};

template <typename In, class R>
__global__ void edistance_center_kernel(const R rd, long long step, int m,
                                        int d, int cw, int pass, int shift,
                                        const bool* __restrict__ keep,
                                        In* __restrict__ ext,
                                        In* __restrict__ out,
                                        double* __restrict__ sums) {
    const unsigned u = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = u % m;
    const long long c0 = (long long)(u / m) * cw;
    if (c0 >= d || (pass && !keep[i])) return;
    const int nw = min((long long)cw, d - c0);
    In win[R::CW];
    rd.fill(win, i * step, (int)c0, nw);
    auto top = edistance_bits(In(0));
    auto* keys = reinterpret_cast<decltype(top)*>(ext) + c0;
    // the threads of a warp (rows i, i + 1, ...) start at different columns, so
    // they do not add to one sum at a time
    for (int k = 0, c = i % nw; k < nw; ++k, c = c + 1 < nw ? c + 1 : 0) {
        if (pass) {
            const In q = rint(ldexp(win[c], shift));
            if (q != 0) atomicAdd(sums + c0 + c, (double)q);
            // a plain read first: only a key above the stored one issues an
            // atomic
            const auto up = edistance_key(win[c]), down = ~up;
            if (up > keys[c]) atomicMax(keys + c, up);
            if (down > keys[d + c]) atomicMax(keys + d + c, down);
        } else {
            top = max(top, edistance_bits(fabs(win[c])));
        }
    }
    if (!pass) atomicMax(reinterpret_cast<decltype(top)*>(out) + i, top);
}
