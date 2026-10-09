#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

// Diffusion simulation of squidpy's `sepal` (`_diffusion`).
//
// Every cell uses the Laplacian of a saturated source cell: saturated cells
// their own, unsaturated cells that of their nearest saturated cell.
//
//   nbrs: (K, n_cells)  neighbours of the cell's source
//   ctr:  (n_cells,)    the source; ctr[j] == j iff cell j is saturated
//
// One explicit Euler step per iteration, clipped at zero. A gene converges at
// the first iteration where the change of the normalised entropy of the
// saturated cells is <= thresh; `results` holds it, or -1.
//
// The entropy change is a tiny difference of two entropies of order log(n),
// so it is accumulated directly from each cell's change d in a
// cancellation-free form (SepalDelta) instead. In float32 each value is also
// a compensated pair hi + lo, since the per-step increments fall far below
// the value's ulp; neighbours only read hi. The entropy terms are float32 for
// either precision.
//
// One gene per thread block cluster (CLUSTER = false: per block), each block
// simulating a contiguous range of cells. STORE selects where the hi values
// live: SEPAL_SMEM2 two shared-memory buffers, SEPAL_SMEM1 one, with new
// values staged in global memory, SEPAL_GLOBAL two global buffers. Persistent
// groups take genes from `counter`. Convergence is evaluated every
// SEPAL_WINDOW iterations from per-warp partial sums, one iteration per lane;
// iterations simulated past convergence do not change the result.

namespace cg = cooperative_groups;

enum SepalStore { SEPAL_SMEM2 = 0, SEPAL_SMEM1 = 1, SEPAL_GLOBAL = 2 };
constexpr int SEPAL_WINDOW = 16;
// squidpy's entropy guard: np.finfo(np.float64).eps
__device__ constexpr double SEPAL_EPS = 2.220446049250313e-16;

// The group simulating one gene: a cluster (sm_90+) or a single block.
template <bool CLUSTER>
struct SepalGroup {
    static __device__ unsigned rank() {
#if __CUDA_ARCH__ >= 900
        if constexpr (CLUSTER) return cg::this_cluster().block_rank();
#endif
        return 0;
    }
    static __device__ unsigned size() {
#if __CUDA_ARCH__ >= 900
        if constexpr (CLUSTER) return cg::this_cluster().num_blocks();
#endif
        return 1;
    }
    static __device__ void sync() {
#if __CUDA_ARCH__ >= 900
        if constexpr (CLUSTER) {
            cg::this_cluster().sync();
            return;
        }
#endif
        __syncthreads();
    }
    // the same shared-memory variable in block `r` of the group
    template <typename T>
    static __device__ T* map(T* p, unsigned r) {
#if __CUDA_ARCH__ >= 900
        if constexpr (CLUSTER) return cg::this_cluster().map_shared_rank(p, r);
#endif
        return p;
    }
};

// Sum hi + lo: TwoSum in float (~44 bits from FP32 arithmetic), plain double.
template <typename V>
struct SepalSum {
    V hi = 0, lo = 0;
    __device__ __forceinline__ void add(SepalSum o) {
        if constexpr (sizeof(V) == 4) {
            const V s = hi + o.hi, bp = s - hi;
            const V e = (hi - (s - bp)) + (o.hi - bp) + lo + o.lo;
            hi = s + e;
            lo = e - (hi - s);
        } else {
            hi += o.hi;
        }
    }
    __device__ __forceinline__ double value() const {
        return static_cast<double>(hi) + lo;
    }
};

template <typename V>
__device__ __forceinline__ SepalSum<V> sepal_shfl_down(SepalSum<V> a, int off) {
    return {__shfl_down_sync(0xffffffff, a.hi, off),
            __shfl_down_sync(0xffffffff, a.lo, off)};
}

// Change of S = sum(x) and L = sum(x log x) over this thread's saturated
// cells, in float32 for either state precision (FP64 transcendentals are slow
// on many GPUs), Kahan-summed with the compensation kept as lo.
struct SepalDelta {
    SepalSum<float> ds, dl;

    static __device__ __forceinline__ void kahan(SepalSum<float>& acc,
                                                 float x) {
        const float y = x + acc.lo, t = acc.hi + y;
        acc.lo = y - (t - acc.hi);
        acc.hi = t;
    }

    // d * log(new) + old * log1p(d / old) == new log new - old log old
    __device__ __forceinline__ void add(float old_v, float new_v, float d) {
        if (d == 0.0f) return;
        float g = 0.0f;
        if (old_v > 0.0f && new_v > 0.0f) {
            const float x = d / old_v;
            // four terms of log1p are exact to float rounding for |x| <= 2^-6
            const float tail =
                fabsf(x) <= 0.015625f
                    ? d * (1.0f + x * (-0.5f + x * (1.0f / 3 + x * -0.25f)))
                    : old_v * log1pf(x);
            g = d * __logf(new_v) + tail;
        } else if (new_v > 0.0f) {
            g = new_v * __logf(new_v);
        } else if (old_v > 0.0f) {
            g = -old_v * __logf(old_v);
        }
        kahan(ds, d);
        kahan(dl, g);
    }
};

// One explicit Euler step of the value hi + lo, clipped at zero; returns d.
__device__ __forceinline__ float sepal_step(float& hi, float& lo, float inc) {
    const float y = inc + lo, s = hi + y, bp = s - hi;
    const float e = (hi - (s - bp)) + (y - bp);
    float new_hi = s + e, new_lo = e - (new_hi - s);
    if (new_hi < 0.0f || (new_hi == 0.0f && new_lo < 0.0f)) new_hi = new_lo = 0;
    const float d = (new_hi - hi) + (new_lo - lo);
    hi = new_hi;
    lo = new_lo;
    return d;
}

__device__ __forceinline__ double sepal_step(double& hi, double&, double inc) {
    const double old_v = hi;
    hi = hi + inc;
    if (hi < 0.0) hi = 0.0;
    return hi - old_v;
}

template <int K, typename V>
__device__ __forceinline__ V sepal_laplacian(V nbr_sum, V center) {
    if constexpr (K == 4) {
        return nbr_sum - V(4) * center;
    } else {
        // numba's fastmath turns squidpy's `/ 3.0` into a reciprocal multiply
        return (V(2) * nbr_sum - V(12) * center) * (V(1) / V(3));
    }
}

__device__ __forceinline__ double sepal_entropy(double s, double l) {
    return s >= SEPAL_EPS ? log(s) - l / s : 0.0;
}

// Shared memory per block besides the hi buffers.
__host__ __device__ constexpr size_t sepal_scratch_bytes(int n_warps) {
    // window slots per warp, group totals per lane (also 64 doubles)
    return (SEPAL_WINDOW * n_warps + 64) * 2 * sizeof(SepalSum<float>);
}

template <typename V, int K, bool CLUSTER, int STORE>
__global__ void __launch_bounds__(1024)
    sepal_kernel(const V* __restrict__ conc, V* __restrict__ buf_all,
                 V* __restrict__ lo_all, const int* __restrict__ nbrs,
                 const int* __restrict__ ctr, int* __restrict__ results,
                 int* __restrict__ counter, int n_genes, int n_cells, int chunk,
                 int n_sat, int n_iter, double dt, double thresh) {
    using G = SepalGroup<CLUSTER>;
    using S = SepalSum<float>;
    const unsigned rank = G::rank(), n_ranks = G::size();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n_warps = blockDim.x >> 5;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    S* slots = reinterpret_cast<S*>(smem_raw);       // [WINDOW][n_warps][2]
    S* totals = slots + 2 * SEPAL_WINDOW * n_warps;  // [32][2]
    V* sbuf = reinterpret_cast<V*>(totals + 128);    // chunk (x2 for SMEM2)
    __shared__ int gene_sh, flag_sh;
    __shared__ double base_s, base_l;  // S and L before the window (rank 0)

    const long long group = blockIdx.x / n_ranks;
    const int base = rank * chunk, end = min(n_cells, base + chunk);
    V* lo = lo_all + group * n_cells;           // owner-only, float32 only
    V* gbuf = buf_all + group * 2LL * n_cells;  // SMEM1: stage, GLOBAL: buffers
    const V dt_v = static_cast<V>(dt);

    while (true) {
        if (rank == 0 && threadIdx.x == 0) gene_sh = atomicAdd(counter, 1);
        G::sync();
        const int gene = *G::map(&gene_sh, 0);
        G::sync();  // gene_sh read everywhere before it changes
        if (gene >= n_genes) break;

        V* cin = STORE == SEPAL_GLOBAL ? gbuf : sbuf;
        V* cout = STORE == SEPAL_SMEM2    ? sbuf + chunk
                  : STORE == SEPAL_GLOBAL ? gbuf + n_cells
                                          : sbuf;
        const int off = STORE == SEPAL_GLOBAL ? 0 : base;  // buffer origin
        const V* src = conc + static_cast<long long>(gene) * n_cells;
        double s = 0.0, l = 0.0;
        for (int j = base + threadIdx.x; j < end; j += blockDim.x) {
            const V x = src[j];
            if constexpr (STORE == SEPAL_GLOBAL) {
                __stcg(cin + j, x);
            } else {
                cin[j - off] = x;
            }
            if constexpr (sizeof(V) == 4) lo[j] = 0;
            if (__ldg(ctr + j) == j && x > V(0)) {
                s += x;
                l += x * log(static_cast<double>(x));
            }
        }
        // S and L of the initial state, summed over the group by rank 0
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            s += __shfl_down_sync(0xffffffff, s, o);
            l += __shfl_down_sync(0xffffffff, l, o);
        }
        double* red = reinterpret_cast<double*>(totals);
        if (lane == 0) {
            red[2 * warp] = s;
            red[2 * warp + 1] = l;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 1; w < n_warps; ++w) {
                s += red[2 * w];
                l += red[2 * w + 1];
            }
            red[0] = s;
            red[1] = l;
        }
        G::sync();
        if (rank == 0 && threadIdx.x == 0) {
            base_s = base_l = 0.0;
            for (unsigned r = 0; r < n_ranks; ++r) {
                const double* p = G::map(red, r);
                base_s += p[0];
                base_l += p[1];
            }
        }
        G::sync();  // red is reused as totals

        // value of cell q in the current state
        auto read = [&](int q) -> V {
            if constexpr (STORE == SEPAL_GLOBAL) return __ldcg(cin + q);
            if (q >= base && q < end) return cin[q - base];
            const unsigned r = q / chunk;
            return *G::map(cin + (q - r * chunk), r);
        };

        int result = -1, first = 0;
        for (int it = 0; it < n_iter; ++it) {
            SepalDelta delta;
            for (int j = base + threadIdx.x; j < end; j += blockDim.x) {
                const int sc = __ldg(ctr + j);
                V nb = read(__ldg(nbrs + j));
#pragma unroll
                for (int m = 1; m < K; ++m)
                    nb += read(
                        __ldg(nbrs + static_cast<long long>(m) * n_cells + j));
                V hi = STORE == SEPAL_GLOBAL ? __ldcg(cin + j) : cin[j - off];
                V lo_j = sizeof(V) == 4 ? lo[j] : V(0);
                const V old_v = hi + lo_j;
                const V d = sepal_step(hi, lo_j,
                                       sepal_laplacian<K>(nb, read(sc)) * dt_v);
                if constexpr (STORE == SEPAL_SMEM1) {
                    gbuf[j] = hi;
                } else if constexpr (STORE == SEPAL_GLOBAL) {
                    __stcg(cout + j, hi);
                } else {
                    cout[j - off] = hi;
                }
                if constexpr (sizeof(V) == 4) lo[j] = lo_j;
                if (sc == j)
                    delta.add(static_cast<float>(old_v),
                              static_cast<float>(hi + lo_j),
                              static_cast<float>(d));
            }
            // per-warp partial sums of this iteration
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                delta.ds.add(sepal_shfl_down(delta.ds, o));
                delta.dl.add(sepal_shfl_down(delta.dl, o));
            }
            if (lane == 0) {
                S* p = slots + 2 * ((it % SEPAL_WINDOW) * n_warps + warp);
                p[0] = delta.ds;
                p[1] = delta.dl;
            }
            G::sync();  // every read of the old state is done
            if constexpr (STORE == SEPAL_SMEM1) {
                for (int j = base + threadIdx.x; j < end; j += blockDim.x)
                    cin[j - base] = gbuf[j];
            } else {
                V* t = cin;
                cin = cout;
                cout = t;
            }
            const int count = it + 1 - first;
            if (count < SEPAL_WINDOW && it + 1 < n_iter) {
                if constexpr (STORE == SEPAL_SMEM1) G::sync();
                continue;
            }

            // Evaluate the window: lane i of warp 0 takes iteration first + i.
            if (warp == 0) {
                S a, b;
                if (lane < count) {
                    const S* p =
                        slots + 2 * ((first + lane) % SEPAL_WINDOW) * n_warps;
                    for (int w = 0; w < n_warps; ++w, p += 2) {
                        a.add(p[0]);
                        b.add(p[1]);
                    }
                }
                totals[2 * lane] = a;
                totals[2 * lane + 1] = b;
            }
            G::sync();
            if (rank == 0 && warp == 0) {
                S a, b;
                for (unsigned r = 0; r < n_ranks; ++r) {
                    const S* p = G::map(totals, r) + 2 * lane;
                    a.add(p[0]);
                    b.add(p[1]);
                }
                const double ds = lane < count ? a.value() : 0.0;
                const double dl = lane < count ? b.value() : 0.0;
                double s_new = ds, l_new = dl;  // inclusive scan
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const double us = __shfl_up_sync(0xffffffff, s_new, o);
                    const double ul = __shfl_up_sync(0xffffffff, l_new, o);
                    if (lane >= o) {
                        s_new += us;
                        l_new += ul;
                    }
                }
                s_new += base_s;
                l_new += base_l;
                const double s_old = s_new - ds, l_old = l_new - dl;
                double diff;
                if (first + lane == 0) {  // squidpy starts from entropy 1.0
                    diff = fabs(sepal_entropy(s_new, l_new) / n_sat - 1.0);
                } else if (s_old < SEPAL_EPS || s_new < SEPAL_EPS) {
                    diff = fabs(sepal_entropy(s_new, l_new) -
                                sepal_entropy(s_old, l_old)) /
                           n_sat;
                } else {  // log(S'/S) - (L'/S' - L/S) from the changes
                    diff = fabs(log1p(ds / s_old) -
                                (dl * s_old - l_old * ds) / (s_new * s_old)) /
                           n_sat;
                }
                const unsigned conv =
                    __ballot_sync(0xffffffff, lane < count && diff <= thresh);
                const double last_s = __shfl_sync(0xffffffff, s_new, count - 1);
                const double last_l = __shfl_sync(0xffffffff, l_new, count - 1);
                if (lane == 0) {
                    base_s = last_s;
                    base_l = last_l;
                    flag_sh = conv ? first + __ffs(conv) - 1 : -1;
                }
            }
            G::sync();
            result = *G::map(&flag_sh, 0);
            if (result >= 0) break;
            first = it + 1;
        }
        if (rank == 0 && threadIdx.x == 0) results[gene] = result;
        G::sync();  // buffers and flag_sh are reused for the next gene
    }
}

// For every unsaturated cell, the first neighbour (in CSR order) that is
// saturated, or -1 if none is.
__global__ void sepal_first_sat_neighbor_kernel(
    const int* __restrict__ unsat, const int* __restrict__ indptr,
    const int* __restrict__ indices, const bool* __restrict__ sat_mask,
    int* __restrict__ nearest, int n_unsat) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_unsat) return;
    const int cell = unsat[i];
    int found = -1;
    for (int k = indptr[cell]; k < indptr[cell + 1]; ++k) {
        if (sat_mask[indices[k]]) {
            found = indices[k];
            break;
        }
    }
    nearest[i] = found;
}

// For each query cell, the saturated cell with the smallest L1 distance; ties
// resolve to the lowest position in `sat`, like np.argmin. One block per query.
__global__ void sepal_nearest_sat_l1_kernel(const double* __restrict__ spatial,
                                            const int* __restrict__ query,
                                            const int* __restrict__ sat,
                                            int* __restrict__ nearest,
                                            int n_sat) {
    __shared__ double best_d[32];
    __shared__ int best_i[32];
    const int cell = query[blockIdx.x];
    const double qx = spatial[2LL * cell], qy = spatial[2LL * cell + 1];
    double bd = INFINITY;
    int bi = n_sat;
    for (int i = threadIdx.x; i < n_sat; i += blockDim.x) {
        const int s = sat[i];
        const double d =
            fabs(qx - spatial[2LL * s]) + fabs(qy - spatial[2LL * s + 1]);
        if (d < bd) {  // ascending scan keeps the first minimum
            bd = d;
            bi = i;
        }
    }
    auto argmin = [&]() {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const double od = __shfl_down_sync(0xffffffff, bd, o);
            const int oi = __shfl_down_sync(0xffffffff, bi, o);
            if (od < bd || (od == bd && oi < bi)) {
                bd = od;
                bi = oi;
            }
        }
    };
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    argmin();
    if (lane == 0) {
        best_d[warp] = bd;
        best_i[warp] = bi;
    }
    __syncthreads();
    if (warp == 0) {
        bd = lane < (blockDim.x >> 5) ? best_d[lane] : INFINITY;
        bi = lane < (blockDim.x >> 5) ? best_i[lane] : n_sat;
        argmin();
        // all-NaN distances: np.argmin returns 0
        if (lane == 0) nearest[blockIdx.x] = sat[bi < n_sat ? bi : 0];
    }
}
