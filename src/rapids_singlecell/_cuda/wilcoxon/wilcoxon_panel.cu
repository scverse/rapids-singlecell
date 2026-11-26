#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cuda/std/algorithm>
#include <thrust/iterator/transform_iterator.h>

#include "wilcoxon_common.cuh"

namespace {
using Key = unsigned long long;
using Rank = long long;

struct ZeroRun {
    int implicit, count;
    Rank rank2;
};

// Each key orders a (gene, float32 value, group) record. Both signed zeros
// have the same key. All other finite values keep their full precision.
__global__ void pack(const float* values, const uint16_t* codes, const int* ptr,
                     Key* keys, int nnz, int cols, int groups, int group_bits,
                     int* invalid) {
    for (Rank col = blockIdx.y; col < cols; col += gridDim.y) {
        int start = ptr[col], end = ptr[col + 1];
        if (start < 0 || start > end || end > nnz || (col == 0 && start != 0) ||
            (col == cols - 1 && end != nnz)) {
            atomicExch(invalid, 1);
            continue;
        }
        for (Rank i = (Rank)start + blockIdx.x * blockDim.x + threadIdx.x;
             i < end; i += gridDim.x * blockDim.x) {
            float v = values[i];
            if (!isfinite(v) || codes[i] >= groups) {
                atomicExch(invalid, 1);
                continue;
            }
            unsigned bits = v == 0 ? 0 : __float_as_uint(v);
            unsigned ordered = bits ^ (bits >> 31 ? ~0u : 0x80000000u);
            keys[i] = (((Key)col << 32 | ordered) << group_bits) | codes[i];
        }
    }
}

// Radix sort groups by gene and value. Only the named-reference comparison
// also orders equal values by their group.
__device__ int boundary(const Key* keys, int lo, int hi, Key target) {
    if (hi - lo > 32 && keys[lo + 31] >= target) hi = lo + 32;
    if (hi - lo > 32 && keys[hi - 32] < target) lo = hi - 31;
    return cuda::std::lower_bound(keys + lo, keys + hi, target) - keys;
}

struct IsReference {
    int mask, group;
    __device__ int operator()(Key key) const {
        return (key & mask) == group;
    }
};

__global__ void find_zeros(const Key* keys, const int* ptr, const int* ref,
                           ZeroRun* zeros, int n, int cols, int group_bits) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    int a = ptr[col], b = ptr[col + 1];
    Key zero = (Key)col << 32 | 0x80000000u;
    int lo = boundary(keys, a, b, zero << group_bits);
    int hi = boundary(keys, lo, b, (zero + 1) << group_bits);
    int missing = n - (ref ? ref[b] - ref[a] : b - a);
    int count = missing + (ref ? ref[hi] - ref[lo] : hi - lo);
    int below = ref ? ref[lo] - ref[a] : lo - a;
    zeros[col] = {missing, count, 2LL * below + count + (ref == nullptr)};
}

struct Partial {
    Key delta2;
    double sum, tie;
    int nonzero;
};

__global__ void accumulate(const Key* keys, const int* ptr,
                           const int* reference, const ZeroRun* zeros,
                           Partial* stats, int nnz, int group_bits, int width,
                           int ref, bool correct, double scale, int shards) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Key key = i < nnz ? keys[i] : 0, value = key >> group_bits;
    unsigned order = (unsigned)value;
    bool live = i < nnz && order != 0x80000000u;
    unsigned active = __ballot_sync(0xffffffffu, live);
    if (!live) return;
    int col = value >> 32, group = key & ((1u << group_bits) - 1);
    int start = ptr[col], end = ptr[col + 1];
    // Aggregate equal (value, group) keys within each warp. For a named
    // reference each whole group run contributes its own tie correction once.
    unsigned same = __match_any_sync(active, key);
    if ((i & 31) != __ffs(same) - 1) return;
    int copies = __popc(same);
    if (ref >= 0) {
        if (i > start && keys[i - 1] == key) return;
        if (i + copies < end && keys[i + copies] == key)
            copies = boundary(keys, i + copies, end, key + 1) - i;
    }
    const Key first = value << group_bits, next = (value + 1) << group_bits;
    int lo =
        i > start && keys[i - 1] >= first ? boundary(keys, start, i, first) : i;
    int hi = i + 1 < end && keys[i + 1] < next
                 ? boundary(keys, i + 1, end, next)
                 : i + 1;
    size_t segment = (size_t)col * shards + (blockIdx.x & (shards - 1));
    Partial& out = stats[segment * width + group];
    unsigned bits = order ^ (order >> 31 ? 0x80000000u : ~0u);
    double x = __uint_as_float(bits);
    atomicAdd(&out.sum, copies * stat_value(x, scale));
    // Start every cell at the zero rank, then add its nonzero difference.
    Rank diff = (order > 0x80000000u ? 2LL * zeros[col].implicit : 0) -
                zeros[col].rank2;
    if (ref >= 0) {
        atomicAdd(&out.nonzero, copies);
        if (group == ref) {
            if (correct && copies > 1) atomicAdd(&out.tie, tie3(copies));
            return;
        }
        int below = reference[lo];
        int equal = reference[hi] - below;
        diff += 2LL * (below - reference[start]) + equal;
        // A target run of size t adds (r+t)^3-(r+t) - (r^3-r)
        // to the contribution already counted for the reference run of size r.
        if (correct && (equal || copies > 1))
            atomicAdd(&out.tie,
                      tie3(copies) + 3. * copies * equal * (copies + equal));
    } else {
        diff += (Rank)(lo - start) + (hi - start) + 1;
        if (correct && i == lo && hi - lo > 1)
            atomicAdd(&stats[segment * width].tie, tie3(hi - lo));
    }
    atomicAdd(&out.delta2, (Key)(copies * diff));
}

__global__ void finish(const Partial* stats, const int* sizes,
                       const ZeroRun* zeros, double* sums, double* ties,
                       double* ranks, int n, int width, int cols, int ref,
                       bool correct, int shards) {
    for (size_t out = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         out < (size_t)cols * width; out += (size_t)gridDim.x * blockDim.x) {
        int col = out / width, g = out % width;
        Partial all{};
        for (int s = 0; s < shards; ++s) {
            size_t segment = (size_t)col * shards + s;
            const auto& part = stats[segment * width + g];
            all.delta2 += part.delta2;
            all.sum += part.sum;
            all.nonzero += part.nonzero;
            all.tie += part.tie;
            if (ref >= 0) all.tie += stats[segment * width + ref].tie;
        }
        sums[out] = all.sum;
        if (g == ref) continue;
        Rank size = sizes[g],
             rank2 = size * zeros[col].rank2 + (Rank)all.delta2;
        if (ref >= 0) rank2 += size * (size + 1);
        ranks[out] = (double)rank2 * 0.5;
        if (ref >= 0) {
            int nz = zeros[col].count + (int)size - all.nonzero;
            ties[out] =
                correct ? finalize_tie_corr(n + size, all.tie + tie3(nz)) : 1.;
        } else if (g == 0) {
            ties[col] =
                correct ? finalize_tie_corr(n, all.tie + tie3(zeros[col].count))
                        : 1.;
        }
    }
}

template <typename Run>
void cub_operation(RmmScratchPool& pool, Run run) {
    size_t bytes = 0;
    cuda_check(run(nullptr, bytes), "Wilcoxon scratch query");
    cuda_check(run(pool.alloc<char>(bytes), bytes), "Wilcoxon CUB operation");
}

template <typename T>
T* zeros(RmmScratchPool& pool, size_t n, cudaStream_t stream) {
    auto* p = pool.alloc<T>(n);
    cuda_check(cudaMemsetAsync(p, 0, n * sizeof(T), stream), "Wilcoxon zeros");
    return p;
}

template <typename T, typename Device>
using Array = gpu_array_c<T, Device>;

template <typename Device>
bool rank_panel(Array<const float, Device> values,
                Array<const uint16_t, Device> codes,
                Array<const int, Device> ptr, Array<const int, Device> sizes,
                Array<double, Device> sums, Array<double, Device> ranks,
                Array<double, Device> ties, int n, int ref, bool correct,
                double scale, std::uintptr_t stream_ptr) {
    nb_require(values.size() <= INT_MAX && values.size() == codes.size() &&
                   ptr.size() >= 2 && ptr.size() - 1 <= INT_MAX &&
                   sizes.size() >= 1 && sizes.size() <= 65536 && n > 0,
               "invalid packed Wilcoxon panel");
    int cols = ptr.size() - 1, width = sizes.size(), nnz = values.size();
    auto bits = [](int n) { return n < 2 ? 0 : 32 - __builtin_clz(n - 1); };
    int group_bits = std::max(1, bits(width)),
        end_bit = group_bits + 32 + bits(cols);
    size_t cells = (size_t)cols * width;
    nb_require(sums.size() == cells && ranks.size() == cells && ref >= -1 &&
                   ref < width &&
                   ties.size() == (ref >= 0 ? cells : (size_t)cols) &&
                   end_bit <= 64,
               "invalid packed Wilcoxon output, reference or key width");
    auto stream = (cudaStream_t)stream_ptr;
    RmmScratchPool pool;
    int* invalid = zeros<int>(pool, 1, stream);
    Key* first = pool.alloc<Key>(nnz);
    pack<<<dim3(128, strided_grid_y(cols, 1)), 256, 0, stream>>>(
        values.data(), codes.data(), ptr.data(), first, nnz, cols, width,
        group_bits, invalid);
    CUDA_CHECK_LAST_ERROR(pack);
    int bad = 0;
    cuda_check(cudaMemcpyAsync(&bad, invalid, sizeof(int),
                               cudaMemcpyDeviceToHost, stream),
               "panel validation");
    cuda_check(cudaStreamSynchronize(stream), "panel validation sync");
    if (bad) return false;
    cub::DoubleBuffer<Key> sorted(first, pool.alloc<Key>(nnz));
    int* reference =
        ref < 0 ? nullptr : zeros<int>(pool, (size_t)nnz + 1, stream);
    if (nnz) {
        cub_operation(pool, [&](void* temp, size_t& bytes) {
            auto query = sorted;
            return cub::DeviceRadixSort::SortKeys(
                temp, bytes, temp ? sorted : query, nnz,
                ref < 0 ? group_bits : 0, end_bit, stream);
        });
        if (reference) {
            auto flags = thrust::make_transform_iterator(
                sorted.Current(), IsReference{(1 << group_bits) - 1, ref});
            cub_operation(pool, [&](void* temp, size_t& bytes) {
                return cub::DeviceScan::InclusiveSum(
                    temp, bytes, flags, reference + 1, nnz, stream);
            });
        }
    }
    ZeroRun* zero_runs = pool.alloc<ZeroRun>(cols);
    find_zeros<<<strided_grid(cols, 256), 256, 0, stream>>>(
        sorted.Current(), ptr.data(), reference, zero_runs, n, cols,
        group_bits);
    CUDA_CHECK_LAST_ERROR(find_zeros);
    // Shard contended reductions while keeping the extra storage at most 4
    // bytes per stored entry.
    int shards = nnz >= 256 * cells ? 32 : nnz >= 64 * cells ? 8 : 1;
    Partial* stats = zeros<Partial>(pool, cells * shards, stream);
    if (nnz) {
        accumulate<<<strided_grid(nnz, 256), 256, 0, stream>>>(
            sorted.Current(), ptr.data(), reference, zero_runs, stats, nnz,
            group_bits, width, ref, correct, scale, shards);
        CUDA_CHECK_LAST_ERROR(accumulate);
    }
    finish<<<strided_grid(cells, 256), 256, 0, stream>>>(
        stats, sizes.data(), zero_runs, sums.data(), ties.data(), ranks.data(),
        n, width, cols, ref, correct, shards);
    CUDA_CHECK_LAST_ERROR(finish);
    cuda_check(cudaStreamSynchronize(stream), "packed Wilcoxon finish");
    return true;
}
template <typename Device>
void define_panel(nb::module_& m) {
    m.def("rank_panel", &rank_panel<Device>,
          nb::call_guard<nb::gil_scoped_release>());
}
}  // namespace

void register_wilcoxon_panel(nb::module_& m) {
    REGISTER_GPU_BINDINGS(define_panel, m);
}
