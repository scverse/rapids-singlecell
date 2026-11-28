#include <cuda/std/algorithm>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <thrust/iterator/transform_iterator.h>

#include "wilcoxon_common.cuh"

namespace {
using Key = unsigned long long;
using Rank = long long;

// All rows form the ranking population for one-vs-rest, or only reference rows
// for a named comparison. The prefix selects how many population records a
// range contains.
__device__ int population(const int* prefix, int lo, int hi) {
    return prefix ? prefix[hi] - prefix[lo] : hi - lo;
}

struct IsReference {
    int mask, group;
    __device__ int operator()(Key key) const {
        return (key & mask) == group;
    }
};

__device__ int lower(const Key* keys, int lo, int hi, Key target) {
    if (lo == hi || keys[lo] >= target) return lo;
    if (keys[hi - 1] < target) return hi;
    return cuda::std::lower_bound(keys + lo, keys + hi, target) - keys;
}

struct ZeroRun {
    int missing, count;
    Rank rank2;
};

__global__ void find_zeros(const Key* keys, const int* ptr, const int* prefix,
                           int n, int cols, int shift, ZeroRun* zeros) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    int a = ptr[col], b = ptr[col + 1];
    Key zero = ((Key)col << 32) | 0x80000000u;
    int lo = lower(keys, a, b, zero << shift);
    int hi = lower(keys, lo, b, (zero + 1) << shift);
    int missing = n - population(prefix, a, b);
    int count = missing + population(prefix, lo, hi);
    zeros[col] = {missing, count, 2LL * population(prefix, a, lo) + count};
}

// One leader handles each run of equal (gene, value, group) keys.
__global__ void accumulate_runs(const Key* keys, const int* ptr,
                                const int* prefix, const ZeroRun* zeros,
                                int nnz, int width, int shift, int ref,
                                bool correct, double scale, Key* rank2,
                                double* sums, double* ties, int* nonzero) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz || (i && keys[i - 1] == keys[i])) return;
    Key key = keys[i], value = key >> shift;
    unsigned order = (unsigned)value;
    if (order == 0x80000000u) return;
    int col = value >> 32, g = key & ((1u << shift) - 1), a = ptr[col],
        b = ptr[col + 1];
    int lo = lower(keys, a, i, value << shift);
    int end = lower(keys, i + 1, b, key + 1);
    int hi = lower(keys, end, b, (value + 1) << shift);
    int copies = end - i, below = population(prefix, a, lo),
        equal = population(prefix, lo, hi);
    if (order > 0x80000000u) below += zeros[col].missing;
    size_t out = (size_t)col * width + g;
    // Start each cell at the zero midrank and accumulate only the exact
    // differences.
    Rank delta = 2LL * below + equal - zeros[col].rank2;
    atomicAdd(rank2 + out, (Key)(copies * delta));
    if (nonzero) atomicAdd(nonzero + out, copies);
    float x = __uint_as_float(order ^ (order >> 31 ? 0x80000000u : ~0u));
    atomicAdd(sums + out, copies * stat_value(x, scale));
    if (!correct) return;
    if (ref < 0) {
        if (i == lo && equal > 1) atomicAdd(ties + col, tie3(equal));
    } else if (g == ref || equal == 0) {
        if (copies > 1) atomicAdd(ties + out, tie3(copies));
    } else {
        atomicAdd(ties + out,
                  tie3(copies) + 3.0 * copies * equal * (copies + equal));
    }
}

__global__ void finish_runs(const ZeroRun* zeros, const int* sizes,
                            const int* nonzero, int width, int n, int ref,
                            bool correct, Key* ranks, double* ties) {
    int col = blockIdx.x;
    ZeroRun zero = zeros[col];
    if (threadIdx.x == 0 && ref < 0)
        ties[col] =
            correct ? finalize_tie_corr(n, ties[col] + tie3(zero.count)) : 1.;
    for (int g = threadIdx.x; g < width; g += blockDim.x) {
        if (g == ref) continue;
        size_t out = (size_t)col * width + g;
        Rank size = sizes[g];
        // For one-vs-rest, remove the target's self ranks before conversion to
        // float64. A large, imbalanced group can otherwise lose a half-rank.
        ranks[out] =
            (Rank)ranks[out] + size * (zero.rank2 - (ref < 0 ? size : 0));
        if (ref >= 0) {
            double tie = correct ? ties[out] + ties[(size_t)col * width + ref] +
                                       tie3(zero.count + size - nonzero[out])
                                 : 0.;
            ties[out] = correct ? finalize_tie_corr(n + size, tie) : 1.;
        }
    }
}

template <typename T, typename Device, int N = 1>
using Array = nb::ndarray<T, Device, nb::c_contig, nb::ndim<N>>;

template <typename... Arrays>
void check_current_device(const Arrays&... arrays) {
    int device;
    cuda_check(cudaGetDevice(&device), "Wilcoxon device");
    nb_require(((arrays.device_id() == device || !arrays.size()) && ...),
               "Wilcoxon records belong to another device");
}

int key_bits(int n) {
    return n < 2 ? 0 : 32 - __builtin_clz(n - 1);
}

template <typename Device>
void rank_panel(Array<Key, Device> keys, Array<const int, Device> ptr,
                Array<const int, Device> sizes, Array<double, Device, 2> sums,
                Array<Key, Device, 2> ranks, gpu_array_c<double, Device> ties,
                gpu_array_c<int, Device> counts, int n, int ref, bool correct,
                double scale, std::uintptr_t stream_ptr) {
    check_current_device(keys, ptr, sizes, sums, ranks, ties, counts);
    nb_require(keys.size() <= INT_MAX && ptr.size() >= 2 &&
                   ptr.size() - 1 <= INT_MAX && sizes.size() >= 1 &&
                   sizes.size() <= 65536 && n > 0,
               "invalid packed Wilcoxon panel");
    int cols = ptr.size() - 1, width = sizes.size(), nnz = keys.size();
    int group_bits = std::max(1, key_bits(width)),
        panel_bits = key_bits(cols) + group_bits;
    size_t cells = (size_t)cols * width;
    nb_require(sums.shape(0) == cols && sums.shape(1) == width &&
                   ranks.shape(0) == cols && ranks.shape(1) == width &&
                   ref >= -1 && ref < width &&
                   ties.size() == (ref >= 0 ? cells : (size_t)cols) &&
                   (counts.size() == cells ||
                    (!counts.size() && !(ref >= 0 && correct))) &&
                   panel_bits <= 32,
               "invalid packed Wilcoxon output, reference or key width");
    auto stream = (cudaStream_t)stream_ptr;
    RmmScratchPool pool;
    int* reference = ref < 0 ? nullptr : pool.alloc<int>((size_t)nnz + 1);
    if (reference)
        cuda_check(cudaMemsetAsync(reference, 0, sizeof(int), stream),
                   "reference prefix");
    Key* sorted = keys.data();
    if (nnz) {
        cub::DoubleBuffer<Key> buffers(sorted, pool.alloc<Key>(nnz));
        size_t sort_bytes = 0, scan_bytes = 0;
        auto sort_and_scan = [&](void* workspace) {
            cuda_check(
                cub::DeviceRadixSort::SortKeys(workspace, sort_bytes, buffers,
                                               nnz, 0, 32 + panel_bits, stream),
                "Wilcoxon sort");
            if (reference) {
                auto flags = thrust::make_transform_iterator(
                    buffers.Current(), IsReference{(1 << group_bits) - 1, ref});
                cuda_check(
                    cub::DeviceScan::InclusiveSum(workspace, scan_bytes, flags,
                                                  reference + 1, nnz, stream),
                    "reference scan");
            }
        };
        sort_and_scan(nullptr);
        sort_and_scan(pool.alloc<char>(std::max(sort_bytes, scan_bytes)));
        sorted = buffers.Current();
    }
    ZeroRun* zeros = pool.alloc<ZeroRun>(cols);
    find_zeros<<<strided_grid(cols, 256), 256, 0, stream>>>(
        sorted, ptr.data(), reference, n, cols, group_bits, zeros);
    CUDA_CHECK_LAST_ERROR(find_zeros);
    if (nnz) {
        accumulate_runs<<<strided_grid(nnz, 256), 256, 0, stream>>>(
            sorted, ptr.data(), reference, zeros, nnz, width, group_bits, ref,
            correct, scale, ranks.data(), sums.data(), ties.data(),
            counts.size() ? counts.data() : nullptr);
        CUDA_CHECK_LAST_ERROR(accumulate_runs);
    }
    finish_runs<<<cols, 256, 0, stream>>>(zeros, sizes.data(), counts.data(),
                                          width, n, ref, correct, ranks.data(),
                                          ties.data());
    CUDA_CHECK_LAST_ERROR(finish_runs);
    cuda_check(cudaStreamSynchronize(stream), "packed Wilcoxon finish");
}

__device__ unsigned dictionary_slot(unsigned key, unsigned* table,
                                    unsigned mask, int* failed) {
    if (!key) return 0;  // Slot zero represents +0.
    unsigned hash = (key ^ (key >> 16)) * 0x7feb352du;
    hash ^= hash >> 15;
    for (int probe = 0; probe < 128; ++probe) {
        unsigned slot = (hash + probe) & mask;
        if (!slot) continue;
        unsigned old = table[slot];
        if (!old) old = atomicCAS(table + slot, 0u, key);
        if (!old || old == key) return slot;
    }
    atomicExch(failed, 1);
    return 0;
}

// A warp reads consecutive CSR entries and writes one gene across 32 cells.
// The count pass and the write pass visit stripes in the same row order.
__global__ void transpose_tile(const int* ptr, const int* indices,
                               const float* data, const uint16_t* labels,
                               int rows, int tiles_per, int* prefix,
                               int* failed, int first, int stop, int offset,
                               unsigned* words, uint16_t* groups,
                               unsigned* table, unsigned mask, int bits) {
    __shared__ unsigned tile[128][33];
    __shared__ uint8_t present[128][33];
    __shared__ int cursor[128];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int col0 = first + 128 * blockIdx.y, col = col0 + threadIdx.x;
    size_t k = (size_t)col * gridDim.x + blockIdx.x;
    if (threadIdx.x < 128 && col < stop)
        cursor[threadIdx.x] = bits >= 0 && k ? prefix[k - 1] - offset : 0;
    for (int t = 0; t < tiles_per; ++t) {
        Rank row0 = ((Rank)blockIdx.x * tiles_per + t) * 32;
        if (row0 >= rows) break;
        for (int i = threadIdx.x; i < 4096; i += blockDim.x)
            present[i / 32][i % 32] = 0;
        __syncthreads();
        for (int r = warp; r < 32 && row0 + r < rows; r += blockDim.x / 32) {
            int a = ptr[row0 + r], b = ptr[row0 + r + 1];
            if (!lane)
                a = cuda::std::lower_bound(indices + a, indices + b, col0) -
                    indices;
            a = __shfl_sync(0xffffffff, a, 0);
            for (Rank p = (Rank)a + lane; p < b; p += 32) {
                int j = indices[p] - col0;
                if (j >= min(128, stop - col0)) break;
                present[j][r] = 1;
                if (bits >= 0) tile[j][r] = __float_as_uint(data[p]);
            }
        }
        __syncthreads();
        for (int j = warp; j < 128 && col0 + j < stop; j += blockDim.x / 32) {
            unsigned lanes = __ballot_sync(0xffffffff, present[j][lane]);
            int base = __shfl_sync(0xffffffff, lane ? 0 : cursor[j], 0);
            if (!lane) cursor[j] = base + __popc(lanes);
            if (bits < 0 || !present[j][lane]) continue;
            int pos = base + __popc(lanes & ((1u << lane) - 1));
            unsigned key = tile[j][lane], group = labels[row0 + lane];
            if (bits) {
                // A single word holds a range base, larger tables hold
                // dictionary entries.
                unsigned index = mask
                                     ? dictionary_slot(key, table, mask, failed)
                                     : key - table[0];
                key = index << bits | group;
            } else
                groups[pos] = group;
            words[pos] = key;
        }
        __syncthreads();
    }
    if (bits < 0 && threadIdx.x < 128 && col < stop)
        prefix[k] = cursor[threadIdx.x];
}

template <typename Device>
void transpose_csr(
    Array<const int, Device> ptr, Array<const int, Device> indices,
    Array<const float, Device> data, Array<const uint16_t, Device> labels,
    int tiles_per, Array<int, Device, 2> prefix, Array<int, Device> failed,
    int first, int stop, int offset, Array<unsigned, Device> words,
    Array<uint16_t, Device> groups, Array<unsigned, Device> table, int bits,
    std::uintptr_t stream) {
    check_current_device(ptr, indices, data, labels, prefix, failed, words,
                         groups, table);
    size_t rows = labels.size(), slots = table.size();
    nb_require(
        rows > 0 && rows <= INT_MAX && ptr.size() == rows + 1 &&
            indices.size() == data.size() && data.size() <= INT_MAX &&
            failed.size() == 1 && first >= 0 && first < stop &&
            (size_t)stop <= prefix.shape(0) && stop - first <= 128 * 65535 &&
            tiles_per > 0 &&
            prefix.shape(1) == ((rows + 31) / 32 + tiles_per - 1) / tiles_per &&
            offset >= 0 && (size_t)offset + words.size() <= data.size(),
        "invalid CSR transpose shape or range");
    nb_require(bits >= -1 && bits <= 16 &&
                   (bits || groups.size() == words.size()) &&
                   (bits <= 0 || (slots && !(slots & (slots - 1)) &&
                                  slots <= (1ULL << (32 - bits)))),
               "invalid Wilcoxon record buffers");
    dim3 grid(prefix.shape(1), (stop - first + 127) / 128);
    transpose_tile<<<grid, 256, 0, (cudaStream_t)stream>>>(
        ptr.data(), indices.data(), data.data(), labels.data(), rows, tiles_per,
        prefix.data(), failed.data(), first, stop, offset, words.data(),
        groups.data(), table.data(), slots - 1, bits);
    CUDA_CHECK_LAST_ERROR(transpose_tile);
}

// The radix key orders a record by gene, exact float32 value, then group.
// Both signs of zero share a rank. Nonfinite entries select the eager fallback.
__global__ void gather_keys(const unsigned* words, const unsigned* table,
                            const uint16_t* codes, int bits, const int* src,
                            const int* dst, Key* keys, int* invalid, int groups,
                            int group_bits, int n_words, int n_keys,
                            unsigned slots) {
    int col = blockIdx.y, start = src[col], stop = src[col + 1],
        dest = dst[col];
    if (start < 0 || stop < start || stop > n_words || dest < 0 ||
        dest > n_keys - (stop - start)) {
        if (!threadIdx.x) atomicExch(invalid, 1);
        return;
    }
    for (Rank i = start + (Rank)blockIdx.x * blockDim.x + threadIdx.x; i < stop;
         i += gridDim.x * blockDim.x) {
        unsigned word = words[i], index = word >> bits;
        unsigned group = bits ? word & ((1u << bits) - 1) : codes[i];
        if (group >= groups || (bits && slots > 1 && index >= slots)) {
            atomicExch(invalid, 1);
            continue;
        }
        unsigned value = word;
        if (bits) value = slots == 1 ? table[0] + index : table[index];
        if (!isfinite(__uint_as_float(value))) atomicExch(invalid, 1);
        if (!(value << 1)) value = 0;
        unsigned order = value ^ (value >> 31 ? ~0u : 0x80000000u);
        keys[dest + (i - start)] =
            (((Key)col << 32 | order) << group_bits) | group;
    }
}

template <typename Device>
void gather_rank_keys(Array<const unsigned, Device> words,
                      Array<const unsigned, Device> table,
                      Array<const uint16_t, Device> codes, int bits,
                      Array<const int, Device> src,
                      Array<const int, Device> dst, Array<Key, Device> keys,
                      Array<int, Device> invalid, int groups,
                      std::uintptr_t stream) {
    check_current_device(words, table, codes, src, dst, keys, invalid);
    const int group_bits = std::max(1, key_bits(groups));
    nb_require(
        src.size() == dst.size() + 1 && dst.size() && dst.size() <= 65535 &&
            key_bits(dst.size()) + group_bits <= 32 &&
            words.size() <= keys.size() && keys.size() <= INT_MAX &&
            invalid.size() == 1 && groups > 0 && groups <= 65536 && bits >= 0 &&
            bits <= 16 &&
            (bits ? table.size() > 0 && table.size() <= (1ULL << (32 - bits))
                  : codes.size() == words.size()),
        "invalid Wilcoxon gather buffers");
    gather_keys<<<dim3(32, dst.size()), 256, 0, (cudaStream_t)stream>>>(
        words.data(), table.data(), codes.data(), bits, src.data(), dst.data(),
        keys.data(), invalid.data(), groups, group_bits, words.size(),
        keys.size(), table.size());
    CUDA_CHECK_LAST_ERROR(gather_keys);
}

template <typename Device>
void define_panel(nb::module_& m) {
    nb::call_guard<nb::gil_scoped_release> release;
    m.def("transpose_csr", &transpose_csr<Device>, release);
    m.def("gather_rank_keys", &gather_rank_keys<Device>, release);
    m.def("rank_panel", &rank_panel<Device>, release);
}
}  // namespace

void register_wilcoxon_panel(nb::module_& m) {
    REGISTER_GPU_BINDINGS(define_panel, m);
}
