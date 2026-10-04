#pragma once

// Embedded cuTile kernels for Harmony's bfloat16 assignments. Built only with
// CUDA 13+ (see cmake/harmony_cutile.cmake); otherwise every query reports
// "unavailable" and Harmony runs in float32.

#include <cuda_runtime.h>

#include <cstdint>
#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>

#if RSC_CUTILE_ENABLED
#include "harmony_cutile_cubins.h"
#endif

// Module-local (anonymous namespace): each extension module keeps its own
// kernel cache and loads only the cubins it embeds.
namespace harmony_cutile {
namespace {

constexpr int CLUSTER_PAD = 128;  // padded clusters (harmony_tile.py)
// Row tiles and partial-sum blocks per kernel (export_harmony_tile.py).
constexpr int RTZ_LARGE_ROWS = 64, RTZ_LARGE_BLOCKS = 376;
constexpr int RTZ_SMALL_ROWS = 32, RTZ_SMALL_BLOCKS = 188;
constexpr int RTZ_SMALL_MAX_CELLS = 1 << 20;  // below: per-batch variant
constexpr int NEG_RW_ROWS = 128;
constexpr int RTZ_BLOCKS = RTZ_LARGE_BLOCKS;  // partials capacity

inline int pc_width(int n_pcs) {
    return n_pcs <= 64 ? 64 : 128;
}

// Loaded kernel for `symbol` on the current device, or nullptr when the build,
// runtime, architecture or driver cannot run it.
inline cudaKernel_t kernel(const std::string& symbol) {
#if RSC_CUTILE_ENABLED
    int runtime = 0, device = 0, major = 0, minor = 0;
    if (cudaRuntimeGetVersion(&runtime) != cudaSuccess || runtime < 13000 ||
        cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               device) != cudaSuccess)
        return nullptr;
    static std::mutex mutex;
    static std::map<std::pair<int, std::string>, cudaKernel_t> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto key = std::make_pair(device, symbol);
    if (auto it = cache.find(key); it != cache.end()) return it->second;
    cudaKernel_t result = nullptr;
    for (const auto& cubin : kHarmonyCutileCubins) {
        if (cubin.cc_major != major || cubin.cc_minor != minor ||
            symbol != cubin.symbol)
            continue;
        cudaLibrary_t library = nullptr;
        if (cudaLibraryLoadData(&library, cubin.data, nullptr, nullptr, 0,
                                nullptr, nullptr, 0) == cudaSuccess &&
            cudaLibraryGetKernel(&result, library, cubin.symbol) != cudaSuccess)
            result = nullptr;
        break;
    }
    cudaGetLastError();  // a failed load only means "unavailable"
    cache[key] = result;
    return result;
#else
    (void)symbol;
    return nullptr;
#endif
}

inline std::string rtz_symbol(int n_pcs, bool small = false) {
    return std::string("rsc_harmony_rtz_") + (small ? "small" : "large") +
           "_d" + std::to_string(pc_width(n_pcs));
}
inline std::string neg_rw_symbol(int n_pcs) {
    return "rsc_harmony_neg_rw_d" + std::to_string(pc_width(n_pcs));
}

// bfloat16 assignments run when every kernel this module embeds loads for
// this embedding width on the current device.
inline bool available(int n_pcs, int n_clusters) {
#if RSC_CUTILE_ENABLED
    if (n_pcs > 128 || n_clusters > CLUSTER_PAD) return false;
    std::string suffix = "_d" + std::to_string(pc_width(n_pcs));
    bool any = false;
    for (const auto& cubin : kHarmonyCutileCubins) {
        std::string symbol = cubin.symbol;
        if (symbol.size() < suffix.size() ||
            symbol.compare(symbol.size() - suffix.size(), suffix.size(),
                           suffix))
            continue;
        if (!kernel(symbol)) return false;
        any = true;
    }
    return any;
#else
    (void)n_pcs, (void)n_clusters;
    return false;
#endif
}

// cutile_python_v1 calling convention: an array is passed as its pointer,
// then shape and stride per dimension (int32), scalars by value.
struct Args {
    void* values[32];
    alignas(8) int32_t storage[48];
    int count = 0, used = 0;
    void add_ptr(const void* p) {
        static_assert(sizeof(void*) == 8, "64-bit only");
        used += used & 1;  // pointers take an aligned pair of slots
        auto* slot = reinterpret_cast<const void**>(&storage[used]);
        *slot = p;
        values[count++] = slot;
        used += 2;
    }
    void add_int(int32_t v) {
        storage[used] = v;
        values[count++] = &storage[used++];
    }
    void add_matrix(const void* p, int rows, int cols, int64_t ld) {
        add_ptr(p);
        add_int(rows), add_int(cols), add_int((int32_t)ld), add_int(1);
    }
};

inline cudaError_t launch(const std::string& symbol, unsigned grid, Args& args,
                          cudaStream_t stream) {
    return cudaLaunchKernel(reinterpret_cast<const void*>(kernel(symbol)),
                            dim3(grid), dim3(1), args.values, 0, stream);
}

// partials[g] (blocks x CLUSTER_PAD x pc_width) = sum over block g's row tiles
// of R^T Z for R (n_rows x n_clusters, bfloat16) and Z (n_rows x n_pcs).
// Returns the number of blocks used.
inline int rtz_partials(const void* R, const float* Z, int n_rows,
                        int n_clusters, int n_pcs, float* partials,
                        cudaStream_t stream, cudaError_t* status) {
    // Variant by size; a module embeds only the variant it mostly needs.
    bool small = n_rows < RTZ_SMALL_MAX_CELLS;
    if (!kernel(rtz_symbol(n_pcs, small))) small = !small;
    int rows = small ? RTZ_SMALL_ROWS : RTZ_LARGE_ROWS;
    int dp = pc_width(n_pcs);
    int n_tiles = (n_rows + rows - 1) / rows;
    int blocks = std::min(small ? RTZ_SMALL_BLOCKS : RTZ_LARGE_BLOCKS, n_tiles);
    int per = (n_tiles + blocks - 1) / blocks;
    Args args;
    args.add_matrix(R, n_rows, n_clusters, n_clusters);
    args.add_matrix(Z, n_rows, n_pcs, n_pcs);
    args.add_ptr(partials);
    args.add_int(blocks), args.add_int(CLUSTER_PAD), args.add_int(dp);
    args.add_int(CLUSTER_PAD * dp), args.add_int(dp), args.add_int(1);
    args.add_int(per), args.add_int(n_tiles);
    *status = launch(rtz_symbol(n_pcs, small), blocks, args, stream);
    return blocks;
}

// out[k * ld_out + d] = sum over blocks of partials[g][k][d], in block order.
__global__ void reduce_rtz_partials_kernel(const float* partials, int blocks,
                                           int dp, int n_clusters, int n_pcs,
                                           float* out, int64_t ld_out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_clusters * n_pcs;
         i += blockDim.x * gridDim.x) {
        int k = i / n_pcs, d = i % n_pcs;
        float sum = 0.f;
        for (int g = 0; g < blocks; ++g)
            sum += partials[((size_t)g * CLUSTER_PAD + k) * dp + d];
        out[(size_t)k * ld_out + d] = sum;
    }
}

inline size_t partials_size(int n_pcs) {
    return (size_t)RTZ_BLOCKS * CLUSTER_PAD * pc_width(n_pcs);
}

// out (n_clusters x n_pcs, row stride ld_out) = R^T Z, deterministic.
inline cudaError_t rtz(const void* R, const float* Z, int n_rows,
                       int n_clusters, int n_pcs, float* partials, float* out,
                       int64_t ld_out, cudaStream_t stream) {
    cudaError_t status;
    int blocks = rtz_partials(R, Z, n_rows, n_clusters, n_pcs, partials, stream,
                              &status);
    if (status != cudaSuccess) return status;
    reduce_rtz_partials_kernel<<<(n_clusters * n_pcs + 255) / 256, 256, 0,
                                 stream>>>(partials, blocks, pc_width(n_pcs),
                                           n_clusters, n_pcs, out, ld_out);
    return cudaGetLastError();
}

// Out (n_rows x n_pcs) = -(R @ W), W (n_clusters x n_pcs) with row stride ldw.
inline cudaError_t neg_rw(const void* R, const float* W, int64_t ldw,
                          float* out, int n_rows, int n_clusters, int n_pcs,
                          cudaStream_t stream) {
    Args args;
    args.add_matrix(R, n_rows, n_clusters, n_clusters);
    args.add_matrix(W, n_clusters, n_pcs, ldw);
    args.add_matrix(out, n_rows, n_pcs, n_pcs);
    return launch(neg_rw_symbol(n_pcs),
                  (n_rows + NEG_RW_ROWS - 1) / NEG_RW_ROWS, args, stream);
}

}  // namespace
}  // namespace harmony_cutile
