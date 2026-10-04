#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <nanobind/stl/optional.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../../cublas_helpers.cuh"
#include "../../nb_types.h"

#include "../normalize/kernels_normalize.cuh"
#include "../scatter/kernels_scatter.cuh"
#include "../scatter/kernels_scatter_reduce.cuh"
#include "../cutile/cutile_runtime.cuh"
#include "kernels_clustering.cuh"

using namespace nb::literals;

constexpr unsigned WARP_SIZE = 32;
constexpr unsigned MAX_BLOCK_DIM = 256;
constexpr int BLOCK_DIM_1D = 256;
constexpr int MAX_BLOCK_DIM_1D = 1024;
constexpr int BLOCKS_PER_SM = 8;

// ---------- Launch helpers ----------

// 1D grid size capped at BLOCKS_PER_SM blocks/SM
static inline int grid_1d(long long n, int n_sm) {
    long long blocks = (n + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D;
    return std::max(1, std::min(n_sm * BLOCKS_PER_SM,
                                (int)std::min<long long>(blocks, INT_MAX)));
}

// Block dim rounded up to nearest warp, capped at MAX_BLOCK_DIM
static inline unsigned warp_aligned_bdim(unsigned n) {
    return std::min(
        MAX_BLOCK_DIM,
        std::max(WARP_SIZE, (n + WARP_SIZE - 1u) / WARP_SIZE * WARP_SIZE));
}

// ---------- CUB temp-storage query ----------

static size_t get_cub_sort_temp_bytes(int n_cells) {
    size_t bytes = 0;
    auto* du = reinterpret_cast<unsigned int*>(1);
    auto* di = reinterpret_cast<int*>(1);
    cudaError_t status = cub::DeviceRadixSort::SortPairs(
        nullptr, bytes, du, du, di, di, n_cells, 0, 32);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cub sort temp query failed: ") +
                                 cudaGetErrorString(status));
    }
    return bytes;
}

template <typename T>
static inline void materialize_marginal_from_joint(
    const T* O_joint, const int* marginal_joint_offsets,
    const int* marginal_joint_indices, T* O, int n_batches, int n_clusters,
    int n_sm, cudaStream_t stream) {
    long long total = (long long)n_batches * n_clusters;
    materialize_marginal_from_joint_kernel<T>
        <<<grid_1d(total, n_sm), BLOCK_DIM_1D, 0, stream>>>(
            O_joint, marginal_joint_offsets, marginal_joint_indices, O,
            n_batches, n_clusters);
    CUDA_CHECK_LAST_ERROR(materialize_marginal_from_joint_kernel);
}

// ---------- Clustering arguments ----------

template <typename T>
struct ClusteringArgs {
    // Input/output; cells are sorted by group (batch, or joint category with
    // several keys).
    const T* Z_norm;
    T* R;
    __nv_bfloat16* R_bf16;  // bfloat16 assignments instead of R
    T* E;
    T* O;
    const T* Pr_b;
    const int* cats;
    const T* theta;
    const int* joint_codes;  // several keys: joint category per cell
    const int* joint_cats;   // several keys: J x n_covariates levels
    const int* marginal_joint_offsets;
    const int* marginal_joint_indices;
    T* O_joint;        // several keys: counts per joint category
    T* group_penalty;  // several keys: J x K log penalties

    // Workspace
    T* Y;
    T* Y_norm;
    int* idx_list;
    int* idx_list_alt;
    unsigned int* sort_keys;
    unsigned int* sort_keys_alt;
    uint8_t* cub_temp;
    T* penalty;
    T* obj_scalar;
    T* last_obj;
    uint8_t* scatter_workspace;
    T* objective_partials;
    int* block_cat_offsets;  // n_blocks x groups + 1, then one block's tiles
    T* holdout_counts;       // 2 x groups x K
    T* assign_partial;       // assignment tile column sums
    size_t assign_partial_size;
    float* tile_partials;  // bfloat16: cuTile R^T Z partials
    T* y_t_general;        // general assignment: transposed centroids
    T* col_workspace;      // general assignment: per-warp column sums
    bool force_general;    // tests: general assignment for every shape

    // Dimensions
    int n_cells;
    int n_pcs;
    int n_clusters;
    int n_batches;
    int n_covariates;
    int n_joint_categories;
    int n_first;  // levels of the first batch key
    int block_size;
    T sigma;
    T tol;
    int max_iter;
    unsigned int seed;
    bool stabilized;
    int shuffle_chunk;  // cells per shuffled run
    cudaStream_t stream;
    cublasHandle_t handle;
};

// ---------- Clustering loop ----------

// Side stream with double-buffered hold-out counts: `held` marks a buffer as
// filled, `used` as consumed by the main stream.
struct HoldoutStream {
    cudaStream_t stream = nullptr;
    cudaEvent_t ready = nullptr, held[2] = {}, used[2] = {};
    bool pending_use[2] = {false, false};
    HoldoutStream() {
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "hold-out stream");
        for (cudaEvent_t* e : {&ready, &held[0], &held[1], &used[0], &used[1]})
            cuda_check(cudaEventCreateWithFlags(e, cudaEventDisableTiming),
                       "hold-out event");
    }
    ~HoldoutStream() {
        cudaStreamSynchronize(stream);
        for (cudaEvent_t e : {ready, held[0], held[1], used[0], used[1]})
            cudaEventDestroy(e);
        cudaStreamDestroy(stream);
    }
    void record_ready(cudaStream_t main) {
        cudaEventRecord(ready, main);
        cudaStreamWaitEvent(stream, ready, 0);
    }
    void begin_holdout(int buffer) {
        if (pending_use[buffer]) cudaStreamWaitEvent(stream, used[buffer], 0);
        pending_use[buffer] = false;
    }
    void end_holdout(int buffer) {
        cudaEventRecord(held[buffer], stream);
    }
    void wait_holdout(int buffer, cudaStream_t main) {
        cudaStreamWaitEvent(main, held[buffer], 0);
    }
    void release_holdout(int buffer, cudaStream_t main) {
        cudaEventRecord(used[buffer], main);
        pending_use[buffer] = true;
    }
};

// Same iteration as the general loop, without the N x K similarity matrix,
// block gathers or column sums: each block is held out of O and added back by
// the deterministic tiled scatter reading R in place, and one warp per cell
// computes similarities, assignments and the objective terms.
template <typename T>
static void penalty_from_counts(const T* O, const T* Pr_b, const T* theta, T* E,
                                T* penalty, int n_batches, int n_clusters,
                                bool stabilized, cudaStream_t stream,
                                int n_first = 0) {
    if (n_first < 1) n_first = n_batches;
    if (stabilized)
        penalty_from_counts_kernel<T, true><<<n_clusters, 256, 0, stream>>>(
            O, Pr_b, theta, E, penalty, n_batches, n_clusters, n_first);
    else
        penalty_from_counts_kernel<T, false><<<n_clusters, 256, 0, stream>>>(
            O, Pr_b, theta, E, penalty, n_batches, n_clusters, n_first);
    CUDA_CHECK_LAST_ERROR(penalty_from_counts_kernel);
}

// One assignment pass over the cells `idx[offsets[0] .. offsets[n_batches])`
// (identity when idx is null) grouped by category: writes R and the per-cell
// objective terms and adds the new counts to O. Tiles are sized so at most
// `capacity` (+ one partial tile per category) fit `assign_partial`.
template <typename T, typename RT = T>
static void fused_assign_pass(
    const T* Z_norm, const T* Y_norm, const T* penalty, const int* idx,
    const int* offsets, int n_rows, int n_batches, int* tiles,
    T* assign_partial, long long capacity, RT* R, T* objective_partials, T* O,
    T term, T sigma, int n_pcs, int n_clusters, cudaStream_t stream,
    bool log_pen = false, T* y_t_general = nullptr, T* col_workspace = nullptr,
    bool force_general = false) {
    size_t smem = fused_assign_smem_bytes<T>(n_pcs, n_clusters);
    // Shapes beyond the fused kernel (more than 128 clusters, or centroids
    // beyond shared memory) take the general kernel.
    bool general = force_general || !fused_assign_fits(n_pcs, n_clusters, smem);
    if (general && (!y_t_general || !col_workspace))
        throw std::invalid_argument(
            "this shape needs the general assignment workspace");
    int z_slots = (fused_padded_pcs(n_pcs) + 31) / 32;
    auto pick = [&](auto log_tag) {
        constexpr bool L = decltype(log_tag)::value;
        return z_slots == 1   ? fused_assign_kernel<T, RT, 1, L>
               : z_slots == 2 ? fused_assign_kernel<T, RT, 2, L>
               : z_slots == 3 ? fused_assign_kernel<T, RT, 3, L>
                              : fused_assign_kernel<T, RT, 4, L>;
    };
    auto kernel = log_pen ? pick(std::true_type{}) : pick(std::false_type{});
    if (!general && smem > 48 * 1024)
        cuda_check(
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem),
            "fused assign shared memory");
    long long rows_per_warp = FUSED_ROWS * (FUSED_THREADS / 32);
    long long tile_rows = (n_rows + capacity - 1) / capacity;
    tile_rows = std::max(rows_per_warp, (tile_rows + rows_per_warp - 1) /
                                            rows_per_warp * rows_per_warp);
    scatter_tile_offsets_kernel<<<1, SCATTER_SCAN_THREADS, 0, stream>>>(
        offsets, n_batches, tiles, (int)tile_rows);
    CUDA_CHECK_LAST_ERROR(scatter_tile_offsets_kernel);
    size_t max_tiles = std::min<size_t>(
        n_rows, (n_rows + tile_rows - 1) / tile_rows + n_batches);
    if (general) {
        int pcs = fused_padded_pcs(n_pcs);
        int k_stride = fused_cluster_stride(n_clusters);
        transpose_centroids_kernel<T>
            <<<(pcs * k_stride + 255) / 256, 256, 0, stream>>>(
                Y_norm, y_t_general, n_pcs, pcs, n_clusters, k_stride);
        CUDA_CHECK_LAST_ERROR(transpose_centroids_kernel);
        auto general_kernel = log_pen
                                  ? fused_assign_general_kernel<T, RT, true>
                                  : fused_assign_general_kernel<T, RT, false>;
        general_kernel<<<max_tiles, FUSED_THREADS, 0, stream>>>(
            Z_norm, y_t_general, k_stride, penalty, idx, offsets, tiles,
            n_batches, (int)tile_rows, R, objective_partials, assign_partial,
            col_workspace, term, sigma, n_pcs, n_clusters);
    } else {
        kernel<<<max_tiles, FUSED_THREADS, smem, stream>>>(
            Z_norm, Y_norm, penalty, idx, offsets, tiles, n_batches,
            (int)tile_rows, R, objective_partials, assign_partial, term, sigma,
            n_pcs, n_clusters);
    }
    CUDA_CHECK_LAST_ERROR(fused_assign_kernel);
    scatter_finish_kernel<T>
        <<<std::min((size_t)n_batches, max_tiles) * ((n_clusters + 31) / 32),
           dim3(32, 8), 0, stream>>>(assign_partial, n_clusters, n_batches,
                                     tiles, 1, O);
    CUDA_CHECK_LAST_ERROR(scatter_finish_kernel);
}

// Objective from the per-cell terms of the last pass plus the diversity term.
template <typename T>
static T fused_objective(const T* objective_partials, long long n_cells,
                         T* stage, const T* O, const T* E, const T* theta,
                         T sigma, int n_batches, int n_clusters,
                         bool stabilized, T* obj_scalar, cudaStream_t stream) {
    objective_stage_kernel<T><<<OBJECTIVE_REDUCE_BLOCKS, 256, 0, stream>>>(
        objective_partials, n_cells, stage);
    CUDA_CHECK_LAST_ERROR(objective_stage_kernel);
    objective_diversity_kernel<T>
        <<<1, 256, 0, stream>>>(O, E, theta, sigma, n_batches, n_clusters,
                                stabilized, stage + OBJECTIVE_REDUCE_BLOCKS);
    CUDA_CHECK_LAST_ERROR(objective_diversity_kernel);
    objective_reduce_kernel<T>
        <<<1, 256, 0, stream>>>(stage, OBJECTIVE_REDUCE_BLOCKS + 1, obj_scalar);
    CUDA_CHECK_LAST_ERROR(objective_reduce_kernel);
    T host_obj;
    cuda_check(cudaMemcpyAsync(&host_obj, obj_scalar, sizeof(T),
                               cudaMemcpyDeviceToHost, stream),
               "objective copy");
    cuda_check(cudaStreamSynchronize(stream), "objective synchronization");
    return host_obj;
}

template <typename T, typename RT>
static void fused_clustering_loop_body(const ClusteringArgs<T>& a, RT* R) {
    if (!a.block_cat_offsets)
        throw std::invalid_argument(
            "fused clustering requires block_cat_offsets");

    size_t cub_temp_bytes = get_cub_sort_temp_bytes(a.n_cells);
    cublasHandle_t handle = a.handle;
    cublas_check_status(cublasSetStream(handle, a.stream), "cublasSetStream");
    T term = T(-2) / a.sigma;
    T one = T(1), zero = T(0);
    int device;
    cudaGetDevice(&device);
    int n_sm;
    cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, device);
    // Cells are grouped by batch (one key) or joint category (several keys).
    // Several keys keep counts per joint category and rebuild the marginal O.
    bool multi = a.n_covariates > 1;
    if (multi && (!a.joint_codes || !a.joint_cats || !a.O_joint ||
                  !a.marginal_joint_offsets || !a.marginal_joint_indices ||
                  !a.group_penalty))
        throw std::invalid_argument(
            "fused clustering with several keys needs the joint arrays");
    int n_groups = multi ? a.n_joint_categories : a.n_batches;
    const int* group_codes = multi ? a.joint_codes : a.cats;
    T* O_group = multi ? a.O_joint : a.O;
    const T* group_penalty = multi ? a.group_penalty : a.penalty;
    int n_blocks = (a.n_cells + a.block_size - 1) / a.block_size;
    int n_keys = n_blocks * n_groups;
    int key_bits = scatter_category_bits(n_keys);
    int* tiles = reinterpret_cast<int*>(a.scatter_workspace);
    T* partial = reinterpret_cast<T*>(a.scatter_workspace +
                                      scatter_grouped_int_bytes(n_groups));
    // Assignment tiles are sized to the block so small blocks still fill the
    // GPU; their tile offsets follow the block's category offsets.
    int* assign_tiles = a.block_cat_offsets + (size_t)n_keys + 1;
    long long assign_capacity =
        (long long)(a.assign_partial_size / a.n_clusters) - n_groups - 1;
    if (!a.assign_partial || assign_capacity < 1)
        throw std::invalid_argument(
            "fused clustering assign_partial is too small");

    if (!a.holdout_counts)
        throw std::invalid_argument("fused clustering requires holdout_counts");
    int ob_total = n_groups * a.n_clusters;
    auto marginal_counts = [&]() {
        if (multi)
            materialize_marginal_from_joint<T>(
                a.O_joint, a.marginal_joint_offsets, a.marginal_joint_indices,
                a.O, a.n_batches, a.n_clusters, n_sm, a.stream);
    };
    HoldoutStream side;
    std::vector<T> objectives;
    constexpr int WINDOW_SIZE = 3;
    for (int iter = 0; iter < a.max_iter; iter++) {
        if constexpr (std::is_same_v<RT, __nv_bfloat16>)
            cuda_check(harmony_cutile::rtz(R, a.Z_norm, a.n_cells, a.n_clusters,
                                           a.n_pcs, a.tile_partials, a.Y,
                                           a.n_pcs, a.stream),
                       "cuTile centroids");
        else
            cublas_check_status(
                cublas_gemm<T>(handle, CUBLAS_OP_N, CUBLAS_OP_T, a.n_pcs,
                               a.n_clusters, a.n_cells, &one, a.Z_norm, a.n_pcs,
                               R, a.n_clusters, &zero, a.Y, a.n_pcs),
                "cublas_gemm(centroids)");
        l2_row_normalize_kernel<T>
            <<<a.n_clusters, warp_aligned_bdim(a.n_pcs), 0, a.stream>>>(
                a.Y, a.Y_norm, a.n_clusters, a.n_pcs);
        CUDA_CHECK_LAST_ERROR(l2_row_normalize_kernel);

        // Shuffle, then order each block's cells by category (stable).
        pcg_hash_kernel<<<grid_1d(a.n_cells, n_sm), BLOCK_DIM_1D, 0,
                          a.stream>>>(a.sort_keys, a.idx_list, a.n_cells,
                                      a.seed + iter, a.shuffle_chunk);
        CUDA_CHECK_LAST_ERROR(pcg_hash_kernel);
        cuda_check(cub::DeviceRadixSort::SortPairs(
                       a.cub_temp, cub_temp_bytes, a.sort_keys, a.sort_keys_alt,
                       a.idx_list, a.idx_list_alt, a.n_cells, 0, 32, a.stream),
                   "cub radix sort");
        block_category_keys_kernel<<<grid_1d(a.n_cells, n_sm), BLOCK_DIM_1D, 0,
                                     a.stream>>>(a.idx_list_alt, group_codes,
                                                 a.sort_keys, a.n_cells,
                                                 a.block_size, n_groups);
        CUDA_CHECK_LAST_ERROR(block_category_keys_kernel);
        cuda_check(
            cub::DeviceRadixSort::SortPairs(
                a.cub_temp, cub_temp_bytes, a.sort_keys, a.sort_keys_alt,
                a.idx_list_alt, a.idx_list, a.n_cells, 0, key_bits, a.stream),
            "cub block-category sort");
        scatter_category_offsets_kernel<<<(n_keys + 127) / 128, 128, 0,
                                          a.stream>>>(
            reinterpret_cast<const int*>(a.sort_keys_alt), a.n_cells, n_keys,
            a.block_cat_offsets);
        CUDA_CHECK_LAST_ERROR(scatter_category_offsets_kernel);

        // Hold-outs read only their own block's rows, so they run ahead on
        // the side stream while earlier blocks are assigned; the main stream
        // applies them to O in block order.
        side.record_ready(a.stream);
        auto holdout = [&](int blk) {
            T* H = a.holdout_counts + (size_t)(blk & 1) * ob_total;
            side.begin_holdout(blk & 1);
            cudaMemsetAsync(H, 0, ob_total * sizeof(T), side.stream);
            const int* offsets = a.block_cat_offsets + (size_t)blk * n_groups;
            scatter_tile_offsets_kernel<<<1, SCATTER_SCAN_THREADS, 0,
                                          side.stream>>>(offsets, n_groups,
                                                         tiles);
            CUDA_CHECK_LAST_ERROR(scatter_tile_offsets_kernel);
            scatter_reduce_tiles<T>(
                R, std::min(a.block_size, a.n_cells - blk * a.block_size),
                a.n_clusters, n_groups, a.idx_list, offsets, tiles, partial, 1,
                H, side.stream);
            side.end_holdout(blk & 1);
        };
        for (int blk = 0; blk < std::min(2, n_blocks); ++blk) holdout(blk);
        for (int blk = 0; blk < n_blocks; ++blk) {
            int pos = blk * a.block_size;
            int bs = std::min(a.block_size, a.n_cells - pos);
            const int* offsets = a.block_cat_offsets + (size_t)blk * n_groups;
            side.wait_holdout(blk & 1, a.stream);
            subtract_kernel<T><<<(ob_total + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D,
                                 BLOCK_DIM_1D, 0, a.stream>>>(
                O_group, a.holdout_counts + (size_t)(blk & 1) * ob_total,
                ob_total);
            CUDA_CHECK_LAST_ERROR(subtract_kernel);
            side.release_holdout(blk & 1, a.stream);
            if (blk + 2 < n_blocks) holdout(blk + 2);
            marginal_counts();
            penalty_from_counts(a.O, a.Pr_b, a.theta, a.E, a.penalty,
                                a.n_batches, a.n_clusters, a.stabilized,
                                a.stream, a.n_first);
            if (multi) {
                joint_log_penalty_kernel<T>
                    <<<grid_1d((long long)ob_total, n_sm), BLOCK_DIM_1D, 0,
                       a.stream>>>(a.penalty, a.joint_cats, a.n_covariates,
                                   n_groups, a.n_clusters, a.group_penalty);
                CUDA_CHECK_LAST_ERROR(joint_log_penalty_kernel);
            }
            fused_assign_pass(a.Z_norm, a.Y_norm, group_penalty, a.idx_list,
                              offsets, bs, n_groups, assign_tiles,
                              a.assign_partial, assign_capacity, R,
                              a.objective_partials, O_group, term, a.sigma,
                              a.n_pcs, a.n_clusters, a.stream, multi,
                              a.y_t_general, a.col_workspace, a.force_general);
        }
        marginal_counts();
        penalty_from_counts(a.O, a.Pr_b, a.theta, a.E, (T*)nullptr, a.n_batches,
                            a.n_clusters, a.stabilized, a.stream, a.n_first);

        // Objective: per-cell terms from the assignment pass plus diversity.
        // The first convergence decision needs WINDOW_SIZE + 1 objectives;
        // when the budget ends there anyway, only the last one is used.
        if (a.max_iter > WINDOW_SIZE + 1 || iter + 1 == a.max_iter) {
            objectives.push_back(
                fused_objective(a.objective_partials, a.n_cells,
                                a.objective_partials + a.n_cells, a.O, a.E,
                                a.theta, a.sigma, a.n_batches, a.n_clusters,
                                a.stabilized, a.obj_scalar, a.stream));

            if (static_cast<int>(objectives.size()) >= WINDOW_SIZE + 1) {
                T obj_old = T(0), obj_new = T(0);
                for (int i = 0; i < WINDOW_SIZE; i++) {
                    obj_old += objectives[objectives.size() - 2 - i];
                    obj_new += objectives[objectives.size() - 1 - i];
                }
                if ((obj_old - obj_new) < a.tol * std::abs(obj_old)) break;
            }
        }
    }
    T final_obj = objectives.empty() ? T(0) : objectives.back();
    cudaMemcpyAsync(a.last_obj, &final_obj, sizeof(T), cudaMemcpyHostToDevice,
                    a.stream);
    cudaStreamSynchronize(a.stream);
}

template <typename T>
static void fused_clustering_loop_impl(const ClusteringArgs<T>& a) {
    if (!a.R_bf16) return fused_clustering_loop_body(a, a.R);
    if constexpr (std::is_same_v<T, float>) {
        if (!a.tile_partials)
            throw std::invalid_argument(
                "bfloat16 assignments need tile_partials");
        return fused_clustering_loop_body(a, a.R_bf16);
    }
    throw std::invalid_argument("bfloat16 assignments require float32");
}

// ---------- Nanobind bindings ----------

template <typename T, typename Device>
static void register_clustering_loop(nb::module_& m) {
    using Ints = std::optional<gpu_array_c<const int, Device>>;
    using Opt = std::optional<gpu_array_c<T, Device>>;
    m.def(
        "clustering_loop",
        [](gpu_array_c<const T, Device> Z_norm, gpu_array_c<T, Device> R,
           gpu_array_c<T, Device> E, gpu_array_c<T, Device> O,
           gpu_array_c<const T, Device> Pr_b,
           gpu_array_c<const int, Device> cats,
           gpu_array_c<const T, Device> theta, gpu_array_c<T, Device> Y,
           gpu_array_c<T, Device> Y_norm, gpu_array_c<int, Device> idx_list,
           gpu_array_c<int, Device> idx_list_alt,
           gpu_array_c<unsigned int, Device> sort_keys,
           gpu_array_c<unsigned int, Device> sort_keys_alt,
           gpu_array_c<uint8_t, Device> cub_temp,
           gpu_array_c<T, Device> penalty, gpu_array_c<T, Device> obj_scalar,
           gpu_array_c<T, Device> last_obj,
           gpu_array_c<uint8_t, Device> scatter_workspace,
           gpu_array_c<T, Device> objective_partials,
           gpu_array_c<int, Device> block_cat_offsets,
           gpu_array_c<T, Device> holdout_counts,
           gpu_array_c<T, Device> assign_partial,
           std::optional<gpu_array_c<uint16_t, Device>> R_bf16,
           Opt tile_partials, Ints joint_codes, Ints joint_cats,
           Ints marginal_joint_offsets, Ints marginal_joint_indices,
           Opt O_joint, Opt group_penalty, Opt y_t_general, Opt col_workspace,
           bool force_general, int n_cells, int n_pcs, int n_clusters,
           int n_batches, int n_covariates, int n_joint_categories, int n_first,
           int block_size, double sigma, double tol, int max_iter,
           unsigned int seed, bool stabilized, int shuffle_chunk,
           std::uintptr_t stream, std::uintptr_t handle) {
            auto ptr = [](auto& a) { return a ? a->data() : nullptr; };
            ClusteringArgs<T> a{
                Z_norm.data(),
                R.data(),
                R_bf16 ? reinterpret_cast<__nv_bfloat16*>(R_bf16->data())
                       : nullptr,
                E.data(),
                O.data(),
                Pr_b.data(),
                cats.data(),
                theta.data(),
                ptr(joint_codes),
                ptr(joint_cats),
                ptr(marginal_joint_offsets),
                ptr(marginal_joint_indices),
                ptr(O_joint),
                ptr(group_penalty),
                Y.data(),
                Y_norm.data(),
                idx_list.data(),
                idx_list_alt.data(),
                sort_keys.data(),
                sort_keys_alt.data(),
                cub_temp.data(),
                penalty.data(),
                obj_scalar.data(),
                last_obj.data(),
                scatter_workspace.data(),
                objective_partials.data(),
                block_cat_offsets.data(),
                holdout_counts.data(),
                assign_partial.data(),
                assign_partial.size(),
                tile_partials ? reinterpret_cast<float*>(tile_partials->data())
                              : nullptr,
                ptr(y_t_general),
                ptr(col_workspace),
                force_general,
                n_cells,
                n_pcs,
                n_clusters,
                n_batches,
                n_covariates,
                n_joint_categories,
                n_first,
                block_size,
                static_cast<T>(sigma),
                static_cast<T>(tol),
                max_iter,
                seed,
                stabilized,
                std::max(1, shuffle_chunk),
                (cudaStream_t)stream,
                (cublasHandle_t)handle,
            };
            fused_clustering_loop_impl(a);
        },
        "Z_norm"_a, nb::kw_only(), "R"_a, "E"_a, "O"_a, "Pr_b"_a, "cats"_a,
        "theta"_a, "Y"_a, "Y_norm"_a, "idx_list"_a, "idx_list_alt"_a,
        "sort_keys"_a, "sort_keys_alt"_a, "cub_temp"_a, "penalty"_a,
        "obj_scalar"_a, "last_obj"_a, "scatter_workspace"_a,
        "objective_partials"_a, "block_cat_offsets"_a, "holdout_counts"_a,
        "assign_partial"_a, "R_bf16"_a = nb::none(),
        "tile_partials"_a = nb::none(), "joint_codes"_a = nb::none(),
        "joint_cats"_a = nb::none(), "marginal_joint_offsets"_a = nb::none(),
        "marginal_joint_indices"_a = nb::none(), "O_joint"_a = nb::none(),
        "group_penalty"_a = nb::none(), "y_t_general"_a = nb::none(),
        "col_workspace"_a = nb::none(), "force_general"_a = false, "n_cells"_a,
        "n_pcs"_a, "n_clusters"_a, "n_batches"_a, "n_covariates"_a = 1,
        "n_joint_categories"_a = 0, "n_first"_a = 0, "block_size"_a, "sigma"_a,
        "tol"_a, "max_iter"_a, "seed"_a, "stabilized"_a, "shuffle_chunk"_a = 1,
        "stream"_a = 0, "handle"_a);
}

template <typename T, typename Device>
static void register_kmeans_selection(nb::module_& m) {
    m.def(
        "select_kmeans_center",
        [](gpu_array_c<const T, Device> X, gpu_array_c<const T, Device> weights,
           gpu_array_c<const double, Device> uniforms,
           gpu_array_c<T, Device> centers, gpu_array_c<double, Device> totals,
           gpu_array_c<int, Device> n_draws, int cluster,
           std::uintptr_t stream) {
            int n_rows = (int)X.shape(0), n_cols = (int)X.shape(1);
            int n_tiles = n_rows / KMEANS_WEIGHT_TILE_ROWS +
                          (n_rows % KMEANS_WEIGHT_TILE_ROWS != 0);
            if (totals.size() < (size_t)n_tiles)
                throw std::invalid_argument(
                    "k-means totals workspace is too small");
            auto cuda_stream = (cudaStream_t)stream;
            kmeans_weight_tiles_kernel<T>
                <<<n_tiles, KMEANS_WEIGHT_THREADS, 0, cuda_stream>>>(
                    weights.data(), totals.data(), n_rows);
            CUDA_CHECK_LAST_ERROR(kmeans_weight_tiles_kernel);
            kmeans_select_center_kernel<T>
                <<<1, KMEANS_WEIGHT_THREADS, 0, cuda_stream>>>(
                    X.data(), weights.data(), totals.data(), uniforms.data(),
                    centers.data(), n_draws.data(), n_rows, n_cols, n_tiles,
                    cluster);
            CUDA_CHECK_LAST_ERROR(kmeans_select_center_kernel);
        },
        "X"_a, nb::kw_only(), "weights"_a, "uniforms"_a, "centers"_a,
        "totals"_a, "n_draws"_a, "cluster"_a, "stream"_a = 0);
}

template <typename T, typename Device>
static void register_scatter(nb::module_& m) {
    m.def(
        "scatter_add",
        [](gpu_array_c<const T, Device> values,
           gpu_array_c<const int, Device> categories,
           gpu_array_c<T, Device> out, gpu_array_c<uint8_t, Device> workspace,
           int n_rows, int n_cols, int n_categories, int n_covariates,
           int switcher, std::uintptr_t stream,
           std::optional<gpu_array_c<const int, Device>> category_offsets,
           std::optional<gpu_array_c<const int, Device>> cell_indices) {
            if (category_offsets.has_value() != cell_indices.has_value())
                throw std::invalid_argument(
                    "grouped scatter requires both category_offsets and "
                    "cell_indices");
            size_t required_bytes =
                scatter_temp_bytes(n_rows, n_cols, n_categories, n_covariates,
                                   sizeof(T), category_offsets.has_value());
            if (workspace.size() < required_bytes)
                throw std::invalid_argument("scatter workspace is too small");
            if (category_offsets) {
                if (category_offsets->ndim() != 1 ||
                    category_offsets->size() != (size_t)n_categories + 1 ||
                    cell_indices->ndim() != 1 ||
                    cell_indices->size() != (size_t)n_rows)
                    throw std::invalid_argument(
                        "grouped scatter CSR arrays have incorrect shapes");
                int* tiles = reinterpret_cast<int*>(workspace.data());
                T* partial = reinterpret_cast<T*>(
                    workspace.data() + scatter_grouped_int_bytes(n_categories));
                scatter_tile_offsets_kernel<<<1, SCATTER_SCAN_THREADS, 0,
                                              (cudaStream_t)stream>>>(
                    category_offsets->data(), n_categories, tiles);
                CUDA_CHECK_LAST_ERROR(scatter_tile_offsets_kernel);
                scatter_reduce_tiles(
                    values.data(), n_rows, n_cols, n_categories,
                    cell_indices->data(), category_offsets->data(), tiles,
                    partial, switcher, out.data(), (cudaStream_t)stream);
                return;
            }
            scatter_reduce(values.data(), categories.data(), n_rows, n_cols,
                           n_categories, n_covariates, switcher, out.data(),
                           workspace.data(), (cudaStream_t)stream);
        },
        "values"_a, nb::kw_only(), "categories"_a, "out"_a, "workspace"_a,
        "n_rows"_a, "n_cols"_a, "n_categories"_a, "n_covariates"_a = 1,
        "switcher"_a, "stream"_a = 0, "category_offsets"_a = nb::none(),
        "cell_indices"_a = nb::none());
}

template <typename T, typename Device>
static void register_kmeans_fused(nb::module_& m) {
    m.def(
        "kmeans_assign",
        [](gpu_array_c<const T, Device> X, gpu_array_c<const T, Device> centers,
           gpu_array_c<int, Device> labels, gpu_array_c<T, Device> minimum,
           std::uintptr_t stream) {
            int n_rows = (int)X.shape(0), n_cols = (int)X.shape(1);
            int n_clusters = (int)centers.shape(0);
            if (n_clusters > 32 * FUSED_MAX_CLUSTER_SLOTS)
                throw std::invalid_argument(
                    "kmeans_assign supports at most 128 clusters");
            size_t smem =
                ((size_t)fused_padded_pcs(n_cols) *
                 (FUSED_CLUSTER_STRIDE + (FUSED_THREADS / 32) * FUSED_ROWS)) *
                sizeof(T);
            if (smem > 48 * 1024)
                cuda_check(
                    cudaFuncSetAttribute(
                        kmeans_assign_kernel<T>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem),
                    "kmeans assign shared memory");
            int device, n_sm;
            cudaGetDevice(&device);
            cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount,
                                   device);
            long long groups =
                ((long long)n_rows + FUSED_ROWS - 1) / FUSED_ROWS;
            int grid = (int)std::max<long long>(
                1, std::min<long long>(n_sm * 4,
                                       (groups + FUSED_THREADS / 32 - 1) /
                                           (FUSED_THREADS / 32)));
            kmeans_assign_kernel<T>
                <<<grid, FUSED_THREADS, smem, (cudaStream_t)stream>>>(
                    X.data(), centers.data(), labels.data(), minimum.data(),
                    n_rows, n_cols, n_clusters);
            CUDA_CHECK_LAST_ERROR(kmeans_assign_kernel);
        },
        "X"_a, nb::kw_only(), "centers"_a, "labels"_a, "minimum"_a,
        "stream"_a = 0);
    m.def(
        "kmeans_closest",
        [](gpu_array_c<const T, Device> X, gpu_array_c<const T, Device> center,
           gpu_array_c<T, Device> closest, std::uintptr_t stream) {
            int n_rows = (int)X.shape(0), n_cols = (int)X.shape(1);
            kmeans_closest_kernel<T>
                <<<std::max(1, std::min(65535, (n_rows + 7) / 8)), 256, 0,
                   (cudaStream_t)stream>>>(X.data(), center.data(),
                                           closest.data(), n_rows, n_cols);
            CUDA_CHECK_LAST_ERROR(kmeans_closest_kernel);
        },
        "X"_a, nb::kw_only(), "center"_a, "closest"_a, "stream"_a = 0);
}

template <typename T, typename Device>
static void register_fused_initialize(nb::module_& m) {
    m.def(
        "fused_initialize",
        [](gpu_array_c<const T, Device> Z_norm,
           gpu_array_c<const T, Device> Y_norm,
           gpu_array_c<const int, Device> cat_offsets, gpu_array_c<T, Device> R,
           gpu_array_c<T, Device> O, gpu_array_c<T, Device> E,
           gpu_array_c<const T, Device> Pr_b,
           gpu_array_c<const T, Device> theta, gpu_array_c<int, Device> tiles,
           gpu_array_c<T, Device> assign_partial,
           gpu_array_c<T, Device> objective_partials,
           gpu_array_c<T, Device> obj_scalar, double sigma, bool stabilized,
           std::optional<gpu_array_c<uint16_t, Device>> R_bf16,
           std::optional<gpu_array_c<T, Device>> O_joint,
           std::optional<gpu_array_c<const int, Device>> marginal_joint_offsets,
           std::optional<gpu_array_c<const int, Device>> marginal_joint_indices,
           int n_first, std::optional<gpu_array_c<T, Device>> y_t_general,
           std::optional<gpu_array_c<T, Device>> col_workspace,
           bool force_general, std::uintptr_t stream) {
            // Unpenalized soft assignment of every cell (cells sorted by
            // category, so the batch offsets are the tile groups), O, E, and
            // the initial objective. O must be zero on entry.
            int n_cells = (int)Z_norm.shape(0), n_pcs = (int)Z_norm.shape(1);
            int n_clusters = (int)Y_norm.shape(0);
            // Groups are batches, or joint categories with O_joint (several
            // keys).
            int n_groups = (int)cat_offsets.size() - 1;
            int n_batches = (int)O.shape(0);
            if (O_joint && (!marginal_joint_offsets || !marginal_joint_indices))
                throw std::invalid_argument(
                    "fused_initialize with O_joint needs the marginal maps");

            long long capacity =
                (long long)(assign_partial.size() / n_clusters) - n_groups - 1;
            if (capacity < 1 ||
                objective_partials.size() <
                    (size_t)n_cells + OBJECTIVE_REDUCE_BLOCKS + 1)
                throw std::invalid_argument(
                    "fused_initialize workspace too small");
            auto s = (cudaStream_t)stream;
            T term = T(-2) / T(sigma);
            auto assign = [&](auto* R_out) {
                fused_assign_pass<T>(
                    Z_norm.data(), Y_norm.data(), nullptr, nullptr,
                    cat_offsets.data(), n_cells, n_groups, tiles.data(),
                    assign_partial.data(), capacity, R_out,
                    objective_partials.data(),
                    O_joint ? O_joint->data() : O.data(), term, T(sigma), n_pcs,
                    n_clusters, s, false,
                    y_t_general ? y_t_general->data() : nullptr,
                    col_workspace ? col_workspace->data() : nullptr,
                    force_general);
            };
            if (R_bf16)
                assign(reinterpret_cast<__nv_bfloat16*>(R_bf16->data()));
            else
                assign(R.data());
            if (O_joint) {
                int device, n_sm;
                cudaGetDevice(&device);
                cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount,
                                       device);
                materialize_marginal_from_joint<T>(
                    O_joint->data(), marginal_joint_offsets->data(),
                    marginal_joint_indices->data(), O.data(), n_batches,
                    n_clusters, n_sm, s);
            }
            penalty_from_counts<T>(O.data(), Pr_b.data(), theta.data(),
                                   E.data(), nullptr, n_batches, n_clusters,
                                   stabilized, s, n_first);
            return (double)fused_objective<T>(
                objective_partials.data(), n_cells,
                objective_partials.data() + n_cells, O.data(), E.data(),
                theta.data(), T(sigma), n_batches, n_clusters, stabilized,
                obj_scalar.data(), s);
        },
        "Z_norm"_a, nb::kw_only(), "Y_norm"_a, "cat_offsets"_a, "R"_a, "O"_a,
        "E"_a, "Pr_b"_a, "theta"_a, "tiles"_a, "assign_partial"_a,
        "objective_partials"_a, "obj_scalar"_a, "sigma"_a, "stabilized"_a,
        "R_bf16"_a = nb::none(), "O_joint"_a = nb::none(),
        "marginal_joint_offsets"_a = nb::none(),
        "marginal_joint_indices"_a = nb::none(), "n_first"_a = 0,
        "y_t_general"_a = nb::none(), "col_workspace"_a = nb::none(),
        "force_general"_a = false, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def(
        "cutile_bf16_available",
        [](int n_pcs, int n_clusters) {
            return harmony_cutile::available(n_pcs, n_clusters);
        },
        "n_pcs"_a, "n_clusters"_a,
        "Whether bfloat16 assignments can run (CUDA 13 cuTile build, "
        "supported GPU) for this embedding width and cluster count.");
    m.def(
        "cutile_partials_size",
        [](int n_pcs) { return harmony_cutile::partials_size(n_pcs); },
        "n_pcs"_a);
    m.def(
        "get_cub_sort_temp_bytes",
        [](int n_cells) { return get_cub_sort_temp_bytes(n_cells); },
        nb::kw_only(), "n_cells"_a);
    m.def(
        "get_scatter_temp_bytes",
        [](int n_rows, int n_cols, int n_categories, int n_covariates,
           int itemsize, bool grouped) {
            return scatter_temp_bytes(n_rows, n_cols, n_categories,
                                      n_covariates, itemsize, grouped);
        },
        nb::kw_only(), "n_rows"_a, "n_cols"_a, "n_categories"_a,
        "n_covariates"_a = 1, "itemsize"_a = 4, "grouped"_a = false);

    register_clustering_loop<float, Device>(m);
    register_clustering_loop<double, Device>(m);
    register_scatter<float, Device>(m);
    register_scatter<double, Device>(m);
    register_kmeans_selection<float, Device>(m);
    register_kmeans_selection<double, Device>(m);
    register_kmeans_fused<float, Device>(m);
    register_kmeans_fused<double, Device>(m);
    register_fused_initialize<float, Device>(m);
    register_fused_initialize<double, Device>(m);
}

NB_MODULE(_harmony_clustering_cuda, m) {
    m.attr("KMEANS_WEIGHT_TILE_ROWS") = nb::int_(KMEANS_WEIGHT_TILE_ROWS);
    m.attr("OBJECTIVE_REDUCE_BLOCKS") = nb::int_(OBJECTIVE_REDUCE_BLOCKS);
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
