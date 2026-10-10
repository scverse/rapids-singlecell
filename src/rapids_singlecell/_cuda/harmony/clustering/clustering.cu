#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <nanobind/stl/optional.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../../nb_types.h"

#include "../scatter/kernels_scatter.cuh"
#include "../scatter/kernels_scatter_reduce.cuh"
#include "../segment_gemm.cuh"
#include "kernels_clustering.cuh"

using namespace nb::literals;

constexpr int BLOCK_DIM_1D = 256;

static int device_attribute(cudaDeviceAttr attribute) {
    int device, value;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&value, attribute, device);
    return value;
}

template <typename T>
static void l2_row_normalize(const T* src, T* dst, int n_rows, int n_cols,
                             cudaStream_t stream) {
    int threads = std::clamp((n_cols + 31) / 32 * 32, 32, 256);
    l2_row_normalize_kernel<T>
        <<<n_rows, threads, 0, stream>>>(src, dst, n_cols);
    CUDA_CHECK_LAST_ERROR(l2_row_normalize_kernel);
}

// ---------- Update blocks ----------

// The update blocks of a harmonize call. The cells (sorted by group) are cut
// into runs of `chunk` cells and units of `unit` cells; within each unit the
// runs are ranked by a hash of their global index and dealt to the blocks in
// turn. Every block takes the same share of every unit, and a cell's block
// depends only on its global position (`first` + local index; `first` a
// multiple of `unit`), not on how the cells are split across GPUs. idx_list
// holds the cells by block, then group, in position order; block_cat_offsets
// the CSR over keys block * n_groups + group.
static void draw_blocks(const int* group_codes, int n_cells, long long first,
                        int n_groups, int n_blocks, int unit, unsigned int seed,
                        int chunk, int* idx_list, int* block_cat_offsets,
                        int* idx_list_alt, unsigned int* keys,
                        unsigned int* keys_alt, uint8_t* cub_temp,
                        size_t cub_bytes, cudaStream_t stream) {
    if (unit % chunk || first % unit)
        throw std::invalid_argument(
            "draw_blocks: unit must divide the shard "
            "start and be a multiple of chunk");
    int n_runs = (n_cells + chunk - 1) / chunk, unit_runs = unit / chunk;
    int n_units = (n_runs + unit_runs - 1) / unit_runs;
    int n_keys = n_blocks * n_groups;
    auto sort = [&](unsigned int* k_in, unsigned int* k_out, int* v_in,
                    int* v_out, int n, int bits) {
        cuda_check(
            cub::DeviceRadixSort::SortPairs(cub_temp, cub_bytes, k_in, k_out,
                                            v_in, v_out, n, 0, bits, stream),
            "cub radix sort");
    };
    int run_grid = strided_grid(n_runs, BLOCK_DIM_1D);
    // Runs by (unit, hash): two stable sorts.
    pcg_hash_kernel<<<run_grid, BLOCK_DIM_1D, 0, stream>>>(
        keys, idx_list, n_runs, (unsigned int)(first / chunk), seed);
    CUDA_CHECK_LAST_ERROR(pcg_hash_kernel);
    sort(keys, keys_alt, idx_list, idx_list_alt, n_runs, 32);
    run_unit_kernel<<<run_grid, BLOCK_DIM_1D, 0, stream>>>(idx_list_alt, keys,
                                                           n_runs, unit_runs);
    CUDA_CHECK_LAST_ERROR(run_unit_kernel);
    sort(keys, keys_alt, idx_list_alt, idx_list, n_runs,
         scatter_category_bits(n_units));
    deal_runs_kernel<<<run_grid, BLOCK_DIM_1D, 0, stream>>>(
        idx_list, keys, n_runs, unit_runs, n_blocks);
    CUDA_CHECK_LAST_ERROR(deal_runs_kernel);
    // Cells by (block, group), stable, so each run stays in position order.
    block_category_keys_kernel<<<strided_grid(n_cells, BLOCK_DIM_1D),
                                 BLOCK_DIM_1D, 0, stream>>>(
        keys, group_codes, keys_alt, idx_list_alt, n_cells, chunk, n_groups);
    CUDA_CHECK_LAST_ERROR(block_category_keys_kernel);
    sort(keys_alt, keys, idx_list_alt, idx_list, n_cells,
         scatter_category_bits(n_keys));
    scatter_category_offsets_kernel<<<(n_keys + 127) / 128, 128, 0, stream>>>(
        reinterpret_cast<const int*>(keys), n_cells, n_keys, block_cat_offsets);
    CUDA_CHECK_LAST_ERROR(scatter_category_offsets_kernel);
}

// ---------- Clustering arguments ----------

template <typename T>
struct ClusteringArgs {
    // Input/output; cells are sorted by group (batch, or joint category with
    // several keys).
    const T* Z_norm;
    T* R;
    __nv_bfloat16* R_bf16;  // bfloat16 assignments instead of R
    T *E, *O;
    const T *Pr_b, *theta;
    const int* joint_cats;  // several keys: J x n_covariates levels
    const int *marginal_joint_offsets, *marginal_joint_indices;
    T* O_joint;        // several keys: counts per joint category
    T* group_penalty;  // several keys: J x K log penalties

    // Workspace
    T* Y_norm;  // centroids, normalized in place
    int* idx_list;
    T* penalty;              // batches x K log penalties
    T* objective_partials;   // n_cells, then objective scratch
    int* block_cat_offsets;  // n_blocks x groups + 1, then one block's tiles
    // (n_blocks + 3) x groups x K fixed-point counts: each block's, O's, and
    // two slots of new block counts.
    long long* block_counts;
    const int* seg_start;  // centroid segments (see segment_gemm.cuh)
    int n_seg;
    double* y_scale;           // centroid fixed-point scale (1)
    long long* y_acc;          // centroid sums: limbs x K x D
    T* y_t;                    // centroids transposed (transpose_centroids)
    long long* col_workspace;  // general assignment: per-warp column sums
    bool force_general;        // tests: general assignment for every shape

    int n_cells;
    long long n_total;  // cells over all GPUs (fixed-point bounds)
    int n_pcs, n_clusters, n_batches, n_covariates, n_joint_categories;
    int n_first;  // levels of the first batch key
    int n_blocks;
    T sigma, tol;
    int max_iter;
    unsigned int seed;
    bool stabilized;
    cudaStream_t stream;
    bool initialize;  // unpenalized assignment from Y_norm instead of the loop
};

// ---------- Clustering loop ----------

// Fixed-point scales (see to_fixed) from global bounds: R values <= 1, so a
// count is at most n_total; a cell's objective term is at most
// 4 + sigma log K in magnitude.
template <typename T>
struct FixedScales {
    T count, objective;
    double count_inv, objective_inv;
    FixedScales(long long n_total, double sigma, int n_clusters) {
        int c = std::min(fixed_shift((double)n_total + 1),
                         sizeof(T) == 4 ? 31 : 62);
        int o = fixed_shift(
            (double)n_total *
            (5.0 + std::abs(sigma) * std::log(std::max(n_clusters, 2))));
        count = (T)std::ldexp(1.0, c), count_inv = std::ldexp(1.0, -c);
        objective = (T)std::ldexp(1.0, o), objective_inv = std::ldexp(1.0, -o);
    }
};

// One assignment pass over the cells `idx_list[offsets[0] ..
// offsets[n_groups])` grouped by category: writes R and the per-cell objective
// terms, and adds the exact (fixed-point) counts per category to `counts`.
// Tiles, plus one partial tile per category: general, about
// ASSIGN_TILES_PER_SM per SM; fused, one wave while that is at most three
// groups per tile, else three groups with resident centroids and two CTAs
// per SM (one centroid load per tile) and one group otherwise (so the CTAs'
// phases interleave).
constexpr int ASSIGN_TILES_PER_SM = 4;

// y_t (padded PCs x 128-cluster blocks) = Y_norm^T, zero padded; once per
// centroid update.
template <typename T>
static void transpose_centroids(const ClusteringArgs<T>& a) {
    int pcs = fused_padded_pcs(a.n_pcs);
    int k_stride = (a.n_clusters + FUSED_CLUSTER_STRIDE - 1) /
                   FUSED_CLUSTER_STRIDE * FUSED_CLUSTER_STRIDE;
    transpose_centroids_kernel<T>
        <<<(pcs * k_stride + 255) / 256, 256, 0, a.stream>>>(
            a.Y_norm, a.y_t, a.n_pcs, pcs, a.n_clusters, k_stride);
    CUDA_CHECK_LAST_ERROR(transpose_centroids_kernel);
}

template <typename T, typename RT>
static void fused_assign_pass(const ClusteringArgs<T>& a, RT* R,
                              const T* penalty, const int* offsets, int n_rows,
                              int n_groups, int* tiles, int n_sm,
                              long long* counts, T count_scale) {
    int n_pcs = a.n_pcs, n_clusters = a.n_clusters;
    cudaStream_t stream = a.stream;
    // The fused kernel holds at most 128 clusters; more take the general one.
    bool general = a.force_general || n_clusters > FUSED_CLUSTER_STRIDE;
    if (general && !a.col_workspace)
        throw std::invalid_argument(
            "this shape needs the general assignment workspace");
    // Z pieces: the widest of 16, 8 and 4 bytes dividing a row and Z.
    size_t bits = reinterpret_cast<size_t>(a.Z_norm) | 16 | n_pcs * sizeof(T);
    size_t v = bits & (~bits + 1);
    auto kernel = v == 16  ? fused_assign_kernel<T, RT, 16>
                  : v == 8 ? fused_assign_kernel<T, RT, 8>
                           : fused_assign_kernel<T, RT, sizeof(T)>;
    long long capacity = (long long)ASSIGN_TILES_PER_SM * n_sm;
    long long granule = FUSED_GROUP;
    int per_sm = 1;
    if (!general) {
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &per_sm, kernel, FUSED_THREADS, STREAM_SMEM),
                   "fused assign occupancy");
        capacity =
            std::max<long long>((long long)per_sm * n_sm - n_groups, n_sm);
        granule = FUSED_ROWS;
    }
    long long tile_rows = std::max(granule, (n_rows + capacity - 1) / capacity);
    tile_rows = (tile_rows + granule - 1) / granule * granule;
    if (!general && tile_rows > 3 * FUSED_GROUP)
        tile_rows = per_sm > 1 && fused_resident<T>(fused_padded_pcs(n_pcs))
                        ? 3 * FUSED_GROUP
                        : FUSED_GROUP;
    scatter_tile_offsets_kernel<<<1, SCATTER_SCAN_THREADS, 0, stream>>>(
        offsets, n_groups, tiles, (int)tile_rows);
    CUDA_CHECK_LAST_ERROR(scatter_tile_offsets_kernel);
    size_t max_tiles = std::min<size_t>(
        n_rows, (n_rows + tile_rows - 1) / tile_rows + n_groups);
    T term = T(-2) / a.sigma;
    auto* ucounts = reinterpret_cast<unsigned long long*>(counts);
    if (general) {
        int k_stride = (n_clusters + FUSED_CLUSTER_STRIDE - 1) /
                       FUSED_CLUSTER_STRIDE * FUSED_CLUSTER_STRIDE;
        fused_assign_general_kernel<T, RT>
            <<<max_tiles, FUSED_THREADS, 0, stream>>>(
                a.Z_norm, a.y_t, k_stride, penalty, a.idx_list, offsets, tiles,
                n_groups, (int)tile_rows, R, a.objective_partials, ucounts,
                a.col_workspace, term, a.sigma, count_scale, n_pcs, n_clusters);
    } else {
        kernel<<<max_tiles, FUSED_THREADS, STREAM_SMEM, stream>>>(
            a.Z_norm, a.y_t, penalty, a.idx_list, offsets, tiles, n_groups,
            (int)tile_rows, R, a.objective_partials, ucounts, term, a.sigma,
            count_scale, n_pcs, n_clusters);
    }
    CUDA_CHECK_LAST_ERROR(fused_assign_kernel);
}

// Objective: exact fixed-point sum of the per-cell terms of the last pass,
// plus the diversity term. Scratch: the 8 values after the per-cell terms.
template <typename T>
static double fused_objective(const ClusteringArgs<T>& a,
                              const FixedScales<T>& scales) {
    cudaStream_t stream = a.stream;
    auto* acc = reinterpret_cast<unsigned long long*>(
        (reinterpret_cast<uintptr_t>(a.objective_partials + a.n_cells) + 15) &
        ~uintptr_t(15));
    T* diversity = reinterpret_cast<T*>(acc + 1);
    cudaMemsetAsync(acc, 0, sizeof(*acc), stream);
    if (a.n_cells > 0)
        objective_fixed_kernel<T>
            <<<std::min((a.n_cells + 255) / 256, 4096), 256, 0, stream>>>(
                a.objective_partials, a.n_cells, scales.objective, acc);
    CUDA_CHECK_LAST_ERROR(objective_fixed_kernel);
    objective_diversity_kernel<T>
        <<<1, 256, 0, stream>>>(a.O, a.E, a.theta, a.sigma, a.n_batches,
                                a.n_clusters, a.stabilized, diversity);
    CUDA_CHECK_LAST_ERROR(objective_diversity_kernel);
    struct {
        long long acc;
        T diversity;
    } host;
    cuda_check(cudaMemcpyAsync(&host, acc, sizeof(host), cudaMemcpyDeviceToHost,
                               stream),
               "objective copy");
    cuda_check(cudaStreamSynchronize(stream), "objective synchronization");
    return (double)host.acc * scales.objective_inv + (double)host.diversity;
}

// Returns the last objective (0 without one).
template <typename T, typename RT>
static double fused_clustering_loop_body(const ClusteringArgs<T>& a, RT* R) {
    int n_sm = device_attribute(cudaDevAttrMultiProcessorCount);
    // Cells are grouped by batch (one key) or joint category (several keys).
    // Several keys keep counts per joint category and rebuild the marginal O.
    bool multi = a.n_covariates > 1;
    if (multi && (!a.joint_cats || !a.O_joint || !a.marginal_joint_offsets ||
                  !a.marginal_joint_indices || !a.group_penalty))
        throw std::invalid_argument(
            "fused clustering with several keys needs the joint arrays");
    int n_groups = multi ? a.n_joint_categories : a.n_batches;
    T* O_group = multi ? a.O_joint : a.O;
    const T* group_penalty = multi ? a.group_penalty : a.penalty;
    int n_blocks = a.n_blocks;
    // Assignment tiles are sized to the block so small blocks still fill the
    // GPU; their tile offsets follow the block's category offsets.
    int* assign_tiles = a.block_cat_offsets + (size_t)n_blocks * n_groups + 1;

    // Counts are exact fixed-point sums (see to_fixed): each block's, and O
    // over all cells, which changes by a block's new minus stored counts.
    FixedScales<T> scales(a.n_total, a.sigma, a.n_clusters);
    std::vector<double> objectives;
    // Centroid sums: 2 (float32) or 3 limbs of 30 bits; |Z_norm|, R <= 1.
    int y_limbs = sizeof(T) == 4 ? 2 : 3;
    size_t kd = (size_t)a.n_clusters * a.n_pcs;
    double y_scale =
        harmony_segments::scale_for_bound((double)a.n_total, y_limbs);
    cudaMemcpyAsync(a.y_scale, &y_scale, sizeof(double), cudaMemcpyHostToDevice,
                    a.stream);
    int ob_total = n_groups * a.n_clusters;
    int blocks_1d = (ob_total + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D;
    auto stored = [&](int blk) {
        return a.block_counts + (size_t)blk * ob_total;
    };
    long long* total = stored(n_blocks);
    auto slot = [&](int step) { return stored(n_blocks + 1 + (step & 1)); };
    // O = total after folding in `fold`'s new counts, minus block `hold`'s;
    // zeroes `clear`.
    auto update_counts = [&](int fold, const long long* fresh, int hold,
                             long long* clear) {
        update_counts_kernel<T><<<blocks_1d, BLOCK_DIM_1D, 0, a.stream>>>(
            total, fresh ? stored(fold) : nullptr, fresh,
            hold < 0 ? nullptr : stored(hold), clear, scales.count_inv, O_group,
            ob_total);
        CUDA_CHECK_LAST_ERROR(update_counts_kernel);
        if (!multi) return;
        materialize_marginal_from_joint_kernel<T>
            <<<strided_grid((long long)a.n_batches * a.n_clusters,
                            BLOCK_DIM_1D),
               BLOCK_DIM_1D, 0, a.stream>>>(a.O_joint, a.marginal_joint_offsets,
                                            a.marginal_joint_indices, a.O,
                                            a.n_batches, a.n_clusters);
        CUDA_CHECK_LAST_ERROR(materialize_marginal_from_joint_kernel);
    };
    // E (and the penalty unless null) from the full O.
    auto penalty_from_counts = [&](T* penalty) {
        penalty_from_counts_kernel<T><<<a.n_clusters, 256, 0, a.stream>>>(
            a.O, a.Pr_b, a.theta, a.E, penalty, a.n_batches, a.n_clusters,
            a.n_first, a.stabilized);
        CUDA_CHECK_LAST_ERROR(penalty_from_counts_kernel);
    };
    // Block b's cells are idx_list[starts[b] .. starts[b + 1]).
    std::vector<int> starts(n_blocks + 1);
    cuda_check(
        cudaMemcpy2DAsync(starts.data(), sizeof(int), a.block_cat_offsets,
                          n_groups * sizeof(int), sizeof(int), n_blocks + 1,
                          cudaMemcpyDeviceToHost, a.stream),
        "block starts copy");
    cuda_check(cudaStreamSynchronize(a.stream), "block starts copy");
    auto assign = [&](int blk, const T* penalty, long long* counts) {
        if (starts[blk + 1] == starts[blk]) return;
        fused_assign_pass(a, R, penalty,
                          a.block_cat_offsets + (size_t)blk * n_groups,
                          starts[blk + 1] - starts[blk], n_groups, assign_tiles,
                          n_sm, counts, scales.count);
    };
    // E from the full O, then the objective.
    auto objective = [&](bool evaluate) {
        penalty_from_counts(nullptr);
        if (evaluate) objectives.push_back(fused_objective(a, scales));
    };
    if (a.initialize) {
        // Unpenalized assignment of every block from the k-means centroids.
        transpose_centroids(a);
        cudaMemsetAsync(a.block_counts, 0,
                        (size_t)(n_blocks + 3) * ob_total * sizeof(long long),
                        a.stream);
        for (int blk = 0; blk < n_blocks; ++blk)
            assign(blk, nullptr, stored(blk));
        sum_blocks_kernel<<<blocks_1d, BLOCK_DIM_1D, 0, a.stream>>>(
            a.block_counts, n_blocks, total, ob_total);
        CUDA_CHECK_LAST_ERROR(sum_blocks_kernel);
        update_counts(-1, nullptr, -1, nullptr);
        objective(true);
    }
    constexpr int WINDOW_SIZE = 3;
    for (int iter = 0; iter < a.max_iter; iter++) {
        // Centroids Y = Z_norm^T R, exact over the segments.
        cudaMemsetAsync(a.y_acc, 0, y_limbs * kd * sizeof(long long), a.stream);
        cuda_check(
            harmony_segments::segment_rtz(
                a.Z_norm, R, a.n_pcs, a.n_clusters, a.n_clusters, a.seg_start,
                nullptr, a.n_seg, a.y_scale, a.y_acc, y_limbs, a.stream),
            "centroids");
        cuda_check(harmony_segments::finalize_rtz(
                       a.y_acc, 1, a.n_pcs, a.n_clusters, y_limbs, a.y_scale,
                       a.Y_norm, 0, a.n_pcs, a.stream),
                   "centroids");
        l2_row_normalize(a.Y_norm, a.Y_norm, a.n_clusters, a.n_pcs, a.stream);
        transpose_centroids(a);

        // Blocks in a rotated order, each assigned against O without its
        // stored counts. A block's new counts enter O one block late (before
        // the block after next), the same on any number of GPUs, where their
        // exchange overlaps the next assignment.
        int shift = (int)((a.seed + (unsigned)iter) % (unsigned)n_blocks);
        auto block = [&](int step) { return (step + shift) % n_blocks; };
        for (int step = 0; step < n_blocks; ++step) {
            int fold = step - 2;
            update_counts(fold < 0 ? -1 : block(fold),
                          fold < 0 ? nullptr : slot(fold), block(step),
                          slot(step));
            penalty_from_counts(a.penalty);
            if (multi) {
                joint_log_penalty_kernel<T>
                    <<<strided_grid(ob_total, BLOCK_DIM_1D), BLOCK_DIM_1D, 0,
                       a.stream>>>(a.penalty, a.joint_cats, a.n_covariates,
                                   n_groups, a.n_clusters, a.group_penalty);
                CUDA_CHECK_LAST_ERROR(joint_log_penalty_kernel);
            }
            assign(block(step), group_penalty, slot(step));
        }
        for (int fold = std::max(0, n_blocks - 2); fold < n_blocks; ++fold)
            update_counts(block(fold), slot(fold), -1, nullptr);

        // Objective: per-cell terms from the assignment pass plus diversity.
        // The first convergence decision needs WINDOW_SIZE + 1 objectives;
        // when the budget ends there anyway, only the last one is used.
        objective(a.max_iter > WINDOW_SIZE + 1 || iter + 1 == a.max_iter);
        if (static_cast<int>(objectives.size()) >= WINDOW_SIZE + 1) {
            double obj_old = 0, obj_new = 0;
            for (int i = 0; i < WINDOW_SIZE; i++) {
                obj_old += objectives[objectives.size() - 2 - i];
                obj_new += objectives[objectives.size() - 1 - i];
            }
            if ((obj_old - obj_new) < a.tol * std::abs(obj_old)) break;
        }
    }
    return objectives.empty() ? 0.0 : (double)(T)objectives.back();
}

template <typename T>
static double fused_clustering_loop_impl(const ClusteringArgs<T>& a) {
    if (!a.R_bf16) return fused_clustering_loop_body(a, a.R);
    if constexpr (std::is_same_v<T, float>)
        return fused_clustering_loop_body(a, a.R_bf16);
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
           gpu_array_c<const T, Device> theta, gpu_array_c<T, Device> Y_norm,
           gpu_array_c<int, Device> idx_list, gpu_array_c<T, Device> penalty,
           gpu_array_c<T, Device> objective_partials,
           gpu_array_c<int, Device> block_cat_offsets,
           gpu_array_c<long long, Device> block_counts,
           gpu_array_c<const int, Device> seg_start,
           gpu_array_c<double, Device> y_scale,
           gpu_array_c<long long, Device> y_acc,
           std::optional<gpu_array_c<uint16_t, Device>> R_bf16, Ints joint_cats,
           Ints marginal_joint_offsets, Ints marginal_joint_indices,
           Opt O_joint, Opt group_penalty, gpu_array_c<T, Device> y_t,
           std::optional<gpu_array_c<long long, Device>> col_workspace,
           bool force_general, int n_cells, long long n_total, int n_pcs,
           int n_clusters, int n_batches, int n_covariates,
           int n_joint_categories, int n_first, int n_blocks, double sigma,
           double tol, int max_iter, unsigned int seed, bool stabilized,
           std::uintptr_t stream, bool initialize) {
            int n_groups = n_covariates > 1 ? n_joint_categories : n_batches;
            size_t max_tiles =
                (size_t)ASSIGN_TILES_PER_SM *
                    device_attribute(cudaDevAttrMultiProcessorCount) +
                n_groups + 1;
            if (block_counts.size() <
                    (size_t)(n_blocks + 3) * n_groups * n_clusters ||
                objective_partials.size() < (size_t)n_cells + 8 ||
                y_acc.size() < (size_t)3 * n_clusters * n_pcs ||
                y_t.size() < (size_t)fused_padded_pcs(n_pcs) *
                                 ((n_clusters + 127) / 128 * 128) ||
                reinterpret_cast<uintptr_t>(y_t.data()) % 16 ||
                (col_workspace &&
                 col_workspace->size() <
                     max_tiles * (FUSED_THREADS / 32) * n_clusters))
                throw std::invalid_argument(
                    "clustering workspace is too small");
            auto ptr = [](auto& a) { return a ? a->data() : nullptr; };
            auto* r_bf16 = reinterpret_cast<__nv_bfloat16*>(ptr(R_bf16));
            ClusteringArgs<T> a{Z_norm.data(),
                                R.data(),
                                r_bf16,
                                E.data(),
                                O.data(),
                                Pr_b.data(),
                                theta.data(),
                                ptr(joint_cats),
                                ptr(marginal_joint_offsets),
                                ptr(marginal_joint_indices),
                                ptr(O_joint),
                                ptr(group_penalty),
                                Y_norm.data(),
                                idx_list.data(),
                                penalty.data(),
                                objective_partials.data(),
                                block_cat_offsets.data(),
                                block_counts.data(),
                                seg_start.data(),
                                (int)seg_start.size() - 1,
                                y_scale.data(),
                                y_acc.data(),
                                y_t.data(),
                                ptr(col_workspace),
                                force_general,
                                n_cells,
                                n_total > 0 ? n_total : n_cells,
                                n_pcs,
                                n_clusters,
                                n_batches,
                                n_covariates,
                                n_joint_categories,
                                n_first > 0 ? n_first : n_batches,
                                n_blocks,
                                (T)sigma,
                                (T)tol,
                                max_iter,
                                seed,
                                stabilized,
                                (cudaStream_t)stream,
                                initialize};
            return fused_clustering_loop_impl(a);
        },
        "Z_norm"_a, nb::kw_only(), "R"_a, "E"_a, "O"_a, "Pr_b"_a, "theta"_a,
        "Y_norm"_a, "idx_list"_a, "penalty"_a, "objective_partials"_a,
        "block_cat_offsets"_a, "block_counts"_a, "seg_start"_a, "y_scale"_a,
        "y_acc"_a, "R_bf16"_a = nb::none(), "joint_cats"_a = nb::none(),
        "marginal_joint_offsets"_a = nb::none(),
        "marginal_joint_indices"_a = nb::none(), "O_joint"_a = nb::none(),
        "group_penalty"_a = nb::none(), "y_t"_a, "col_workspace"_a = nb::none(),
        "force_general"_a = false, "n_cells"_a, "n_total"_a = 0, "n_pcs"_a,
        "n_clusters"_a, "n_batches"_a, "n_covariates"_a = 1,
        "n_joint_categories"_a = 0, "n_first"_a = 0, "n_blocks"_a, "sigma"_a,
        "tol"_a = 0.0, "max_iter"_a, "seed"_a = 0, "stabilized"_a,
        "stream"_a = 0, "initialize"_a = false);
}

template <typename T, typename Device>
static void register_kmeans(nb::module_& m) {
    m.def(
        "select_kmeans_center",
        [](gpu_array_c<const T, Device> X, gpu_array_c<const T, Device> weights,
           gpu_array_c<const double, Device> uniforms,
           gpu_array_c<T, Device> centers, gpu_array_c<double, Device> totals,
           gpu_array_c<int, Device> n_draws, int cluster,
           std::uintptr_t stream) {
            int n_rows = (int)X.shape(0), n_cols = (int)X.shape(1);
            int n_tiles = (n_rows + KMEANS_WEIGHT_TILE_ROWS - 1) /
                          KMEANS_WEIGHT_TILE_ROWS;
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
    m.def(
        "scatter_add",
        [](gpu_array_c<const T, Device> values,
           gpu_array_c<const int, Device> categories,
           gpu_array_c<T, Device> out, gpu_array_c<uint8_t, Device> workspace,
           std::uintptr_t stream) {
            int n_rows = (int)values.shape(0), n_cols = (int)values.shape(1);
            int n_categories = (int)out.shape(0);
            if (workspace.size() <
                scatter_temp_bytes(n_rows, n_cols, n_categories, sizeof(T)))
                throw std::invalid_argument("scatter workspace is too small");
            scatter_reduce(values.data(), categories.data(), n_rows, n_cols,
                           n_categories, out.data(), workspace.data(),
                           (cudaStream_t)stream);
        },
        "values"_a, nb::kw_only(), "categories"_a, "out"_a, "workspace"_a,
        "stream"_a = 0);
    m.def(
        "kmeans_assign",
        [](gpu_array_c<const T, Device> X, gpu_array_c<const T, Device> centers,
           gpu_array_c<int, Device> labels, gpu_array_c<T, Device> minimum,
           std::uintptr_t stream) {
            int n_rows = (int)X.shape(0), n_cols = (int)X.shape(1);
            int n_clusters = (int)centers.shape(0);
            if (n_clusters > FUSED_CLUSTER_STRIDE)
                throw std::invalid_argument(
                    "kmeans_assign supports at most 128 clusters");
            // Centers transposed and FUSED_ROWS staged rows per warp.
            size_t smem =
                (size_t)fused_padded_pcs(n_cols) *
                (FUSED_CLUSTER_STRIDE + FUSED_THREADS / 32 * FUSED_ROWS) *
                sizeof(T);
            if (smem > 48 * 1024)
                cuda_check(
                    cudaFuncSetAttribute(
                        kmeans_assign_kernel<T>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem),
                    "kmeans assign shared memory");
            // FUSED_ROWS rows per warp, at most four CTAs per SM.
            constexpr int rows = FUSED_ROWS * FUSED_THREADS / 32;
            int grid = (int)std::clamp<long long>(
                ((long long)n_rows + rows - 1) / rows, 1,
                device_attribute(cudaDevAttrMultiProcessorCount) * 4);
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
    m.def(
        "l2_row_normalize",
        [](gpu_array_c<const T, Device> src, gpu_array_c<T, Device> dst,
           std::uintptr_t stream) {
            l2_row_normalize(src.data(), dst.data(), (int)src.shape(0),
                             (int)src.shape(1), (cudaStream_t)stream);
        },
        "src"_a, nb::kw_only(), "dst"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def(
        "draw_blocks",
        [](gpu_array_c<const int, Device> group_codes,
           gpu_array_c<int, Device> idx_list,
           gpu_array_c<int, Device> block_cat_offsets,
           gpu_array_c<int, Device> idx_list_alt,
           gpu_array_c<unsigned int, Device> sort_keys,
           gpu_array_c<unsigned int, Device> sort_keys_alt,
           gpu_array_c<uint8_t, Device> cub_temp, int n_groups, int n_blocks,
           int unit, unsigned int seed, int shuffle_chunk, long long first,
           std::uintptr_t stream) {
            draw_blocks(group_codes.data(), (int)group_codes.size(), first,
                        n_groups, n_blocks, unit, seed, shuffle_chunk,
                        idx_list.data(), block_cat_offsets.data(),
                        idx_list_alt.data(), sort_keys.data(),
                        sort_keys_alt.data(), cub_temp.data(), cub_temp.size(),
                        (cudaStream_t)stream);
        },
        "group_codes"_a, nb::kw_only(), "idx_list"_a, "block_cat_offsets"_a,
        "idx_list_alt"_a, "sort_keys"_a, "sort_keys_alt"_a, "cub_temp"_a,
        "n_groups"_a, "n_blocks"_a, "unit"_a, "seed"_a, "shuffle_chunk"_a = 1,
        "first"_a = 0, "stream"_a = 0);
    m.def(
        "get_cub_sort_temp_bytes",
        [](int n_cells) { return sort_temp_bytes(n_cells, 32); }, nb::kw_only(),
        "n_cells"_a);
    m.def(
        "get_scatter_temp_bytes",
        [](int n_rows, int n_cols, int n_categories, int itemsize) {
            return scatter_temp_bytes(n_rows, n_cols, n_categories, itemsize);
        },
        nb::kw_only(), "n_rows"_a, "n_cols"_a, "n_categories"_a,
        "itemsize"_a = 4);
    register_clustering_loop<float, Device>(m);
    register_clustering_loop<double, Device>(m);
    register_kmeans<float, Device>(m);
    register_kmeans<double, Device>(m);
}

NB_MODULE(_harmony_clustering_cuda, m) {
    m.attr("KMEANS_WEIGHT_TILE_ROWS") = nb::int_(KMEANS_WEIGHT_TILE_ROWS);
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
