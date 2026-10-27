#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../../nb_types.h"

#include "../scatter/kernels_scatter_reduce.cuh"
#include "../comm.cuh"
#include "../segment_gemm.cuh"
#include "kernels_clustering.cuh"

using namespace nb::literals;

constexpr int BLOCK_DIM_1D = 256;

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
    int n_runs = int(((long long)n_cells + chunk - 1) / chunk),
        unit_runs = unit / chunk;
    int n_units = int(((long long)n_runs + unit_runs - 1) / unit_runs);
    int n_keys = n_blocks * n_groups;  // below 2^31 (see harmonize)
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
    const int* joints;      // each cell's group (several keys: joint category)

    // Workspace
    T* Y_norm;  // centroids, normalized in place
    int* idx_list;
    T* penalty;             // batches x K log penalties
    T* objective_partials;  // n_cells, then objective scratch
    // n_blocks x groups + 1, then each block's tiles (groups + 1); groups are
    // batches, one with several keys.
    int* block_cat_offsets;
    const int* block_starts;  // host: block b is idx_list[starts[b] ..
                              // starts[b + 1])
    // (n_stored + 3) x batches x K fixed-point counts: each block's (n_stored
    // = n_blocks) or three recounted ones, O's, and two slots of new counts.
    long long* block_counts;
    int n_stored;
    const int* seg_start;  // centroid segments (see segment_gemm.cuh)
    int n_seg;
    const short* y_bound;      // centroid sum bound exponents (n_pcs)
    long long* y_acc;          // centroid sums: limbs x K x D
    T* y_t;                    // centroids transposed (transpose_centroids)
    long long* col_workspace;  // cluster blocks: per-warp column sums, then
                               // the tile counter
    T* dots_workspace;         // cluster blocks: dot products, 64 rows a CTA
    int n_slots;               // cluster blocks: CTAs the scratch holds
    bool force_blocks;         // tests: cluster blocks for every shape

    int n_cells;
    long long n_total;  // cells over all GPUs (fixed-point bounds)
    int n_pcs, n_clusters, n_batches, n_covariates;
    int n_first;  // levels of the first batch key
    int n_blocks;
    T sigma, tol;
    int max_iter;
    unsigned int seed;
    bool stabilized;
    cudaStream_t stream;
    bool initialize;  // unpenalized assignment from Y_norm instead of the loop
    harmony_comm::Comm* comm;  // several GPUs: this one's rank, else null
    int rank;
};

// ---------- Clustering loop ----------

// An assignment pass over a block's cells `idx_list[offsets[0] ..
// offsets[n_groups])` grouped by category (`assign` below): writes R and the
// per-cell objective terms, and adds the exact (fixed-point) counts per
// category to `counts`. Tiles (assign_tile_rows), plus one partial tile per
// category: cluster blocks, about
// ASSIGN_TILES_PER_SM per SM, taken by the CTAs whose scratch the workspace
// holds (n_slots, see fused_assign_kernel); otherwise one wave while that is at
// most three groups per tile, else three groups with resident centroids and two
// CTAs per SM (one centroid load per tile) and one group otherwise (so the
// CTAs' phases interleave).
constexpr int ASSIGN_TILES_PER_SM = 4;

// y_t (padded PCs x 128-cluster blocks) = Y_norm^T, zero padded; once per
// centroid update.
template <typename T>
static void transpose_centroids(const ClusteringArgs<T>& a) {
    int pcs = fused_padded_pcs(a.n_pcs);
    int k_stride = (a.n_clusters + FUSED_CLUSTER_STRIDE - 1) /
                   FUSED_CLUSTER_STRIDE * FUSED_CLUSTER_STRIDE;
    transpose_centroids_kernel<T>
        <<<strided_grid((long long)pcs * k_stride, 256), 256, 0, a.stream>>>(
            a.Y_norm, a.y_t, a.n_pcs, pcs, a.n_clusters, k_stride);
    CUDA_CHECK_LAST_ERROR(transpose_centroids_kernel);
}

// The assignment kernel of a call: cluster blocks (more than 128 clusters;
// tests: any shape), else Z pieces of the widest of 16, 8 and 4 bytes
// dividing a row and Z; JOINT with several keys.
template <typename T, typename RT>
static auto assign_kernel(const ClusteringArgs<T>& a, bool blocks) {
    size_t bits = reinterpret_cast<size_t>(a.Z_norm) | 16 | a.n_pcs * sizeof(T);
    size_t v = bits & (~bits + 1);
    auto pick = [&](auto joint) {
        constexpr bool J = decltype(joint)::value;
        return blocks    ? fused_assign_kernel<T, RT, sizeof(T), true, J>
               : v == 16 ? fused_assign_kernel<T, RT, 16, false, J>
               : v == 8  ? fused_assign_kernel<T, RT, 8, false, J>
                         : fused_assign_kernel<T, RT, sizeof(T), false, J>;
    };
    auto kernel = pick(std::false_type{});
    if constexpr (std::is_same_v<RT, T>) {  // bfloat16 R needs one key
        if (a.n_covariates > 1) kernel = pick(std::true_type{});
    }
    return kernel;
}

// Rows per assignment tile of a block of n_rows cells (see
// ASSIGN_TILES_PER_SM); per_sm: resident CTAs per SM of the kernel.
template <typename T>
static int assign_tile_rows(const ClusteringArgs<T>& a, int n_rows,
                            int n_groups, int n_sm, int per_sm, bool blocks) {
    long long capacity = (long long)ASSIGN_TILES_PER_SM * n_sm;
    long long granule = FUSED_GROUP;
    if (!blocks) {
        capacity =
            std::max<long long>((long long)per_sm * n_sm - n_groups, n_sm);
        granule = FUSED_ROWS;
    }
    long long tile_rows = std::max(granule, (n_rows + capacity - 1) / capacity);
    tile_rows = (tile_rows + granule - 1) / granule * granule;
    if (!blocks && tile_rows > 3 * FUSED_GROUP)
        tile_rows = per_sm > 1 && fused_resident<T>(fused_padded_pcs(a.n_pcs))
                        ? 3 * FUSED_GROUP
                        : FUSED_GROUP;
    return (int)tile_rows;
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
    cuda_check(cudaMemsetAsync(acc, 0, sizeof(*acc), stream),
               "objective memset");
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
    harmony_comm::allreduce(a.comm, a.rank, reinterpret_cast<long long*>(acc),
                            1, stream);
    cuda_check(cudaMemcpyAsync(&host, acc, sizeof(host), cudaMemcpyDeviceToHost,
                               stream),
               "objective copy");
    cuda_check(cudaStreamSynchronize(stream), "objective synchronization");
    return (double)host.acc * scales.objective_inv + (double)host.diversity;
}

// Returns the last objective (0 without one).
template <typename T, typename RT>
static double fused_clustering_loop_body(const ClusteringArgs<T>& a, RT* R) {
    // Tiles hold one batch (one key) or any cells of a block, in joint
    // category order (several keys; see fused_assign_kernel's JOINT). Counts
    // are per batch level either way.
    bool multi = a.n_covariates > 1;
    if (multi && (!a.joint_cats || !a.joints))
        throw std::invalid_argument(
            "fused clustering with several keys needs the joint arrays");
    if (multi && !std::is_same_v<RT, T>)
        throw std::invalid_argument("bfloat16 assignments need one batch key");
    int n_groups = multi ? 1 : a.n_batches;
    int n_blocks = a.n_blocks;

    // Counts are exact fixed-point sums (see to_fixed): each block's, and O
    // over all cells, which changes by a block's new minus stored counts.
    FixedScales<T> scales(a.n_total, a.sigma, a.n_clusters);
    std::vector<double> objectives;
    // Centroid sums: 2 (float32) or 3 limbs of 30 bits; |Z_norm|, R <= 1.
    int y_limbs = sizeof(T) == 4 ? 2 : 3;
    size_t kd = (size_t)a.n_clusters * a.n_pcs;
    // Every segment adds into the top limb, whose atomics ran about half as
    // fast across a 128 KiB boundary (measured on an RTX PRO 6000 Blackwell):
    // float32 moves it up to the next one when its spare limb allows.
    const uintptr_t span = 1 << 17,
                    gap = (span - (uintptr_t)a.y_acc % span) % span;
    const size_t spare = (3 - y_limbs) * kd * sizeof(long long);
    long long* y_acc = a.y_acc + (gap <= spare ? gap / sizeof(long long) : 0);
    int ob_total = a.n_batches * a.n_clusters;
    int blocks_1d = (ob_total + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D;
    auto stored = [&](int blk) {
        return a.block_counts + (size_t)blk * ob_total;
    };
    long long* total = stored(a.n_stored);
    auto slot = [&](int step) { return stored(a.n_stored + 1 + (step & 1)); };
    // Each block's counts are kept, unless that table would outgrow R; then
    // the old counts are recounted from R (see the block loop).
    bool keep = a.n_stored >= n_blocks;
    // O = total after folding in new counts `fresh` (old ones `fold`), minus
    // `hold`; starts `clear` (see update_counts_kernel).
    auto update_counts =
        [&](long long* fold, const long long* fresh, const long long* hold,
            long long* clear, long long* cur = nullptr,
            const long long* next = nullptr, const long long* after = nullptr) {
            update_counts_kernel<T><<<blocks_1d, BLOCK_DIM_1D, 0, a.stream>>>(
                total, fold, fresh, hold, clear, scales.count_inv, a.O,
                ob_total, cur, next, after);
            CUDA_CHECK_LAST_ERROR(update_counts_kernel);
        };
    // E (and the penalty unless null) from the full O.
    auto penalty_from_counts = [&](T* penalty) {
        penalty_from_counts_kernel<T><<<a.n_clusters, 256, 0, a.stream>>>(
            a.O, a.Pr_b, a.theta, a.E, penalty, a.n_batches, a.n_clusters,
            a.n_first, a.stabilized);
        CUDA_CHECK_LAST_ERROR(penalty_from_counts_kernel);
    };
    // Block b's cells are idx_list[starts[b] .. starts[b + 1]); its
    // assignment tiles are sized to it so small blocks still fill the GPU,
    // their table following the block-group offsets. Both are set up once:
    // the blocks are fixed per harmonize call.
    const int* starts = a.block_starts;
    int* tiles = a.block_cat_offsets + (size_t)n_blocks * n_groups + 1;
    bool blocks = a.force_blocks || a.n_clusters > FUSED_CLUSTER_STRIDE;
    auto kernel = assign_kernel<T, RT>(a, blocks);
    int n_sm = device_attribute(cudaDevAttrMultiProcessorCount), per_sm = 1;
    if (!blocks)
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &per_sm, kernel, FUSED_THREADS, STREAM_SMEM),
                   "fused assign occupancy");
    if (blocks && a.n_slots < 1)
        throw std::invalid_argument(
            "this shape needs the cluster-block assignment workspace");
    // Cluster blocks: the resident CTAs claim tiles through the counter after
    // their scratch, zeroed before each pass.
    int* next_tile =
        blocks ? reinterpret_cast<int*>(a.col_workspace +
                                        (size_t)a.n_slots *
                                            (FUSED_THREADS / 32) * a.n_clusters)
               : nullptr;
    std::vector<int> tile_rows(n_blocks);
    for (int blk = 0; blk < n_blocks; ++blk) {
        int n_rows = starts[blk + 1] - starts[blk];
        if (n_rows == 0) continue;
        tile_rows[blk] =
            assign_tile_rows(a, n_rows, n_groups, n_sm, per_sm, blocks);
        scatter_tile_offsets_kernel<<<1, SCATTER_SCAN_THREADS, 0, a.stream>>>(
            a.block_cat_offsets + (size_t)blk * n_groups, n_groups,
            tiles + (size_t)blk * (n_groups + 1), tile_rows[blk]);
        CUDA_CHECK_LAST_ERROR(scatter_tile_offsets_kernel);
    }
    int k_stride = (a.n_clusters + FUSED_CLUSTER_STRIDE - 1) /
                   FUSED_CLUSTER_STRIDE * FUSED_CLUSTER_STRIDE;
    auto assign = [&](int blk, const T* penalty, long long* counts) {
        int n_rows = starts[blk + 1] - starts[blk], rows = tile_rows[blk];
        if (n_rows == 0) return;
        size_t max_tiles =
            std::min<size_t>(n_rows, (n_rows + rows - 1) / rows + n_groups);
        if (blocks) {
            max_tiles = std::min<size_t>(max_tiles, a.n_slots);
            cuda_check(cudaMemsetAsync(next_tile, 0, sizeof(int), a.stream),
                       "tile counter memset");
        }
        kernel<<<max_tiles, FUSED_THREADS, STREAM_SMEM, a.stream>>>(
            a.Z_norm, a.y_t, k_stride, penalty, a.idx_list,
            a.block_cat_offsets + (size_t)blk * n_groups,
            tiles + (size_t)blk * (n_groups + 1), n_groups, rows, R,
            a.objective_partials, reinterpret_cast<unsigned long long*>(counts),
            a.dots_workspace, a.col_workspace, T(-2) / a.sigma, a.sigma,
            scales.count, a.n_pcs, a.n_clusters, a.joints, a.joint_cats,
            a.n_covariates, next_tile);
        CUDA_CHECK_LAST_ERROR(fused_assign_kernel);
    };
    // Several GPUs: a block's new counts are summed on a side stream while
    // the next block is assigned, and folded in once summed.
    struct Side {
        cudaStream_t stream = nullptr;
        cudaEvent_t ready[2], summed[2];
        ~Side() {
            if (!stream) return;
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
            for (int s = 0; s < 2; ++s)
                cudaEventDestroy(ready[s]), cudaEventDestroy(summed[s]);
        }
    } side;
    if (a.comm) {
        cuda_check(
            cudaStreamCreateWithFlags(&side.stream, cudaStreamNonBlocking),
            "exchange stream");
        for (int s = 0; s < 2; ++s) {
            cudaEventCreateWithFlags(&side.ready[s], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&side.summed[s], cudaEventDisableTiming);
        }
    }
    auto exchange = [&](int step) {
        if (!a.comm) return;
        cudaEventRecord(side.ready[step & 1], a.stream);
        cudaStreamWaitEvent(side.stream, side.ready[step & 1]);
        harmony_comm::allreduce(a.comm, a.rank, slot(step), ob_total,
                                side.stream);
        cudaEventRecord(side.summed[step & 1], side.stream);
    };
    auto summed = [&](int step) {
        if (a.comm) cudaStreamWaitEvent(a.stream, side.summed[step & 1]);
    };
    // E from the full O, then the objective.
    auto objective = [&](bool evaluate) {
        penalty_from_counts(nullptr);
        if (evaluate) objectives.push_back(fused_objective(a, scales));
    };
    if (a.initialize) {
        // Unpenalized assignment of every block from the k-means centroids.
        transpose_centroids(a);
        cuda_check(cudaMemsetAsync(
                       a.block_counts, 0,
                       (size_t)(a.n_stored + 3) * ob_total * sizeof(long long),
                       a.stream),
                   "block counts memset");
        for (int blk = 0; blk < n_blocks; ++blk)
            assign(blk, nullptr, keep ? stored(blk) : total);
        harmony_comm::allreduce(a.comm, a.rank, keep ? a.block_counts : total,
                                (size_t)(keep ? n_blocks : 1) * ob_total,
                                a.stream);
        if (keep) {
            sum_blocks_kernel<<<blocks_1d, BLOCK_DIM_1D, 0, a.stream>>>(
                a.block_counts, n_blocks, total, ob_total);
            CUDA_CHECK_LAST_ERROR(sum_blocks_kernel);
        }
        update_counts(nullptr, nullptr, nullptr, nullptr);
        objective(true);
    }
    constexpr int WINDOW_SIZE = 3;
    for (int iter = 0; iter < a.max_iter; iter++) {
        // Centroids Y = Z_norm^T R, exact over the segments.
        cuda_check(cudaMemsetAsync(y_acc, 0, y_limbs * kd * sizeof(long long),
                                   a.stream),
                   "centroid sums memset");
        cuda_check(
            harmony_segments::segment_rtz(
                a.Z_norm, R, a.n_pcs, a.n_clusters, a.n_clusters, a.seg_start,
                nullptr, a.n_seg, a.y_bound, y_acc, y_limbs, a.stream),
            "centroids");
        harmony_comm::allreduce(a.comm, a.rank, y_acc, y_limbs * kd, a.stream);
        cuda_check(harmony_segments::finalize_rtz(
                       y_acc, 1, a.n_pcs, a.n_clusters, y_limbs, a.y_bound,
                       a.Y_norm, 0, a.n_pcs, a.stream),
                   "centroids");
        l2_row_normalize(a.Y_norm, a.Y_norm, a.n_clusters, a.n_pcs, a.stream);
        transpose_centroids(a);

        // Blocks in a rotated order, each assigned against O without its
        // stored counts. A block's new counts enter O one block late (before
        // the block after next), the same on any number of GPUs, where their
        // exchange overlaps the next assignment.
        int shift = (int)((a.seed + (unsigned)iter) % (unsigned)n_blocks);
        auto block = [&](int step) {
            return int(((long long)step + shift) % n_blocks);
        };
        auto kept = [&](int step) {  // block(step)'s kept counts, if kept
            return keep && step >= 0 ? stored(block(step)) : nullptr;
        };
        // Recounting: block(j)'s old counts (a ring of three, this GPU's) are
        // counted from R at step j - 2 and zeroed when read at step j.
        // slot(step) starts from old(step + 1) - old(step) - old(step + 2):
        // folded in two steps later, with block(step)'s new counts, it turns O
        // without block(step + 1) into O without block(step + 2), and after
        // the last step total into all counts. Steps -2 and -1 start the
        // first two; each step sums one slot over the GPUs, as when kept.
        auto old = [&](int j) {
            return !keep && j >= 0 && j < n_blocks ? stored(j % 3) : nullptr;
        };
        auto recount = [&](int j) {
            int blk = block(j), first = starts[blk], end = starts[blk + 1];
            long long n = (long long)(end - first + RECOUNT_CELLS - 1) /
                          RECOUNT_CELLS * a.n_clusters;
            if (n == 0) return;
            recount_kernel<T>
                <<<strided_grid(n, BLOCK_DIM_1D), BLOCK_DIM_1D, 0, a.stream>>>(
                    R, a.n_clusters, a.idx_list, first, end, a.joints,
                    multi ? a.joint_cats : nullptr, a.n_covariates,
                    scales.count,
                    reinterpret_cast<unsigned long long*>(old(j)));
            CUDA_CHECK_LAST_ERROR(recount_kernel);
        };
        int first = keep ? 0 : -2;
        for (int step = first; step < n_blocks; ++step) {
            int fold = step - 2;
            if (!keep && step + 2 < n_blocks) recount(step + 2);
            if (fold >= first) summed(fold);
            update_counts(kept(fold), fold >= first ? slot(fold) : nullptr,
                          kept(step), slot(step), old(step), old(step + 1),
                          old(step + 2));
            if (step >= 0) {
                penalty_from_counts(a.penalty);
                assign(block(step), a.penalty, slot(step));
            }
            exchange(step);
        }
        for (int fold = std::max(0, n_blocks - 2); fold < n_blocks; ++fold) {
            summed(fold);
            update_counts(kept(fold), slot(fold), nullptr, nullptr);
        }

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

// Resident CTAs of the cluster-block assignment on this GPU, the most of its
// variants for T: each takes a slot of its scratch (see fused_assign_kernel).
template <typename T>
static int assign_slots() {
    int most = 1;
    auto fit = [&](auto kernel) {
        int n;
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &n, kernel, FUSED_THREADS, STREAM_SMEM),
                   "fused assign occupancy");
        most = std::max(most, n);
    };
    fit(fused_assign_kernel<T, T, sizeof(T), true, false>);
    fit(fused_assign_kernel<T, T, sizeof(T), true, true>);
    if constexpr (std::is_same_v<T, float>) {
        fit(fused_assign_kernel<T, __nv_bfloat16, sizeof(T), true, false>);
    }
    return most * device_attribute(cudaDevAttrMultiProcessorCount);
}

// Shared memory of kmeans_assign_kernel: the centers transposed and
// FUSED_ROWS staged rows per warp.
static size_t kmeans_assign_smem(int n_cols, int itemsize) {
    return (size_t)fused_padded_pcs(n_cols) *
           (FUSED_CLUSTER_STRIDE + FUSED_THREADS / 32 * FUSED_ROWS) * itemsize;
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
           host_array<const int> block_starts,
           gpu_array_c<long long, Device> block_counts,
           gpu_array_c<const int, Device> seg_start,
           gpu_array_c<const short, Device> y_bound,
           gpu_array_c<long long, Device> y_acc,
           std::optional<gpu_array_c<uint16_t, Device>> R_bf16, Ints joint_cats,
           Ints joints, gpu_array_c<T, Device> y_t,
           std::optional<gpu_array_c<long long, Device>> col_workspace,
           Opt dots_workspace, bool force_blocks, int n_cells,
           long long n_total, int n_pcs, int n_clusters, int n_batches,
           int n_covariates, int n_first, int n_blocks, double sigma,
           double tol, int max_iter, unsigned int seed, bool stabilized,
           std::uintptr_t stream, bool initialize, std::uintptr_t comm,
           int rank) {
            bool multi = n_covariates > 1;
            int n_groups = multi ? 1 : n_batches;
            size_t cells = n_cells, kb = (size_t)n_batches * n_clusters;
            size_t k_stride = (n_clusters + FUSED_CLUSTER_STRIDE - 1) /
                              FUSED_CLUSTER_STRIDE * FUSED_CLUSTER_STRIDE;
            // A smaller table holds three blocks' counts, recounted by group.
            bool recount = block_counts.size() < (n_blocks + 3) * kb;
            if (kb == 0 || Z_norm.size() < cells * n_pcs ||
                (R_bf16 ? R_bf16->size() : R.size()) < cells * n_clusters ||
                idx_list.size() < cells || E.size() < kb || O.size() < kb ||
                penalty.size() < kb ||
                Y_norm.size() < (size_t)n_clusters * n_pcs ||
                ((multi || recount) && (!joints || joints->size() < cells)) ||
                (multi && (!joint_cats || joint_cats->ndim() != 2 ||
                           joint_cats->shape(1) != (size_t)n_covariates)) ||
                block_cat_offsets.size() <
                    (size_t)n_blocks * (2 * n_groups + 1) + 1 ||
                block_starts.size() < (size_t)n_blocks + 1 ||
                block_counts.size() < (std::min(n_blocks, 3) + 3) * kb ||
                objective_partials.size() < cells + 8 ||
                y_acc.size() < (size_t)3 * n_clusters * n_pcs ||
                y_bound.size() < (size_t)n_pcs ||
                y_t.size() < (size_t)fused_padded_pcs(n_pcs) * k_stride ||
                reinterpret_cast<uintptr_t>(y_t.data()) % 16)
                throw std::invalid_argument(
                    "clustering workspace is too small");
            // Cluster-block scratch of n_slots CTAs, then the tile counter.
            size_t n_slots =
                col_workspace && dots_workspace && col_workspace->size()
                    ? std::min(
                          dots_workspace->size() / (FUSED_GROUP * k_stride),
                          (col_workspace->size() - 1) /
                              ((FUSED_THREADS / 32) * n_clusters))
                    : 0;
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
                                ptr(joints),
                                Y_norm.data(),
                                idx_list.data(),
                                penalty.data(),
                                objective_partials.data(),
                                block_cat_offsets.data(),
                                block_starts.data(),
                                block_counts.data(),
                                recount ? 3 : n_blocks,
                                seg_start.data(),
                                (int)seg_start.size() - 1,
                                y_bound.data(),
                                y_acc.data(),
                                y_t.data(),
                                ptr(col_workspace),
                                ptr(dots_workspace),
                                (int)n_slots,
                                force_blocks,
                                n_cells,
                                n_total > 0 ? n_total : n_cells,
                                n_pcs,
                                n_clusters,
                                n_batches,
                                n_covariates,
                                n_first > 0 ? n_first : n_batches,
                                n_blocks,
                                (T)sigma,
                                (T)tol,
                                max_iter,
                                seed,
                                stabilized,
                                (cudaStream_t)stream,
                                initialize,
                                (harmony_comm::Comm*)comm,
                                rank};
            return fused_clustering_loop_impl(a);
        },
        nb::call_guard<nb::gil_scoped_release>(), "Z_norm"_a.noconvert(),
        nb::kw_only(), "R"_a.noconvert(), "E"_a.noconvert(), "O"_a.noconvert(),
        "Pr_b"_a.noconvert(), "theta"_a.noconvert(), "Y_norm"_a.noconvert(),
        "idx_list"_a.noconvert(), "penalty"_a.noconvert(),
        "objective_partials"_a.noconvert(), "block_cat_offsets"_a.noconvert(),
        "block_starts"_a.noconvert(), "block_counts"_a.noconvert(),
        "seg_start"_a.noconvert(), "y_bound"_a.noconvert(),
        "y_acc"_a.noconvert(), "R_bf16"_a.noconvert() = nb::none(),
        "joint_cats"_a.noconvert() = nb::none(),
        "joints"_a.noconvert() = nb::none(), "y_t"_a.noconvert(),
        "col_workspace"_a.noconvert() = nb::none(),
        "dots_workspace"_a.noconvert() = nb::none(), "force_blocks"_a = false,
        "n_cells"_a, "n_total"_a = 0, "n_pcs"_a, "n_clusters"_a, "n_batches"_a,
        "n_covariates"_a = 1, "n_first"_a = 0, "n_blocks"_a, "sigma"_a,
        "tol"_a = 0.0, "max_iter"_a, "seed"_a = 0, "stabilized"_a,
        "stream"_a = 0, "initialize"_a = false, "comm"_a = 0, "rank"_a = 0);
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
        "X"_a.noconvert(), nb::kw_only(), "weights"_a.noconvert(),
        "uniforms"_a.noconvert(), "centers"_a.noconvert(),
        "totals"_a.noconvert(), "n_draws"_a.noconvert(), "cluster"_a,
        "stream"_a = 0);
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
        "values"_a.noconvert(), nb::kw_only(), "categories"_a.noconvert(),
        "out"_a.noconvert(), "workspace"_a.noconvert(), "stream"_a = 0);
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
            size_t smem = kmeans_assign_smem(n_cols, sizeof(T));
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
        "X"_a.noconvert(), nb::kw_only(), "centers"_a.noconvert(),
        "labels"_a.noconvert(), "minimum"_a.noconvert(), "stream"_a = 0);
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
        "X"_a.noconvert(), nb::kw_only(), "center"_a.noconvert(),
        "closest"_a.noconvert(), "stream"_a = 0);
    m.def(
        "l2_row_normalize",
        [](gpu_array_c<const T, Device> src, gpu_array_c<T, Device> dst,
           std::uintptr_t stream) {
            l2_row_normalize(src.data(), dst.data(), (int)src.shape(0),
                             (int)src.shape(1), (cudaStream_t)stream);
        },
        "src"_a.noconvert(), nb::kw_only(), "dst"_a.noconvert(),
        "stream"_a = 0);
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
        "group_codes"_a.noconvert(), nb::kw_only(), "idx_list"_a.noconvert(),
        "block_cat_offsets"_a.noconvert(), "idx_list_alt"_a.noconvert(),
        "sort_keys"_a.noconvert(), "sort_keys_alt"_a.noconvert(),
        "cub_temp"_a.noconvert(), "n_groups"_a, "n_blocks"_a, "unit"_a,
        "seed"_a, "shuffle_chunk"_a = 1, "first"_a = 0, "stream"_a = 0);
    m.def(
        "assign_slots",
        [](int itemsize) {
            return itemsize == 8 ? assign_slots<double>()
                                 : assign_slots<float>();
        },
        nb::kw_only(), "itemsize"_a);
    m.def(
        "get_cub_sort_temp_bytes",
        [](int n_cells) { return sort_temp_bytes(n_cells, 32); }, nb::kw_only(),
        "n_cells"_a);
    m.def("kmeans_assign_smem", &kmeans_assign_smem, nb::kw_only(), "n_cols"_a,
          "itemsize"_a);
    m.def(
        "fused_padded_pcs", [](int n_pcs) { return fused_padded_pcs(n_pcs); },
        "n_pcs"_a);
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
    // Layout of the assignment workspace (see fused_assign_kernel).
    m.attr("FUSED_CLUSTER_STRIDE") = nb::int_(FUSED_CLUSTER_STRIDE);
    m.attr("FUSED_GROUP") = nb::int_(FUSED_GROUP);
    m.attr("FUSED_WARPS") = nb::int_(FUSED_THREADS / 32);
    using harmony_comm::Comm;
    nb::class_<Comm>(m, "Comm")
        .def(nb::init<const std::vector<int>&, const std::vector<uintptr_t>&,
                      size_t, bool, const std::vector<size_t>&>(),
             "devices"_a, "buffers"_a, "capacity"_a, "peer"_a, "groups"_a)
        .def_prop_ro("handle", [](Comm& c) { return (uintptr_t)&c; })
        .def("abort", [](Comm& c) { c.aborted = true; })
        .def("synchronize", &Comm::synchronize,
             nb::call_guard<nb::gil_scoped_release>());
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
