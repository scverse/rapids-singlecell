#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>

#include <cub/device/device_scan.cuh>

#include "wilcoxon_ovo.cuh"
#include "wilcoxon_ovr.cuh"

// Every sparse layout streams CSC column batches to the device and hands each
// batch to one of two rankers: OVR (sort + rank all groups) or OVO (densify
// reference and test-group rows, sort the reference, rank the groups).

constexpr int CSR_RANGE_U16_COLUMN_CAPACITY =
    (int)std::numeric_limits<uint16_t>::max() + 1;
constexpr size_t CSR_RANGE_STAGE_ITEMS = (size_t)32 * 1024 * 1024;
constexpr int CSR_RANGE_STAGE_SLOTS = 3;
constexpr int CSR_RANGE_METADATA_SLOTS = 2;
constexpr size_t CSR_RANGE_COUNT_CACHE_LIMIT_BYTES = (size_t)384 * 1024 * 1024;

// Count entries in [col_start, col_stop) for each canonical CSR row.
template <typename IndexT, typename IndptrT>
__global__ void csr_column_range_count_kernel(
    const IndexT* __restrict__ indices, const IndptrT* __restrict__ indptr,
    IndptrT* __restrict__ local_indptr, int n_rows, int col_start,
    int col_stop) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         row < n_rows; row += stride) {
        IndptrT row_start = indptr[row];
        IndptrT row_stop = indptr[row + 1];

        IndptrT lo = row_start;
        IndptrT hi = row_stop;
        while (lo < hi) {
            IndptrT mid = lo + ((hi - lo) >> 1);
            if (indices[mid] < (IndexT)col_start)
                lo = mid + 1;
            else
                hi = mid;
        }
        IndptrT range_start = lo;

        hi = row_stop;
        while (lo < hi) {
            IndptrT mid = lo + ((hi - lo) >> 1);
            if (indices[mid] < (IndexT)col_stop)
                lo = mid + 1;
            else
                hi = mid;
        }
        local_indptr[row + 1] = lo - range_start;
    }
}

// Gather a column range into compact CSR and rebase its column indices.
template <typename InT, typename IndexT, typename IndptrT>
__global__ void csr_column_range_gather_kernel(
    const InT* __restrict__ data, const IndexT* __restrict__ indices,
    const IndptrT* __restrict__ indptr,
    const IndptrT* __restrict__ local_indptr, InT* __restrict__ local_data,
    IndexT* __restrict__ local_indices, int n_rows, int col_start,
    int col_stop) {
    __shared__ IndptrT source_start;
    __shared__ IndptrT source_stop;

    for (int row = (int)blockIdx.x; row < n_rows; row += (int)gridDim.x) {
        if (threadIdx.x == 0) {
            IndptrT row_start = indptr[row];
            IndptrT row_stop = indptr[row + 1];

            IndptrT lo = row_start;
            IndptrT hi = row_stop;
            while (lo < hi) {
                IndptrT mid = lo + ((hi - lo) >> 1);
                if (indices[mid] < (IndexT)col_start)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            source_start = lo;

            hi = row_stop;
            while (lo < hi) {
                IndptrT mid = lo + ((hi - lo) >> 1);
                if (indices[mid] < (IndexT)col_stop)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            source_stop = lo;
        }
        __syncthreads();

        IndptrT destination = local_indptr[row];
        IndptrT count = source_stop - source_start;
        for (IndptrT item = (IndptrT)threadIdx.x; item < count;
             item += (IndptrT)blockDim.x) {
            IndptrT source = source_start + item;
            local_data[destination + item] = data[source];
            local_indices[destination + item] =
                indices[source] - (IndexT)col_start;
        }
        __syncthreads();
    }
}

// One CSC column batch on the device: f32 values, int32 rows, offsets rebased
// to the batch.
struct CscBatch {
    const float* vals;
    const int* rows;
    const int* offsets;  // [n_cols + 1]
    int nnz;
    int n_cols;
};

// OVR: sort each column's stored values and rank them for every group.
struct OvrBatchRanker {
    const int* group_codes;
    const double* group_sizes;
    int n_groups;
    int n_rows;
    bool tie;
    double* rank_sums;  // [group * ld + col]
    double* tie_corr;
    int ld;

    struct Scratch {
        float* keys;
        int* rows;
        uint8_t* cub;
        size_t cub_bytes;
    };
    int max_cols() const {
        return std::numeric_limits<int>::max();
    }
    static size_t cub_temp(size_t max_nnz, int sb_cols) {
        if (max_nnz == 0) return 0;
        return cub_segmented_sortpairs_temp_bytes(
            checked_cub_items(max_nnz, "OVR batch nnz"), sb_cols);
    }
    size_t bytes(size_t max_nnz, int sb_cols) const {
        return max_nnz * (sizeof(float) + sizeof(int)) +
               cub_temp(max_nnz, sb_cols);
    }
    Scratch alloc(RmmScratchPool& pool, size_t max_nnz, int sb_cols,
                  cudaStream_t) const {
        size_t cub_bytes = cub_temp(max_nnz, sb_cols);
        return {pool.alloc<float>(max_nnz), pool.alloc<int>(max_nnz),
                pool.alloc<uint8_t>(cub_bytes), cub_bytes};
    }
    void operator()(const CscBatch& b, const Scratch& s, int col,
                    cudaStream_t stream) const {
        if (b.nnz > 0)
            cub_segmented_sortpairs(s.cub, s.cub_bytes, b.vals, s.keys, b.rows,
                                    s.rows, b.nnz, b.n_cols, b.offsets,
                                    b.offsets + 1, stream,
                                    "OVR segmented sort");
        launch_ovr_rank<int>(s.keys, s.rows, b.offsets, group_codes,
                             group_sizes, rank_sums + col, tie_corr + col, ld,
                             n_rows, b.n_cols, n_groups, tie, stream);
    }
};

// CSC selected rows -> two pre-zeroed dense F-order tiles.
// Each input column is scanned once and entries are routed through both maps.
template <typename IndexT = int, typename IndptrT = int>
__global__ void csc_extract_dual_mapped_kernel(
    const float* __restrict__ data, const IndexT* __restrict__ indices,
    const IndptrT* __restrict__ indptr, const int* __restrict__ ref_row_map,
    const int* __restrict__ grp_row_map, float* __restrict__ ref_out,
    float* __restrict__ grp_out, int n_ref, int n_grp, int col_start) {
    int col_local = blockIdx.x;
    int col = col_start + col_local;

    IndptrT start = indptr[col];
    IndptrT end = indptr[col + 1];

    for (IndptrT p = start + threadIdx.x; p < end; p += blockDim.x) {
        int source_row = (int)indices[p];
        int ref_row = ref_row_map[source_row];
        int grp_row = grp_row_map[source_row];
        if (ref_row >= 0 || grp_row >= 0) {
            float value = data[p];
            if (ref_row >= 0) {
                ref_out[(long long)col_local * n_ref + ref_row] = value;
            }
            if (grp_row >= 0) {
                grp_out[(long long)col_local * n_grp + grp_row] = value;
            }
        }
    }
}

// Nonzeros per column batch that fit 80% of free memory once the ranker's
// per-column scratch (OVO dense slabs) for sb columns is reserved.
template <typename Ranker>
static size_t batch_nnz_cap(const Ranker& rank, int sb, int n_streams,
                            size_t bytes_per_nnz) {
    size_t budget = rmm_available_device_bytes(0.8);
    size_t reserve = rank.bytes(0, sb);
    budget = budget > reserve ? budget - reserve : 0;
    return std::max<size_t>(
        1, std::min(SAFE_BATCH_NNZ, budget / n_streams / bytes_per_nnz));
}

// f64 group sums / nnz (and OVR per-column totals) of the stored values (or of
// expm1(expm1_scale * x), see stat_value), added into zeroed [code * ld + col]
// outputs while casting them to f32 sort keys.
struct SparseStats {
    const int* codes;
    int n_groups;
    double* sums;
    double* nnz;
    double* total_sums;
    double* total_nnz;
    int ld;
    bool compute_nnz;
    bool totals;
    double expm1_scale;
};

// CSR rows row_ids[i] -> row i of a pre-zeroed F-order slab for the column
// window [col_start, col_start + n_cols), one thread per slab row so the slab
// writes coalesce; adds each row's group stats ([code * ld + col], zeroed).
template <typename InT, typename IndexT, typename IndptrT>
__global__ void csr_extract_rows_kernel(const InT* __restrict__ data,
                                        const IndexT* __restrict__ indices,
                                        const IndptrT* __restrict__ indptr,
                                        const int* __restrict__ row_ids,
                                        float* __restrict__ out, int n_out,
                                        int col_start, int n_cols,
                                        SparseStats st) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_out) return;
    int row = row_ids[i];
    int g = st.codes[row];
    bool stats = g >= 0 && g < st.n_groups;
    IndptrT p = indptr[row];
    IndptrT hi = indptr[row + 1];
    while (p < hi) {  // first entry >= col_start
        IndptrT m = p + ((hi - p) >> 1);
        if (indices[m] < (IndexT)col_start)
            p = m + 1;
        else
            hi = m;
    }
    for (IndptrT end = indptr[row + 1]; p < end; ++p) {
        int c = (int)(indices[p] - (IndexT)col_start);
        if (c >= n_cols) break;
        InT raw = data[p];
        out[(long long)c * n_out + i] = (float)raw;
        if (!stats) continue;
        atomicAdd(&st.sums[(long long)g * st.ld + c],
                  stat_value((double)raw, st.expm1_scale));
        if (st.compute_nnz && raw != 0)
            atomicAdd(&st.nnz[(long long)g * st.ld + c], 1.0);
    }
}

// OVO: densify the batch's reference and test-group rows (row maps give each
// row's slab position or -1), sort the reference and rank the groups.
struct OvoBatchRanker {
    const int* ref_map;  // row -> slab position or -1 (CSC batches)
    const int* grp_map;
    const int* ref_rows;  // slab position -> row (CSR batches)
    const int* grp_rows;
    int n_ref;
    int n_all_grp;
    const int* grp_offsets;
    const OvoRanker* ranker;
    double* rank_sums;  // [group * ld + col]
    double* tie_corr;
    int ld;

    struct Scratch {
        float* ref;
        float* ref_sorted;
        float* grp;
        int* ref_offsets;
        uint8_t* cub;
        size_t cub_bytes;
        OvoScratch ovo;
    };
    size_t cub_temp(int sb_cols) const {
        return cub_segmented_sortkeys_temp_bytes(
            checked_cub_items((size_t)n_ref * sb_cols, "OVO reference batch"),
            sb_cols);
    }
    size_t bytes(size_t, int sb_cols) const {
        size_t grp_items = (size_t)n_all_grp * sb_cols;
        return ((size_t)2 * n_ref * sb_cols + grp_items) * sizeof(float) +
               cub_temp(sb_cols) + ranker->scratch_bytes(grp_items, sb_cols);
    }
    // Dense slabs bound the batch width: int32 items and half the free memory.
    int max_cols() const {
        size_t cols = SAFE_BATCH_NNZ / (size_t)std::max(n_ref, n_all_grp);
        cols = std::min(cols, rmm_available_device_bytes(0.5) / bytes(0, 1));
        return (int)std::max<size_t>(1, std::min<size_t>(cols, INT_MAX));
    }
    Scratch alloc(RmmScratchPool& pool, size_t, int sb_cols,
                  cudaStream_t stream) const {
        Scratch s;
        s.ref = pool.alloc<float>((size_t)n_ref * sb_cols);
        s.ref_sorted = pool.alloc<float>((size_t)n_ref * sb_cols);
        s.grp = pool.alloc<float>((size_t)n_all_grp * sb_cols);
        s.ref_offsets = pool.alloc<int>(sb_cols + 1);
        upload_linear_offsets(s.ref_offsets, sb_cols, n_ref, stream);
        s.cub_bytes = cub_temp(sb_cols);
        s.cub = pool.alloc<uint8_t>(s.cub_bytes);
        s.ovo = ranker->alloc(pool, (size_t)n_all_grp * sb_cols, sb_cols);
        return s;
    }
    void zero_slabs(const Scratch& s, int n_cols, cudaStream_t stream) const {
        cudaMemsetAsync(s.ref, 0, (size_t)n_ref * n_cols * sizeof(float),
                        stream);
        cudaMemsetAsync(s.grp, 0, (size_t)n_all_grp * n_cols * sizeof(float),
                        stream);
    }
    void rank_slabs(const Scratch& s, int col, int n_cols,
                    cudaStream_t stream) const {
        cub_segmented_sortkeys(s.cub, s.cub_bytes, s.ref, s.ref_sorted,
                               n_ref * n_cols, n_cols, s.ref_offsets,
                               s.ref_offsets + 1, stream, "OVO reference sort");
        ranker->rank(s.ref_sorted, s.grp, grp_offsets, s.ovo, rank_sums + col,
                     tie_corr + col, ld, n_ref, n_all_grp, n_cols, stream);
    }
    void operator()(const CscBatch& b, const Scratch& s, int col,
                    cudaStream_t stream) const {
        zero_slabs(s, b.n_cols, stream);
        csc_extract_dual_mapped_kernel<<<b.n_cols, UTIL_BLOCK_SIZE, 0,
                                         stream>>>(b.vals, b.rows, b.offsets,
                                                   ref_map, grp_map, s.ref,
                                                   s.grp, n_ref, n_all_grp, 0);
        CUDA_CHECK_LAST_ERROR(csc_extract_dual_mapped_kernel);
        rank_slabs(s, col, b.n_cols, stream);
    }
    // CSR rows (device input or a gathered host batch): densify directly
    // without a transpose, stats included; rank_slabs finishes the batch.
    template <typename InT, typename IndexT, typename IndptrT>
    void extract_csr(const InT* data, const IndexT* indices,
                     const IndptrT* indptr, int col, int n_cols,
                     const SparseStats& st, const Scratch& s,
                     cudaStream_t stream) const {
        zero_slabs(s, n_cols, stream);
        SparseStats batch_st = st;
        batch_st.sums += col;
        batch_st.nnz += col;
        auto launch = [&](const int* rows, float* out, int n_out) {
            csr_extract_rows_kernel<<<(n_out + UTIL_BLOCK_SIZE - 1) /
                                          UTIL_BLOCK_SIZE,
                                      UTIL_BLOCK_SIZE, 0, stream>>>(
                data, indices, indptr, rows, out, n_out, col, n_cols, batch_st);
            CUDA_CHECK_LAST_ERROR(csr_extract_rows_kernel);
        };
        launch(ref_rows, s.ref, n_ref);
        launch(grp_rows, s.grp, n_all_grp);
    }
};

// Raw batch values for cast_with_stats: f32 input casts in place, so its raw
// buffer is the f32 key buffer itself.
template <typename InT>
constexpr size_t RAW_BYTES = std::is_same_v<InT, float> ? 0 : sizeof(InT);

template <typename InT>
static InT* raw_values(RmmScratchPool& pool, float* vals, size_t n) {
    if constexpr (std::is_same_v<InT, float>)
        return vals;
    else
        return pool.alloc<InT>(n);
}

template <typename InT>
static void cast_with_stats(const InT* data, float* vals, const int* rows,
                            const int* offsets, int sb_cols, int col,
                            const SparseStats& st, cudaStream_t stream) {
    launch_ovr_cast_and_accumulate<InT, int>(
        data, vals, rows, offsets, st.codes, st.sums + col, st.nnz + col,
        st.total_sums + col, st.total_nnz + col, st.ld, sb_cols, st.n_groups,
        st.compute_nnz, st.totals, st.expm1_scale, stream);
}

// CSC input: host batches stream through pinned memory (whole arrays pinned
// when moderate and pinnable, else a bounded ring that also narrows row ids
// to int32); device batches are read in place. Stored values are cast to
// f32 keys while their group stats accumulate.
template <typename InT, typename IndexT, typename Ranker>
static void sparse_csc(const InT* data, const IndexT* indices,
                       const IndexT* indptr, bool on_host, int n_rows,
                       int n_cols, int sub_batch_cols, const Ranker& rank,
                       const SparseStats& stats) {
    if (n_rows == 0 || n_cols == 0) return;
    std::vector<IndexT> indptr_copy;
    const IndexT* h_indptr = indptr;
    if (!on_host) {
        indptr_copy.resize(n_cols + 1);
        cuda_check(
            cudaMemcpy(indptr_copy.data(), indptr,
                       (n_cols + 1) * sizeof(IndexT), cudaMemcpyDeviceToHost),
            "CSC indptr D2H");
        h_indptr = indptr_copy.data();
    }
    constexpr size_t BYTES_PER_NNZ =
        RAW_BYTES<InT> + sizeof(IndexT) + 3 * sizeof(float) + 2 * sizeof(int);
    int sb = std::min(sub_batch_cols, rank.max_cols());
    ColumnBatchPlan batches = plan_csc_column_batches(
        h_indptr, n_cols, sb, batch_nnz_cap(rank, sb, N_STREAMS, BYTES_PER_NNZ),
        "CSC column batch nnz");
    sb = batches.sub_batch_cols;
    size_t max_nnz = batches.max_nnz;
    size_t per_stream = max_nnz * BYTES_PER_NNZ + rank.bytes(max_nnz, sb);
    int n_streams =
        clamp_streams_by_budget(clamp_streams_by_cols(n_cols, sb), per_stream,
                                rmm_available_device_bytes(0.8));

    // pool first: streams drain before it frees their scratch (see guard doc).
    RmmScratchPool pool;
    size_t total_nnz = (size_t)h_indptr[n_cols];
    bool bounded = on_host && total_nnz * (sizeof(InT) + sizeof(IndexT)) >
                                  HOST_STREAMING_DIRECT_PIN_LIMIT_BYTES;
    HostRegisterGuard pin_data, pin_indices;
    std::unique_ptr<PinnedRing<InT, int>> stage;
    if (bounded) {
        stage.reset(new PinnedRing<InT, int>(n_streams, max_nnz));
    } else if (on_host) {
        // Pinning only speeds the copies up (read-only maps cannot be pinned).
        pin_data = HostRegisterGuard(const_cast<InT*>(data),
                                     total_nnz * sizeof(InT), 0, true);
        pin_indices = HostRegisterGuard(const_cast<IndexT*>(indices),
                                        total_nnz * sizeof(IndexT), 0, true);
    }
    ScopedCudaStreams streams(n_streams, cudaStreamDefault);
    int* d_offsets = upload_batch_offsets(batches, pool);
    struct StreamBuf {
        InT* data;
        IndexT* indices;
        float* vals;
        int* rows;
        typename Ranker::Scratch rank;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].vals = pool.alloc<float>(max_nnz);
        bufs[s].data =
            on_host ? raw_values<InT>(pool, bufs[s].vals, max_nnz) : nullptr;
        bufs[s].indices =
            on_host && !bounded ? pool.alloc<IndexT>(max_nnz) : nullptr;
        bufs[s].rows = bounded || sizeof(IndexT) > sizeof(int)
                           ? pool.alloc<int>(max_nnz)
                           : nullptr;
        bufs[s].rank = rank.alloc(pool, max_nnz, sb, streams[s]);
    }
    cudaDeviceSynchronize();

    for (int col = 0, b = 0; col < n_cols; col += sb, ++b) {
        int sb_cols = std::min(sb, n_cols - col);
        int s = b % n_streams;
        cudaStream_t stream = streams[s];
        auto& buf = bufs[s];
        IndexT p0 = h_indptr[col];
        int nnz = checked_int_span((size_t)(h_indptr[col + sb_cols] - p0),
                                   "CSC batch nnz");
        const InT* d = data + p0;
        const IndexT* ix = indices + p0;
        const int* rows = buf.rows;
        if (bounded) {
            stage->wait(s);
            host_copy_slice_as<InT, int>(data, indices, (size_t)p0, nnz,
                                         stage->template get<0>(s),
                                         stage->template get<1>(s));
            cudaMemcpyAsync(buf.data, stage->template get<0>(s),
                            (size_t)nnz * sizeof(InT), cudaMemcpyHostToDevice,
                            stream);
            cudaMemcpyAsync(buf.rows, stage->template get<1>(s),
                            (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice,
                            stream);
            stage->record(s, stream);
            d = buf.data;
        } else {
            if (on_host) {
                cudaMemcpyAsync(buf.data, d, (size_t)nnz * sizeof(InT),
                                cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(buf.indices, ix, (size_t)nnz * sizeof(IndexT),
                                cudaMemcpyHostToDevice, stream);
                d = buf.data;
                ix = buf.indices;
            }
            if constexpr (sizeof(IndexT) > sizeof(int)) {
                if (nnz > 0) {
                    cast_array_kernel<IndexT, int>
                        <<<(nnz + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE,
                           UTIL_BLOCK_SIZE, 0, stream>>>(ix, buf.rows,
                                                         (size_t)nnz);
                    CUDA_CHECK_LAST_ERROR(cast_array_kernel);
                }
            } else {
                rows = reinterpret_cast<const int*>(ix);
            }
        }
        const int* offsets = d_offsets + (size_t)b * (sb + 1);
        cast_with_stats(d, buf.vals, rows, offsets, sb_cols, col, stats,
                        stream);
        rank(CscBatch{buf.vals, rows, offsets, nnz, sb_cols}, buf.rank, col,
             stream);
    }
    sync_streams(streams, "sparse CSC");
}

// Device CSR OVO: column batches bounded by the dense slabs, extracted from
// the CSR rows directly.
template <typename InT, typename IndexT>
static void ovo_csr_device(const InT* data, const IndexT* indices,
                           const IndexT* indptr, int n_rows, int n_cols,
                           int sub_batch_cols, const OvoBatchRanker& rank,
                           const SparseStats& stats) {
    if (n_rows == 0 || n_cols == 0) return;
    int sb = std::min(sub_batch_cols, rank.max_cols());
    int n_streams = clamp_streams_by_budget(clamp_streams_by_cols(n_cols, sb),
                                            rank.bytes(0, sb),
                                            rmm_available_device_bytes(0.8));
    RmmScratchPool pool;
    ScopedCudaStreams streams(n_streams, cudaStreamDefault);
    std::vector<OvoBatchRanker::Scratch> scratch;
    for (int s = 0; s < n_streams; s++)
        scratch.push_back(rank.alloc(pool, 0, sb, streams[s]));
    cudaDeviceSynchronize();
    for (int col = 0, b = 0; col < n_cols; col += sb, ++b) {
        int sb_cols = std::min(sb, n_cols - col);
        cudaStream_t stream = streams[b % n_streams];
        rank.extract_csr(data, indices, indptr, col, sb_cols, stats,
                         scratch[b % n_streams], stream);
        rank.rank_slabs(scratch[b % n_streams], col, sb_cols, stream);
    }
    sync_streams(streams, "OVO device CSR");
}

// Device CSR input: a column histogram plans the batches, each of which is
// transposed into a CSC batch by an atomic scatter.
template <typename InT, typename IndexT, typename Ranker>
static void sparse_csr_device(const InT* data, const IndexT* indices,
                              const IndexT* indptr, int n_rows, int n_cols,
                              int sub_batch_cols, const Ranker& rank,
                              const SparseStats& stats) {
    if (n_rows == 0 || n_cols == 0) return;
    RmmScratchPool pool;
    std::vector<unsigned int> col_counts(n_cols);
    {
        unsigned int* d_counts = pool.alloc<unsigned int>(n_cols);
        cudaMemset(d_counts, 0, n_cols * sizeof(unsigned int));
        csr_col_histogram_kernel<<<(n_rows + UTIL_BLOCK_SIZE - 1) /
                                       UTIL_BLOCK_SIZE,
                                   UTIL_BLOCK_SIZE>>>(indices, indptr, d_counts,
                                                      n_rows, n_cols);
        CUDA_CHECK_LAST_ERROR(csr_col_histogram_kernel);
        cuda_check(
            cudaMemcpy(col_counts.data(), d_counts,
                       n_cols * sizeof(unsigned int), cudaMemcpyDeviceToHost),
            "CSR column histogram D2H");
    }
    constexpr size_t BYTES_PER_NNZ =
        RAW_BYTES<InT> + 3 * sizeof(float) + 2 * sizeof(int) + 8;
    int sb = std::min(sub_batch_cols, rank.max_cols());
    ColumnBatchPlan batches = plan_column_batches_from_counts(
        n_cols, sb, batch_nnz_cap(rank, sb, N_STREAMS, BYTES_PER_NNZ),
        [&](int c) { return (size_t)col_counts[c]; }, "CSR column batch nnz");
    sb = batches.sub_batch_cols;
    size_t max_nnz = batches.max_nnz;
    size_t per_stream = max_nnz * BYTES_PER_NNZ + rank.bytes(max_nnz, sb);
    int n_streams =
        clamp_streams_by_budget(clamp_streams_by_cols(n_cols, sb), per_stream,
                                rmm_available_device_bytes(0.8));
    ScopedCudaStreams streams(n_streams, cudaStreamDefault);
    int* d_offsets = upload_batch_offsets(batches, pool);
    struct StreamBuf {
        int* write_pos;
        InT* data;
        float* vals;
        int* rows;
        typename Ranker::Scratch rank;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].write_pos = pool.alloc<int>(sb);
        bufs[s].vals = pool.alloc<float>(max_nnz);
        bufs[s].data = raw_values<InT>(pool, bufs[s].vals, max_nnz);
        bufs[s].rows = pool.alloc<int>(max_nnz);
        bufs[s].rank = rank.alloc(pool, max_nnz, sb, streams[s]);
    }
    cudaDeviceSynchronize();

    for (int col = 0, b = 0; col < n_cols; col += sb, ++b) {
        int sb_cols = std::min(sb, n_cols - col);
        cudaStream_t stream = streams[b % n_streams];
        auto& buf = bufs[b % n_streams];
        const int* offsets = d_offsets + (size_t)b * (sb + 1);
        int nnz = (int)batches.nnz[b];
        if (nnz > 0) {
            cudaMemcpyAsync(buf.write_pos, offsets, sb_cols * sizeof(int),
                            cudaMemcpyDeviceToDevice, stream);
            csr_scatter_to_csc_kernel<<<(n_rows + UTIL_BLOCK_SIZE - 1) /
                                            UTIL_BLOCK_SIZE,
                                        UTIL_BLOCK_SIZE, 0, stream>>>(
                data, indices, indptr, buf.write_pos, buf.data, buf.rows,
                n_rows, col, col + sb_cols);
            CUDA_CHECK_LAST_ERROR(csr_scatter_to_csc_kernel);
        }
        cast_with_stats(buf.data, buf.vals, buf.rows, offsets, sb_cols, col,
                        stats, stream);
        rank(CscBatch{buf.vals, buf.rows, offsets, nnz, sb_cols}, buf.rank, col,
             stream);
    }
    sync_streams(streams, "sparse device CSR");
}

// lower_bound on one row's sorted column ids: gallop out from an interpolated
// guess, which touches fewer cache lines than bisection on near-uniform rows.
template <typename T>
static const T* interp_lower_bound(const T* lo, const T* hi, T v) {
    if (lo == hi || !(*lo < v)) return lo;
    size_t n = (size_t)(hi - lo);
    if (lo[n - 1] < v) return hi;
    size_t g = 1 + (size_t)((double)(v - lo[0]) * (double)(n - 1) /
                            ((double)(lo[n - 1] - lo[0]) + 1.0));
    g = std::min(g, n - 1);
    size_t a, b, step = 1;  // lo[a] < v <= lo[b]
    if (lo[g] < v) {
        for (a = g; a + step < n - 1 && lo[a + step] < v; step <<= 1) a += step;
        b = std::min(a + step, n - 1);
    } else {
        for (b = g; step < b && !(lo[b - step] < v); step <<= 1) b -= step;
        a = step < b ? b - step : 0;
    }
    return std::lower_bound(lo + a + 1, lo + b + 1, v);
}

// Below this many stored entries one thread scans faster than the pool wakes.
constexpr int64_t CANONICAL_SCAN_SERIAL_NNZ = 1 << 22;

// scipy's has_canonical_format for CSR/CSC (indptr non-decreasing, indices
// strictly increasing within each row), but scanned by the host workers over
// nnz-balanced row ranges instead of on one thread.
template <typename IndexT>
static bool has_canonical_format_host(const IndexT* indptr,
                                      const IndexT* indices, int64_t n_rows) {
    for (int64_t r = 0; r < n_rows; r++)
        if (indptr[r] > indptr[r + 1]) return false;
    int64_t nnz = n_rows > 0 ? (int64_t)indptr[n_rows] : 0;
    int n_threads = nnz < CANONICAL_SCAN_SERIAL_NNZ ? 1 : host_worker_count();
    auto first_row = [&](int t) {  // first row starting in worker t's share
        if (t == n_threads) return n_rows;
        IndexT p = (IndexT)(nnz * t / n_threads);
        return (int64_t)(std::lower_bound(indptr, indptr + n_rows, p) - indptr);
    };
    std::atomic<bool> ok{true};
    auto scan = [&](int, int t0, int t1) {
        for (int64_t r = first_row(t0), stop = first_row(t1);
             r < stop && ok.load(std::memory_order_relaxed); r++) {
            bool sorted = true;
            for (IndexT p = indptr[r] + 1; p < indptr[r + 1]; p++)
                sorted &= indices[p - 1] < indices[p];
            if (!sorted) ok = false;
        }
    };
    if (n_threads == 1)
        scan(0, 0, 1);
    else
        host_worker_pool(n_threads).run(n_threads, scan);
    return ok;
}

// Plan CSR range batches without a full per-column histogram. Sorted row spans
// let each shard find batch boundaries with lower_bound, so planning touches
// O(n_rows * n_batches) indices instead of every source nonzero.
struct CsrRangeBatchPlan {
    ColumnBatchPlan columns;
    // Batch-major. Empty means the bounded-cache fallback is active.
    std::vector<int> row_counts;
};

template <typename IndexT, typename IndptrT>
static CsrRangeBatchPlan plan_csr_range_batches(
    const IndexT* h_indices, const IndptrT* h_indptr, const IndptrT* row_starts,
    const IndptrT* row_stops, int n_rows, int input_col_start, int n_cols,
    int sub_batch_cols, size_t cap, const char* what) {
    if (sub_batch_cols < 1) sub_batch_cols = 1;
    if (cap < 1) cap = 1;

    size_t range_nnz = 0;
    for (int r = 0; r < n_rows; r++) {
        if (row_starts[r] < h_indptr[r] || row_stops[r] < row_starts[r] ||
            row_stops[r] > h_indptr[r + 1]) {
            throw std::runtime_error(std::string(what) +
                                     " has an invalid row span");
        }
        size_t row_nnz = (size_t)(row_stops[r] - row_starts[r]);
        if (row_nnz > std::numeric_limits<size_t>::max() - range_nnz) {
            throw std::runtime_error(std::string(what) +
                                     " total nnz overflows size_t");
        }
        range_nnz += row_nnz;
    }
    size_t min_batches = range_nnz == 0 ? 0 : 1 + (range_nnz - 1) / cap;
    bool count_cache_allocation_failed = false;

    for (;;) {
        CsrRangeBatchPlan range_plan;
        ColumnBatchPlan& plan = range_plan.columns;
        plan.sub_batch_cols = sub_batch_cols;
        plan.n_batches = 1 + (n_cols - 1) / plan.sub_batch_cols;
        plan.nnz.assign(plan.n_batches, 0);

        // Skip widths that cannot fit even under a perfectly even nnz split.
        // The accepted width still goes through the exact per-row validator.
        if (plan.sub_batch_cols > 1 && (size_t)plan.n_batches < min_batches) {
            sub_batch_cols = std::max(1, sub_batch_cols / 2);
            continue;
        }

        int n_workers = host_worker_count();
        bool cache_counts = !count_cache_allocation_failed;
        constexpr size_t max_count_cache_items =
            CSR_RANGE_COUNT_CACHE_LIMIT_BYTES / sizeof(int);
        if ((size_t)plan.n_batches > max_count_cache_items / (size_t)n_rows) {
            cache_counts = false;
        }
        if (cache_counts) {
            try {
                range_plan.row_counts.resize((size_t)plan.n_batches *
                                             (size_t)n_rows);
            } catch (const std::bad_alloc&) {
                range_plan.row_counts.clear();
                count_cache_allocation_failed = true;
                cache_counts = false;
            }
        }

        std::vector<std::vector<size_t>> local_counts(
            n_workers, std::vector<size_t>(plan.n_batches, 0));
        std::vector<unsigned char> invalid_cached_count(n_workers, 0);
        int used = host_parallel_chunks(n_rows, [&](int w, int r0, int r1) {
            std::vector<size_t>& counts = local_counts[w];
            for (int r = r0; r < r1; r++) {
                const IndexT* lo = h_indices + row_starts[r];
                const IndexT* hi = h_indices + row_stops[r];
                if (lo < hi && *lo < (IndexT)input_col_start) {
                    lo = std::lower_bound(lo, hi, (IndexT)input_col_start);
                }
                for (int b = 0; b < plan.n_batches; b++) {
                    int local_start = b * plan.sub_batch_cols;
                    int width =
                        std::min(plan.sub_batch_cols, n_cols - local_start);
                    IndexT input_col_end =
                        (IndexT)(input_col_start + local_start + width);
                    const IndexT* next =
                        interp_lower_bound(lo, hi, input_col_end);
                    size_t row_count = (size_t)(next - lo);
                    counts[b] += row_count;
                    if (cache_counts) {
                        if (row_count <=
                            (size_t)std::numeric_limits<int>::max()) {
                            range_plan.row_counts[(size_t)b * (size_t)n_rows +
                                                  (size_t)r] = (int)row_count;
                        } else {
                            invalid_cached_count[w] = 1;
                        }
                    }
                    lo = next;
                }
            }
        });

        bool fits = true;
        for (int b = 0; b < plan.n_batches; b++) {
            size_t batch_nnz = 0;
            for (int w = 0; w < used; w++) batch_nnz += local_counts[w][b];
            plan.nnz[b] = batch_nnz;
            plan.max_nnz = std::max(plan.max_nnz, batch_nnz);
            if (batch_nnz > cap) fits = false;
        }
        if (cache_counts &&
            std::any_of(invalid_cached_count.begin(),
                        invalid_cached_count.begin() + used,
                        [](unsigned char invalid) { return invalid != 0; })) {
            range_plan.row_counts.clear();
        }

        if (fits || plan.sub_batch_cols == 1) {
            for (size_t batch_nnz : plan.nnz) checked_int_span(batch_nnz, what);
            return range_plan;
        }
        sub_batch_cols = std::max(1, sub_batch_cols / 2);
    }
}

// Count one range batch before materializing it so the compact row offsets can
// be copied once while values/indices are transferred in bounded row blocks.
template <typename IndexT, typename IndptrT>
static int prepare_csr_range_batch(const IndexT* h_indices,
                                   const IndptrT* h_indptr,
                                   const IndptrT* row_stops, int n_rows,
                                   int col_start, int col_end, IndptrT* cursor,
                                   int* fallback_row_counts,
                                   const int* planned_row_counts,
                                   int* compact_indptr, const char* what) {
    const int* row_counts = planned_row_counts;
    if (row_counts == nullptr) {
        host_parallel_ranges(n_rows, [&](int r0, int r1) {
            for (int r = r0; r < r1; r++) {
                const IndexT* row_base = h_indices + h_indptr[r];
                const IndexT* lo = row_base + cursor[r];
                const IndexT* hi = h_indices + row_stops[r];
                if (lo < hi && *lo < (IndexT)col_start) {
                    lo = std::lower_bound(lo, hi, (IndexT)col_start);
                    cursor[r] = (IndptrT)(lo - row_base);
                }
                fallback_row_counts[r] =
                    (int)(interp_lower_bound(lo, hi, (IndexT)col_end) - lo);
            }
        });
        row_counts = fallback_row_counts;
    }

    compact_indptr[0] = 0;
    for (int r = 0; r < n_rows; r++) {
        compact_indptr[r + 1] = checked_int_span(
            (size_t)compact_indptr[r] + (size_t)row_counts[r], what);
    }
    return compact_indptr[n_rows];
}

// Materialize a row-aligned slice of a prepared range batch. Column counts are
// derived from the compacted device indices so this host pass only copies and
// rebases each item.
template <typename StageIndexT, typename InT, typename IndexT, typename IndptrT>
static void materialize_csr_range_row_block(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    int row_begin, int row_end, int stage_col_offset, IndptrT* cursor,
    const int* row_counts, const int* compact_indptr, int block_nnz_begin,
    InT* stage_vals, StageIndexT* stage_cols) {
    auto materialize_rows = [&](int r0, int r1) {
        for (int r = r0; r < r1; r++) {
            IndptrT base = h_indptr[r] + cursor[r];
            size_t dst = (size_t)(compact_indptr[r] - block_nnz_begin);
            int count = row_counts[r];
            std::memcpy(stage_vals + dst, h_data + base,
                        (size_t)count * sizeof(InT));
            const IndexT* __restrict__ source_cols = h_indices + base;
            StageIndexT* __restrict__ dest_cols = stage_cols + dst;
            for (int k = 0; k < count; k++) {
                dest_cols[k] =
                    (StageIndexT)(source_cols[k] - (IndexT)stage_col_offset);
            }
            cursor[r] += count;
        }
    };

    int block_rows = row_end - row_begin;
    int n_workers = host_worker_count();
    if (n_workers <= 1 || block_rows < 4096) {
        materialize_rows(row_begin, row_end);
        return;
    }

    // Equal-row chunks leave stragglers when cell nnz varies. The compact CSR
    // offsets provide an exact nnz prefix; adding one work unit per row keeps
    // the prefix strictly increasing across empty rows. Snap equal-work targets
    // to row boundaries so every worker's writes remain disjoint.
    int block_nnz_end = compact_indptr[row_end];
    size_t block_nnz = (size_t)(block_nnz_end - block_nnz_begin);
    size_t total_work = block_nnz + (size_t)block_rows;
    std::vector<int> worker_row_bounds;
    worker_row_bounds.reserve((size_t)n_workers + 1);
    worker_row_bounds.push_back(row_begin);
    for (int w = 1; w < n_workers; w++) {
        size_t target =
            total_work / (size_t)n_workers * (size_t)w +
            total_work % (size_t)n_workers * (size_t)w / (size_t)n_workers;
        int lo = row_begin;
        int hi = row_end;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            size_t completed = (size_t)(compact_indptr[mid] - block_nnz_begin) +
                               (size_t)(mid - row_begin);
            if (completed < target)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo > worker_row_bounds.back() && lo < row_end)
            worker_row_bounds.push_back(lo);
    }
    worker_row_bounds.push_back(row_end);
    int n_partitions = (int)worker_row_bounds.size() - 1;
    host_worker_pool(n_workers).run(n_partitions, [&](int, int p0, int p1) {
        for (int p = p0; p < p1; p++)
            materialize_rows(worker_row_bounds[p], worker_row_bounds[p + 1]);
    });
}

// Host CSR input over the column range [col_start, col_start + n_cols) of
// sorted row spans: cursors advance once, so each stored value is gathered on
// the host and transferred once as a mini-CSR column batch.
template <typename StageIndexT, typename InT, typename IndexT, typename Ranker>
static void sparse_csr_host_typed(const InT* h_data, const IndexT* h_indices,
                                  const IndexT* h_indptr,
                                  const IndexT* row_starts,
                                  const IndexT* row_stops, int input_col_start,
                                  int n_rows, int n_cols, int sub_batch_cols,
                                  const Ranker& rank,
                                  const SparseStats& stats) {
    if (n_rows == 0 || n_cols == 0) return;
    RmmScratchPool pool;
    int tpb = UTIL_BLOCK_SIZE;

    // OVO ranks the gathered rows directly, in OVO_SLAB_COLS-wide chunks of a
    // wide batch (fewer host passes over all rows), from two mini-CSR buffers
    // so the next batch uploads while this one is ranked. OVR frees its single
    // buffer early by transposing the batch to CSC. Column batches must
    // satisfy int32 CUB limits and the device budget: per nnz, the gather
    // buffers plus, for OVR, the CSC accumulator, sort output and CUB.
    constexpr bool ovo = std::is_same_v<Ranker, OvoBatchRanker>;
    constexpr int n_gather = ovo ? 2 : 1;
    constexpr size_t GATHER_BYTES =
        n_gather * (sizeof(InT) + sizeof(StageIndexT));
    constexpr size_t BYTES_PER_NNZ = ovo ? GATHER_BYTES
                                         : GATHER_BYTES + RAW_BYTES<InT> +
                                               2 * sizeof(float) +
                                               2 * sizeof(int) + 12;
    // The column histogram keeps one int per batch column in shared memory.
    sub_batch_cols =
        std::min({sub_batch_cols, n_cols,
                  device_attribute(cudaDevAttrMaxSharedMemoryPerBlock) /
                      (int)sizeof(int)});
    int slab_cols = std::min(sub_batch_cols, rank.max_cols());
    if constexpr (ovo) slab_cols = std::min(slab_cols, OVO_SLAB_COLS);
    CsrRangeBatchPlan range_batches =
        plan_csr_range_batches(h_indices, h_indptr, row_starts, row_stops,
                               n_rows, input_col_start, n_cols, sub_batch_cols,
                               batch_nnz_cap(rank, slab_cols, 1, BYTES_PER_NNZ),
                               "rowstream range batch nnz");
    ColumnBatchPlan batches = std::move(range_batches.columns);

    sub_batch_cols = batches.sub_batch_cols;
    int n_batches = batches.n_batches;
    size_t max_batch_nnz = batches.max_nnz;

    size_t scan_temp_bytes = 0;
    {
        auto* histogram = reinterpret_cast<int*>(1);
        cuda_check(
            cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes, histogram,
                                          histogram, sub_batch_cols + 1),
            "rowstream range scan temp-size query");
    }
    // Host gather staging is pinned; full CSR stays pageable on CPU.
    // Only the compacted column interval crosses the bus.
    size_t stage_nnz = max_batch_nnz ? max_batch_nnz : 1;
    size_t range_stage_target =
        std::max(CSR_RANGE_STAGE_ITEMS, (size_t)sub_batch_cols);
    size_t gather_stage_nnz = std::min(stage_nnz, range_stage_target);
    PinnedRing<InT, StageIndexT> gather_stage(CSR_RANGE_STAGE_SLOTS,
                                              gather_stage_nnz, true);
    PinnedRing<int> indptr_stage(CSR_RANGE_METADATA_SLOTS, (size_t)n_rows + 1);
    std::vector<IndexT> cursor(n_rows, 0);  // offset within each (sorted) row
    host_parallel_ranges(n_rows, [&](int r0, int r1) {
        for (int r = r0; r < r1; r++) {
            const IndexT* row_base = h_indices + h_indptr[r];
            const IndexT* lo = h_indices + row_starts[r];
            const IndexT* hi = h_indices + row_stops[r];
            if (lo < hi && *lo < (IndexT)input_col_start) {
                lo = std::lower_bound(lo, hi, (IndexT)input_col_start);
            }
            cursor[r] = (IndexT)(lo - row_base);
        }
    });

    InT* d_gather_vals[n_gather];
    StageIndexT* d_gather_cols[n_gather];
    int* d_gather_indptr[n_gather];
    for (int i = 0; i < n_gather; i++) {
        d_gather_vals[i] = pool.alloc<InT>(max_batch_nnz);
        d_gather_cols[i] = pool.alloc<StageIndexT>(max_batch_nnz);
        d_gather_indptr[i] = pool.alloc<int>(n_rows + 1);
    }
    slab_cols = std::min(slab_cols, sub_batch_cols);
    size_t csc_nnz = ovo ? 0 : max_batch_nnz;
    int* d_col_offsets[CSR_RANGE_METADATA_SLOTS] = {
        pool.alloc<int>(sub_batch_cols + 1),
        pool.alloc<int>(sub_batch_cols + 1)};
    int* write_pos = pool.alloc<int>(sub_batch_cols);
    float* csc_vals_f32 = pool.alloc<float>(csc_nnz);
    InT* csc_vals_orig = raw_values<InT>(pool, csc_vals_f32, csc_nnz);
    int* csc_row_idx = pool.alloc<int>(csc_nnz);
    uint8_t* scan_temp = pool.alloc<uint8_t>(scan_temp_bytes);
    std::vector<std::unique_ptr<ScopedCudaEvent>> batch_ready;
    std::vector<std::unique_ptr<ScopedCudaEvent>> gather_consumed;
    ScopedCudaStream row_stream(cudaStreamDefault);
    cudaStream_t stream = row_stream.get();
    ScopedCudaStream range_copy_stream(cudaStreamNonBlocking);
    cudaStream_t copy_stream = range_copy_stream.get();
    auto rank_scratch = rank.alloc(pool, max_batch_nnz, slab_cols, stream);
    batch_ready.reserve(n_batches);
    gather_consumed.reserve(n_batches);
    for (int b = 0; b < n_batches; b++) {
        batch_ready.emplace_back(
            std::make_unique<ScopedCudaEvent>(cudaEventDisableTiming));
        gather_consumed.emplace_back(
            std::make_unique<ScopedCudaEvent>(cudaEventDisableTiming));
    }
    int range_histogram_blocks =
        std::max(1, 4 * device_attribute(cudaDevAttrMultiProcessorCount));
    // Codes, sizes and the zeroed outputs come from the caller's stream.
    cudaDeviceSynchronize();

    // One ascending column pass; sorted-row cursors make transfer one-shot.
    // Threaded gather counts row runs, prefix-sums, then copies disjoint
    // ranges. The GPU derives CSC column offsets from the staged column ids.
    std::vector<int> g_count;
    if (range_batches.row_counts.empty()) g_count.resize(n_rows);
    size_t range_stage_sequence = 0;

    int col = 0;
    for (int b = 0; b < n_batches; b++) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int input_col = input_col_start + col;
        int input_col_end = input_col + sb_cols;
        int metadata_slot = b % CSR_RANGE_METADATA_SLOTS;
        int* active_col_offsets = d_col_offsets[metadata_slot];
        InT* gather_vals = d_gather_vals[b % n_gather];
        StageIndexT* gather_cols = d_gather_cols[b % n_gather];
        int* gather_indptr = d_gather_indptr[b % n_gather];
        if (b >= n_gather) {
            cuda_check(
                cudaStreamWaitEvent(copy_stream,
                                    gather_consumed[b - n_gather]->get(), 0),
                "rowstream copy wait for gather consumption");
        }
        indptr_stage.wait(metadata_slot);
        int* h_gather_indptr = indptr_stage.template get<0>(metadata_slot);
        const int* planned_row_counts =
            range_batches.row_counts.empty()
                ? nullptr
                : range_batches.row_counts.data() + (size_t)b * (size_t)n_rows;
        int* fallback_row_counts = g_count.empty() ? nullptr : g_count.data();

        int batch_nnz = prepare_csr_range_batch(
            h_indices, h_indptr, row_stops, n_rows, input_col, input_col_end,
            cursor.data(), fallback_row_counts, planned_row_counts,
            h_gather_indptr, "rowstream range gather nnz");
        if ((size_t)batch_nnz != batches.nnz[b]) {
            throw std::runtime_error(
                "rowstream planned and materialized batch nnz differ");
        }
        const int* batch_row_counts = planned_row_counts != nullptr
                                          ? planned_row_counts
                                          : fallback_row_counts;

        int row_begin = 0;
        while (row_begin < n_rows && batch_nnz > 0) {
            int block_nnz_begin = h_gather_indptr[row_begin];
            size_t block_nnz_limit =
                std::min((size_t)batch_nnz,
                         (size_t)block_nnz_begin + gather_stage.capacity);
            int block_nnz_limit_i32 = checked_int_span(
                block_nnz_limit, "rowstream range stage block limit");
            const int* row_limit = std::upper_bound(
                h_gather_indptr + row_begin + 1, h_gather_indptr + n_rows + 1,
                block_nnz_limit_i32);
            int row_end = (int)(row_limit - h_gather_indptr) - 1;
            if (row_end <= row_begin) {
                throw std::runtime_error(
                    "rowstream range row exceeds pinned stage capacity");
            }
            int block_nnz = h_gather_indptr[row_end] - block_nnz_begin;
            if (block_nnz > 0) {
                int stage_slot =
                    (int)(range_stage_sequence % CSR_RANGE_STAGE_SLOTS);
                gather_stage.wait(stage_slot);
                InT* h_gather_vals = gather_stage.template get<0>(stage_slot);
                StageIndexT* h_gather_cols =
                    gather_stage.template get<1>(stage_slot);
                materialize_csr_range_row_block<StageIndexT>(
                    h_data, h_indices, h_indptr, row_begin, row_end,
                    input_col_start, cursor.data(), batch_row_counts,
                    h_gather_indptr, block_nnz_begin, h_gather_vals,
                    h_gather_cols);
                cuda_check(cudaMemcpyAsync(gather_vals + block_nnz_begin,
                                           h_gather_vals,
                                           (size_t)block_nnz * sizeof(InT),
                                           cudaMemcpyHostToDevice, copy_stream),
                           "rowstream range gathered vals H2D");
                cuda_check(cudaMemcpyAsync(
                               gather_cols + block_nnz_begin, h_gather_cols,
                               (size_t)block_nnz * sizeof(StageIndexT),
                               cudaMemcpyHostToDevice, copy_stream),
                           "rowstream range gathered cols H2D");
                gather_stage.record(stage_slot, copy_stream);
                range_stage_sequence++;
            }
            row_begin = row_end;
        }

        cuda_check(cudaMemcpyAsync(gather_indptr, h_gather_indptr,
                                   (n_rows + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice, copy_stream),
                   "rowstream range indptr H2D");
        indptr_stage.record(metadata_slot, copy_stream);
        if constexpr (ovo) {
            batch_ready[b]->record(copy_stream);
            cuda_check(cudaStreamWaitEvent(stream, batch_ready[b]->get(), 0),
                       "rowstream compute wait for gathered batch");
            for (int c = col; c < col + sb_cols; c += slab_cols) {
                int w = std::min(slab_cols, col + sb_cols - c);
                rank.extract_csr(gather_vals, gather_cols, gather_indptr, c, w,
                                 stats, rank_scratch, stream);
                rank.rank_slabs(rank_scratch, c, w, stream);
            }
            gather_consumed[b]->record(stream);
            col += sb_cols;
            continue;
        }
        cuda_check(cudaMemsetAsync(active_col_offsets, 0,
                                   (sb_cols + 1) * sizeof(int), copy_stream),
                   "rowstream range histogram reset");
        if (batch_nnz > 0) {
            int histogram_blocks = std::min(
                range_histogram_blocks, 1 + (batch_nnz - 1) / UTIL_BLOCK_SIZE);
            compact_col_histogram_kernel<<<histogram_blocks, UTIL_BLOCK_SIZE,
                                           (size_t)sb_cols * sizeof(int),
                                           copy_stream>>>(
                gather_cols, active_col_offsets, batch_nnz, col, sb_cols);
            CUDA_CHECK_LAST_ERROR(compact_col_histogram_kernel);
        }
        batch_ready[b]->record(copy_stream);
        cuda_check(cudaStreamWaitEvent(stream, batch_ready[b]->get(), 0),
                   "rowstream compute wait for gathered batch");
        if (batch_nnz > 0) {
            size_t temp_bytes = scan_temp_bytes;
            cuda_check(cub::DeviceScan::ExclusiveSum(
                           scan_temp, temp_bytes, active_col_offsets,
                           active_col_offsets, sb_cols + 1, stream),
                       "rowstream range column offset scan");
        }
        cuda_check(cudaMemcpyAsync(write_pos, active_col_offsets,
                                   sb_cols * sizeof(int),
                                   cudaMemcpyDeviceToDevice, stream),
                   "rowstream range write positions D2D");

        // Scatter mini-CSR into the column-batch CSC accumulator.
        csr_scatter_to_csc_kernel<InT, StageIndexT, int>
            <<<(n_rows + tpb - 1) / tpb, tpb, 0, stream>>>(
                gather_vals, gather_cols, gather_indptr, write_pos,
                csc_vals_orig, csc_row_idx, n_rows, col, col + sb_cols, 0);
        CUDA_CHECK_LAST_ERROR(csr_scatter_to_csc_kernel);
        gather_consumed[b]->record(stream);

        cast_with_stats(csc_vals_orig, csc_vals_f32, csc_row_idx,
                        active_col_offsets, sb_cols, col, stats, stream);
        rank(CscBatch{csc_vals_f32, csc_row_idx, active_col_offsets, batch_nnz,
                      sb_cols},
             rank_scratch, col, stream);
        col += sb_cols;
    }
    cuda_check(cudaStreamSynchronize(stream), "rowstream sync");
}

// Range shards rebase column ids to the local interval; stage them as uint16
// whenever that interval fits.
template <typename InT, typename IndexT, typename Ranker>
static void sparse_csr_host(const InT* h_data, const IndexT* h_indices,
                            const IndexT* h_indptr, const IndexT* row_starts,
                            const IndexT* row_stops, int col_start, int n_rows,
                            int n_cols, int sub_batch_cols, const Ranker& rank,
                            const SparseStats& stats) {
    if (n_cols <= CSR_RANGE_U16_COLUMN_CAPACITY)
        sparse_csr_host_typed<uint16_t>(h_data, h_indices, h_indptr, row_starts,
                                        row_stops, col_start, n_rows, n_cols,
                                        sub_batch_cols, rank, stats);
    else
        sparse_csr_host_typed<int>(h_data, h_indices, h_indptr, row_starts,
                                   row_stops, col_start, n_rows, n_cols,
                                   sub_batch_cols, rank, stats);
}
