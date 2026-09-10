#pragma once

template <typename R, typename C, typename I, typename T, bool with_distances>
__global__ void assemble_graphs_kernel(
    const R* __restrict__ rows, const C* __restrict__ columns,
    const T* __restrict__ distances, const long long n_edges,
    const long long n_obs, const bool set_diag,
    I* __restrict__ adj_indptr, I* __restrict__ dst_indptr,
    I* __restrict__ adj_columns, I* __restrict__ dst_columns,
    float* __restrict__ adj_data, T* __restrict__ dst_data
) {
    const long long edge = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    // Sparse rows get independent diagonal threads to avoid long serial gaps.
    const bool sparse_rows = n_edges < n_obs;
    if (sparse_rows && edge >= n_edges) {
        const long long row = edge - n_edges;
        if (row > n_obs) return;
        long long left = 0, right = n_edges;
        while (left < right) {
            const long long middle = left + (right - left) / 2;
            if (rows[middle] < row) left = middle + 1;
            else right = middle;
        }
        const long long out = left + row;
        adj_indptr[row] = out;
        if (with_distances) dst_indptr[row] = out;
        if (row < n_obs) {
            adj_columns[out] = row;
            adj_data[out] = set_diag ? 1.0f : 0.0f;
            if (with_distances) {
                dst_columns[out] = row;
                dst_data[out] = 0;
            }
        }
        return;
    }
    if (edge > n_edges) return;
    const long long row = edge < n_edges ? rows[edge] : n_obs;
    if (edge < n_edges) {
        const long long out = edge + row + 1;
        adj_columns[out] = columns[edge];
        adj_data[out] = 1.0f;
        if (with_distances) {
            dst_columns[out] = columns[edge];
            dst_data[out] = distances[edge];
        }
    }
    if (sparse_rows) return;
    // Place a stored diagonal before every row, including skipped empty rows.
    const long long previous = edge ? rows[edge - 1] : -1;
    for (long long r = previous + 1; r <= row; ++r) {
        const long long out = edge + r;
        adj_indptr[r] = out;
        if (with_distances) dst_indptr[r] = out;
        if (r < n_obs) {
            adj_columns[out] = r;
            adj_data[out] = set_diag ? 1.0f : 0.0f;
            if (with_distances) {
                dst_columns[out] = r;
                dst_data[out] = 0;
            }
        }
    }
}
