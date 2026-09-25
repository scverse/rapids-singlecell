#pragma once

// CSR graphs from row-grouped unique off-diagonal edges. Row r stores its
// diagonal first (zeros too), at (edges of rows < r) + r: a row's first edge
// thread writes it, and row threads set indptr and edgeless rows' diagonals.
template <typename R, typename C, typename I, typename T>
__global__ void assemble_graphs(const R* rows, const C* cols, const T* values,
                                long long n_edges, long long n_obs, float diag,
                                I* indptr, I* indices, float* adj,
                                I* dst_indices, T* dst) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i > n_edges + n_obs) return;
    long long e = i, r = -1;  // write the diagonal of row r at e + r
    if (i < n_edges) {
        const long long slot = e + rows[i] + 1;
        indices[slot] = cols[i], adj[slot] = 1;
        if (dst) dst_indices[slot] = cols[i], dst[slot] = values[i];
        if (i == 0 || rows[i - 1] != rows[i]) r = rows[i];
    } else {
        const long long row = i - n_edges;
        long long hi = n_edges;
        for (e = 0; e < hi;) {  // e = first edge of row
            const long long mid = (e + hi) / 2;
            if (rows[mid] < row)
                e = mid + 1;
            else
                hi = mid;
        }
        indptr[row] = e + row;
        if (row < n_obs && (e == n_edges || rows[e] != row)) r = row;
    }
    if (r < 0) return;
    indices[e + r] = r, adj[e + r] = diag;
    if (dst) dst_indices[e + r] = r, dst[e + r] = 0;
}

// cupy.percentile's linear interpolation within each library's sorted values.
template <typename T>
__global__ void library_percentile(const double* index, const T* values,
                                   const long long* starts,
                                   const long long* sizes, long long n,
                                   T* out) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n || !sizes[i]) return;
    const long long below = floor(index[i]), last = sizes[i] - 1;
    const long long bottom = starts[i] + below;
    const long long top = starts[i] + min(below + 1, last);
    const T weight = index[i] - below, diff = values[top] - values[bottom];
    out[i] = weight < 0.5 ? values[bottom] + diff * weight
                          : values[top] - diff * (1 - weight);
}
