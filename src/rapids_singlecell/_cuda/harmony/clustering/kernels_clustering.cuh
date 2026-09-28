#pragma once

#include <cuda_runtime.h>

// ---- PCG hash for random shuffle keys ----
__global__ void pcg_hash_kernel(unsigned int* __restrict__ out,
                                int* __restrict__ indices, int n,
                                unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        unsigned int state = static_cast<unsigned int>(i) ^ seed;
        state = state * 747796405u + 2891336453u;
        unsigned int word =
            ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        out[i] = (word >> 22u) ^ word;
        indices[i] = i;
    }
}

constexpr int KMEANS_WEIGHT_TILE_ROWS = 1024;
constexpr int KMEANS_WEIGHT_THREADS = 256;

template <typename T>
__global__ void kmeans_weight_tiles_kernel(const T* values, double* totals,
                                           int n_rows) {
    __shared__ double partial[KMEANS_WEIGHT_THREADS];
    int tid = threadIdx.x;
    size_t start = (size_t)blockIdx.x * KMEANS_WEIGHT_TILE_ROWS;
    double sum = 0;
    for (int j = tid; j < KMEANS_WEIGHT_TILE_ROWS; j += KMEANS_WEIGHT_THREADS)
        if (start + j < n_rows) sum += (double)values[start + j];
    partial[tid] = sum;
    __syncthreads();
    for (int step = KMEANS_WEIGHT_THREADS / 2; step; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        __syncthreads();
    }
    if (tid == 0) totals[blockIdx.x] = partial[0];
}

template <typename T>
__global__ void kmeans_select_center_kernel(const T* X, const T* weights,
                                            const double* totals,
                                            const double* uniforms, T* centers,
                                            int* n_draws, int n_rows,
                                            int n_cols, int n_tiles,
                                            int cluster) {
    __shared__ int choice;
    if (threadIdx.x == 0) {
        double total = 0;
        for (int tile = 0; tile < n_tiles; ++tile) total += totals[tile];
        choice = cluster % n_rows;
        if (total > 0) {
            double draw =
                fmin(uniforms[*n_draws] * total, nextafter(total, 0.));
            ++*n_draws;
            double previous = 0;
            int selected_tile = n_tiles - 1;
            for (int tile = 0; tile < n_tiles; ++tile) {
                double cumulative = previous + totals[tile];
                if (cumulative > draw) {
                    selected_tile = tile;
                    break;
                }
                previous = cumulative;
            }
            int begin = selected_tile * KMEANS_WEIGHT_TILE_ROWS;
            int end = begin + min(KMEANS_WEIGHT_TILE_ROWS, n_rows - begin);
            double residual = draw - previous;
            double cumulative = 0;
            int last_positive = begin;
            bool found = false;
            for (int row = begin; row < end; ++row) {
                double value = (double)weights[row];
                if (value > 0) last_positive = row;
                cumulative += value;
                if (cumulative > residual) {
                    choice = row;
                    found = true;
                    break;
                }
            }
            // Tree and sequential leaf sums can differ by a final ulp.
            if (!found) choice = last_positive;
        }
    }
    __syncthreads();
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x)
        centers[(size_t)cluster * n_cols + col] =
            X[(size_t)choice * n_cols + col];
}

template <typename T>
__device__ T objective_block_sum(T value, bool broadcast = true) {
    __shared__ T warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    value = threadIdx.x < (blockDim.x >> 5) ? warp_sums[lane] : T(0);
    if (warp == 0)
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
    if (!broadcast) return value;
    if (threadIdx.x == 0) warp_sums[0] = value;
    __syncthreads();
    value = warp_sums[0];
    __syncthreads();
    return value;
}

template <typename T>
__global__ void objective_rows_kernel(const T* R, const T* similarities,
                                      int n_rows, int n_cols, T sigma,
                                      const T* O, const T* E, const T* theta,
                                      int n_batches, bool stabilized,
                                      T* partial) {
    int row = blockIdx.x;
    T row_sum = T(0), distance = T(0);
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        T value = R[(size_t)row * n_cols + col];
        row_sum += value;
        distance +=
            value * T(2) * (T(1) - similarities[(size_t)row * n_cols + col]);
    }
    row_sum = objective_block_sum(row_sum);
    T entropy = T(0);
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        T value = R[(size_t)row * n_cols + col] / row_sum;
        entropy += value * log(value + T(1e-12));
    }
    T diversity = T(0);
    for (size_t i = (size_t)row * blockDim.x + threadIdx.x;
         i < (size_t)n_batches * n_cols; i += (size_t)n_rows * blockDim.x) {
        T numerator = stabilized ? O[i] + E[i] + T(1) : O[i] + T(1);
        diversity +=
            sigma * theta[i / n_cols] * O[i] * log(numerator / (E[i] + T(1)));
    }
    T total =
        objective_block_sum(distance + sigma * entropy + diversity, false);
    if (threadIdx.x == 0) partial[row] = total;
}

template <typename T>
__global__ void objective_reduce_kernel(const T* partial, int n_rows, T* out) {
    T sum = T(0);
    size_t i = threadIdx.x;
    for (; i + 3 * blockDim.x < n_rows; i += 4 * blockDim.x) {
        T first = partial[i], second = partial[i + blockDim.x],
          third = partial[i + 2 * blockDim.x],
          fourth = partial[i + 3 * blockDim.x];
        sum = (((sum + first) + second) + third) + fourth;
    }
    for (; i < n_rows; i += blockDim.x) sum += partial[i];
    sum = objective_block_sum(sum, false);
    if (threadIdx.x == 0) *out = sum;
}

template <typename T>
static T compute_objective(const T* R, const T* similarities, const T* O,
                           const T* E, const T* theta, T sigma, T* obj_scalar,
                           T* partial, int n_cells, int n_clusters,
                           int n_batches, bool stabilized,
                           cudaStream_t stream) {
    objective_rows_kernel<T><<<n_cells, 128, 0, stream>>>(
        R, similarities, n_cells, n_clusters, sigma, O, E, theta, n_batches,
        stabilized, partial);
    CUDA_CHECK_LAST_ERROR(objective_rows_kernel);
    objective_reduce_kernel<T>
        <<<1, 256, 0, stream>>>(partial, n_cells, obj_scalar);
    CUDA_CHECK_LAST_ERROR(objective_reduce_kernel);
    T host_obj;
    cuda_check(cudaMemcpyAsync(&host_obj, obj_scalar, sizeof(T),
                               cudaMemcpyDeviceToHost, stream),
               "objective copy");
    cuda_check(cudaStreamSynchronize(stream), "objective synchronization");
    return host_obj;
}
