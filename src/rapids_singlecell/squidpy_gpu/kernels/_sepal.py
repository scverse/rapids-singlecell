from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

# Kernel for finding nearest saturated node for each unsaturated node
get_nhood_idx_with_distance_kernel = r"""
(
    const int* __restrict__ unsat_nodes,
    const {0}* __restrict__ spatial,
    const int* __restrict__ sat_nodes,
    const int* __restrict__ g_indptr,
    const int* __restrict__ g_indices,
    const bool* __restrict__ sat_mask,
    int* __restrict__ nearest_sat,
    int n_unsat,
    int n_sat
)
{{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_unsat) return;

    int node = unsat_nodes[tid];
    {0} node_x = spatial[node * 2];
    {0} node_y = spatial[node * 2 + 1];

    {0} min_dist = -1.0;  // -1.0 means no closest sat found yet
    int closest = -1;

    // Phase 1: Check graph neighbors for saturated nodes
    int start = g_indptr[node];
    int end = g_indptr[node + 1];

    for (int i = start; i < end; i++) {{
        int neighbor = g_indices[i];
        if (sat_mask[neighbor]) {{
            closest = neighbor;  // Take first
            break;               // Stop immediately
        }}
    }}

    // Phase 2: If no saturated graph neighbors, search ALL saturated nodes
    if (closest == -1) {{
        for (int i = 0; i < n_sat; i++) {{
            int sat_node = sat_nodes[i];
            {0} sat_x = spatial[sat_node * 2];
            {0} sat_y = spatial[sat_node * 2 + 1];
            {0} dist = fabs(node_x - sat_x) + fabs(node_y - sat_y);

            if (min_dist < 0.0 || dist < min_dist) {{
                min_dist = dist;
                closest = sat_node;
            }}
        }}
    }}

    nearest_sat[tid] = closest;
}}
"""

# SEPAL simulation kernel with device function for entropy computation
# This kernel must be compiled with CuPy's RawKernel due to device functions
sepal_simulation_kernel = r"""
extern "C" {
    // Device function: Computes entropy using cooperative thread reduction
    __device__ double compute_entropy_cooperative(
        const double* __restrict__ conc,
        int n_sat,
        const int* __restrict__ sat_nodes,
        int tid,
        int blockSize
    ) {
        __shared__ double total_sum_shared[256];
        __shared__ double entropy_shared[256];
        // np.finfo(np.float64).eps  # ~2.22e-16
        const double eps = 2.220446049250313e-16;

        // Each thread accumulates its portion of nodes
        double local_sum = 0.0;
        for (int i = tid; i < n_sat; i += blockSize) {
            double val = conc[sat_nodes[i]];
            if (val > eps) local_sum += val;
        }

        total_sum_shared[tid] = local_sum;
        __syncthreads();

        // Parallel reduction to sum all values
        for (int s = blockSize / 2; s > 0; s >>= 1) {
            if (tid < s) {
                total_sum_shared[tid] += total_sum_shared[tid + s];
            }
            __syncthreads();
        }

        double total_sum = total_sum_shared[0];
        if (total_sum < eps) return 0.0;
        // see here why
        // https://stats.stackexchange.com/questions/57069/alternative-to-shannons-entropy-when-probability-equal-to-zero/433096

        // Each thread computes entropy for its portion
        double local_entropy = 0.0;
        for (int i = tid; i < n_sat; i += blockSize) {
            double val = conc[sat_nodes[i]];
            if (val > eps) {
                double normalized = val / total_sum;
                local_entropy += -normalized * log(fmax(normalized, eps));
            }
        }

        entropy_shared[tid] = local_entropy;
        __syncthreads();

        // Parallel reduction for entropy
        for (int s = blockSize / 2; s > 0; s >>= 1) {
            if (tid < s) {
                entropy_shared[tid] += entropy_shared[tid + s];
            }
            __syncthreads();
        }

        return entropy_shared[0] / (double)(n_sat);
    }

    // Main SEPAL simulation kernel - processes one gene per block
    __global__ void sepal_simulation(
        double* __restrict__ concentration_all,
        double* __restrict__ derivatives_all,
        const int* __restrict__ sat_nodes,
        const int* __restrict__ sat_idx,
        const int* __restrict__ unsat_nodes,
        const int* __restrict__ unsat_idx,
        double* __restrict__ results,
        int n_cells,
        int n_genes,
        int n_sat,
        int n_unsat,
        int max_neighs,
        int n_iter,
        double dt,
        double thresh
    ) {
        int gene_idx = blockIdx.x;              // Each block handles one gene
        int tid = threadIdx.x;                  // Thread ID (0-255)
        int blockSize = blockDim.x;             // 256 threads per block

        if (gene_idx >= n_genes) return;

        // Per-gene pointers into global arrays
        double* concentration = &concentration_all[gene_idx * n_cells];
        double* derivatives = &derivatives_all[gene_idx * n_cells];

        // Convergence tracking
        __shared__ double prev_entropy;
        __shared__ int convergence_iter;
        __shared__ bool converged_flag;

        if (tid == 0) {
            prev_entropy = 1.0;
            convergence_iter = -1;
            converged_flag = false;
        }
        __syncthreads();

        // Main iteration loop - all threads process their portion of nodes
        for (int iter = 0; iter < n_iter; iter++) {
            // Phase 1: Update derivatives for saturated nodes
            for (int i = tid; i < n_sat; i += blockSize) {
                double neighbor_sum = 0.0;
                for (int j = 0; j < max_neighs; j++) {
                    neighbor_sum += concentration[sat_idx[i * max_neighs + j]];
                }
                int sat_global_idx = sat_nodes[i];
                double center = concentration[sat_global_idx];
                double d2 = 0.0;

                if (max_neighs == 4) {
                    d2 = (neighbor_sum - 4.0 * center);
                } else if (max_neighs == 6) {
                    d2 = (2.0 * neighbor_sum - 12.0 * center) / 3.0;
                }
                derivatives[sat_global_idx] = d2;
            }
            __syncthreads();

            // Phase 2: Update saturated node concentrations
            for (int i = tid; i < n_sat; i += blockSize) {
                int sat_global_idx = sat_nodes[i];
                concentration[sat_global_idx] += derivatives[sat_global_idx] * dt;
                concentration[sat_global_idx] = fmax(0.0, concentration[sat_global_idx]);
            }
            __syncthreads();

            // Phase 3: Update unsaturated nodes based on nearest saturated
            for (int i = tid; i < n_unsat; i += blockSize) {
                int unsat_global_idx = unsat_nodes[i];
                concentration[unsat_global_idx] += derivatives[unsat_idx[i]] * dt;
                concentration[unsat_global_idx] = fmax(0.0, concentration[unsat_global_idx]);
            }
            __syncthreads();

            // Check convergence using entropy
            if (!converged_flag) {
                double current_entropy = compute_entropy_cooperative(
                    concentration, n_sat, sat_nodes, tid, blockSize
                );

                if (tid == 0) {
                    double entropy_diff = fabs(current_entropy - prev_entropy);
                    if (entropy_diff <= thresh && convergence_iter == -1) {
                        convergence_iter = iter;
                        converged_flag = true;
                    }
                    prev_entropy = current_entropy;
                }
                __syncthreads();
            }

            if (converged_flag) break;
        }

        // Store result for this gene
        if (tid == 0) {
            results[gene_idx] = convergence_iter >= 0 ?
                (double)convergence_iter : -1.0;
        }
    }
}
"""


def _get_get_nhood_idx_with_distance(dtype):
    """Get neighborhood index with distance kernel specialized for the given dtype.

    This kernel finds the nearest saturated node for each unsaturated node.
    First checks graph neighbors, then falls back to spatial distance search.
    """
    return cuda_kernel_factory(
        get_nhood_idx_with_distance_kernel, (dtype,), "get_nhood_idx_with_distance"
    )


def _get_sepal_simulation(dtype=None):
    """Get SEPAL simulation kernel.

    This kernel simulates diffusion for multiple genes in parallel.
    Each block processes one gene with 256 threads cooperating.
    Uses double precision for concentration values.

    Parameters
    ----------
    dtype : dtype, optional
        Ignored. The kernel always uses double precision for numerical stability.
        This parameter exists for API compatibility.

    Note
    ----
    This kernel uses device functions and is compiled via CuPy's RawKernel
    rather than cuda_kernel_factory, as it requires special handling for
    the device function (compute_entropy_cooperative).
    """
    import cupy as cp

    return cp.RawKernel(sepal_simulation_kernel, "sepal_simulation")
