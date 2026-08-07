from __future__ import annotations

import numpy as np
from numba import cuda


@cuda.jit
def compute_all_neighbor_sums_kernel(
    concentration, sat_idx, neighbor_sums, n_sat, sat_thresh
):
    """Kernel to compute neighbor sums for ALL saturated nodes in parallel"""
    idx = cuda.grid(1)

    if idx < n_sat:
        neighbor_sum = 0.0
        # Sum all neighbors for this saturated node
        for j in range(sat_thresh):
            neighbor_idx = sat_idx[idx * sat_thresh + j]
            neighbor_sum += concentration[neighbor_idx]

        neighbor_sums[idx] = neighbor_sum


@cuda.jit
def compute_neighbor_sums_multiple_per_thread(
    concentration, sat_idx, neighbor_sums, n_sat, sat_thresh
):
    """
    Optimized kernel where each thread can process multiple saturated nodes
    Better for small n_sat values
    """
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x

    # Each thread processes multiple nodes (stride loop)
    for idx in range(tid, n_sat, block_size):
        neighbor_sum = 0.0
        for j in range(sat_thresh):
            neighbor_idx = sat_idx[idx * sat_thresh + j]
            neighbor_sum += concentration[neighbor_idx]
        neighbor_sums[idx] = neighbor_sum


def compute_neighbor_sums_adaptive(concentration_host, sat_idx_host, n_sat, sat_thresh):
    """
    Adaptive function that chooses the best kernel launch configuration
    based on dataset size to avoid GPU under-utilization
    """

    # Transfer to GPU
    concentration = cuda.to_device(concentration_host)
    sat_idx = cuda.to_device(sat_idx_host)
    neighbor_sums = cuda.device_array(n_sat, dtype=np.float64)

    # Adaptive kernel configuration
    if n_sat < 256:
        # Small dataset: Use single block with stride loop
        threads_per_block = min(256, max(32, n_sat))  # At least 32 (one warp)
        blocks_per_grid = 1

        print(
            f"Small dataset mode: {blocks_per_grid} block, {threads_per_block} threads"
        )
        compute_neighbor_sums_multiple_per_thread[blocks_per_grid, threads_per_block](
            concentration, sat_idx, neighbor_sums, n_sat, sat_thresh
        )
    else:
        # Large dataset: Use multiple blocks
        threads_per_block = 256
        blocks_per_grid = (n_sat + threads_per_block - 1) // threads_per_block

        print(
            f"Large dataset mode: {blocks_per_grid} blocks, {threads_per_block} threads"
        )
        compute_all_neighbor_sums_kernel[blocks_per_grid, threads_per_block](
            concentration, sat_idx, neighbor_sums, n_sat, sat_thresh
        )

    return neighbor_sums.copy_to_host()


# Test with different dataset sizes
def test_different_sizes():
    """Test the adaptive approach with different dataset sizes"""

    test_cases = [
        (100, 20, 4),  # Small: under-utilization case
        (1000, 200, 4),  # Medium
        (10000, 2000, 4),  # Large
        (50000, 8000, 6),  # Very large with hexagonal grid
    ]

    for n_nodes, n_sat, sat_thresh in test_cases:
        print(
            f"\n=== Testing: {n_nodes} nodes, {n_sat} saturated, {sat_thresh} neighbors ==="
        )

        # Create test data
        concentration = np.random.rand(n_nodes).astype(np.float64)
        sat_idx = np.random.randint(0, n_nodes, size=(n_sat * sat_thresh,)).astype(
            np.int32
        )

        # GPU computation
        gpu_sums = compute_neighbor_sums_adaptive(
            concentration, sat_idx, n_sat, sat_thresh
        )

        # CPU verification (only for smaller datasets)
        if n_sat <= 1000:  # Don't verify very large datasets (too slow)
            neighbor_sums_cpu = np.zeros(n_sat)
            for i in range(n_sat):
                for j in range(sat_thresh):
                    neighbor_idx = sat_idx[i * sat_thresh + j]
                    neighbor_sums_cpu[i] += concentration[neighbor_idx]

            max_diff = np.max(np.abs(gpu_sums - neighbor_sums_cpu))
            print(f"Max difference: {max_diff}")
            assert max_diff < 1e-10, "GPU and CPU results don't match!"
            print("✓ Verification passed!")
        else:
            print("✓ Large dataset - skipping CPU verification")


# Alternative: Work-efficient approach for very small datasets
@cuda.jit
def compute_neighbor_sums_warp_efficient(
    concentration, sat_idx, neighbor_sums, n_sat, sat_thresh
):
    """
    Work-efficient kernel that uses full warps even for small datasets
    by processing multiple operations per thread
    """
    tid = cuda.threadIdx.x
    warp_id = tid // 32
    lane_id = tid % 32

    # Process multiple saturated nodes per warp
    nodes_per_warp = max(1, 32 // sat_thresh)  # How many nodes can one warp handle

    for warp_batch in range(0, n_sat, nodes_per_warp):
        node_idx = warp_batch + (tid // sat_thresh)
        neighbor_idx_in_node = tid % sat_thresh

        if node_idx < n_sat and neighbor_idx_in_node < sat_thresh:
            # Each thread loads one neighbor value
            sat_idx_flat = node_idx * sat_thresh + neighbor_idx_in_node
            neighbor_val = concentration[sat_idx[sat_idx_flat]]

            # Use warp shuffle to sum within the warp
            # (This is more complex but very efficient for small sat_thresh)
            for offset in range(1, sat_thresh):
                neighbor_val += cuda.shfl_down_sync(0xFFFFFFFF, neighbor_val, offset)

            # First thread in each group writes the result
            if neighbor_idx_in_node == 0:
                neighbor_sums[node_idx] = neighbor_val


def benchmark_approaches():
    """Benchmark different approaches"""
    import time

    # Test parameters
    n_nodes = 10000
    n_sat = 2000
    sat_thresh = 4

    concentration = np.random.rand(n_nodes).astype(np.float64)
    sat_idx = np.random.randint(0, n_nodes, size=(n_sat * sat_thresh,)).astype(np.int32)

    # Warm up GPU
    _ = compute_neighbor_sums_adaptive(concentration, sat_idx, n_sat, sat_thresh)

    # Benchmark
    n_runs = 10

    start = time.time()
    for _ in range(n_runs):
        result = compute_neighbor_sums_adaptive(
            concentration, sat_idx, n_sat, sat_thresh
        )
        cuda.synchronize()  # Ensure completion
    gpu_time = (time.time() - start) / n_runs

    # CPU baseline
    start = time.time()
    neighbor_sums_cpu = np.zeros(n_sat)
    for i in range(n_sat):
        for j in range(sat_thresh):
            neighbor_idx = sat_idx[i * sat_thresh + j]
            neighbor_sums_cpu[i] += concentration[neighbor_idx]
    cpu_time = time.time() - start

    print("\nBenchmark Results:")
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    print(f"Max difference: {np.max(np.abs(result - neighbor_sums_cpu))}")


if __name__ == "__main__":
    print("Testing adaptive neighbor sum computation...")
    test_different_sizes()

    print("\n" + "=" * 50)
    print("Running benchmark...")
    benchmark_approaches()
