# Memory Management

In rapids-singlecell, efficient memory management is crucial for handling large-scale datasets.
This is facilitated by the integration of the RAPIDS Memory Manager ({mod}`rmm`). {mod}`rmm` is automatically invoked upon importing `rapids-singlecell`, modifying the default allocator for cupy.
Integrating {mod}`rmm` with `rapids-singlecell` slightly modifies the execution speed of {mod}`cupy`. This change typically results in a minimal performance trade-off.
However, it's crucial to be aware that certain specific functions, like {func}`~.pp.harmony_integrate`, might experience a more significant impact on performance efficiency due to this integration.
Users can overwrite the default behavior with {func}`rmm.reinitialize`.
Configure RMM before creating GPU arrays; reinitializing while earlier RMM
allocations are still alive results in undefined behavior.
The `devices` argument configures RMM memory resources; it does not restrict
which GPUs a CUDA process can access. See {doc}`multi_gpu` for device
selection and isolation.

## Quick start

For the common cases, pick one mode based on your dataset and hardware:

- If your data fits in GPU VRAM: use the pool allocator for speed → see [Pool Allocator](#pool-allocator).
- If your data is larger than VRAM: use managed memory to spill to host RAM → see [Managed Memory](#managed-memory).

RMM also supports a pool backed by managed memory. This can reduce allocation
overhead while retaining oversubscription, but it requires deliberate
`initial_pool_size` and `maximum_pool_size` values. In particular, the maximum
must be large enough to grow beyond VRAM if oversubscription is required.
Managed memory remains incompatible with NVLink.

## Managed Memory

- Purpose: use datasets larger than GPU VRAM by spilling to host RAM.
- How it works: VRAM oversubscription; data migrates between GPU and host as needed.
- Trade-off: slower than fully-in-VRAM; slowdown grows with how much you spill.
- Good for: very large datasets that otherwise OOM; exploratory or batch runs where correctness matters more than peak speed.

```python
# Enable `managed_memory`
import rmm
import cupy as cp
from rmm.allocators.cupy import rmm_cupy_allocator

rmm.reinitialize(managed_memory=True, pool_allocator=False)
cp.cuda.set_allocator(rmm_cupy_allocator)
```

## Pool Allocator

- Purpose: speed up allocations and reduce fragmentation when data fits in VRAM.
- How it works: pre-allocates a pool; subsequent allocations come from the pool.
- Trade-off: keeps memory reserved; needs sufficient VRAM.
- Good for: allocation-heavy steps (e.g., neighbor graphs, harmony integration) and repeated runs.

```python
# Enable `pool_allocator`
import rmm
import cupy as cp
from rmm.allocators.cupy import rmm_cupy_allocator
rmm.reinitialize(
    managed_memory=False,
    pool_allocator=True,
)
cp.cuda.set_allocator(rmm_cupy_allocator)
```

## Best Practices
To achieve optimal memory management in rapids-singlecell, consider the following guidelines:

* **Large-scale Data Analysis:** Utilize `managed_memory` for datasets exceeding your VRAM's capacity, keeping in mind the potential performance penalties.
* **Performance-Critical Operations:** Choose `pool_allocator` when speed is critical and sufficient VRAM is available.
* **Advanced managed-memory pool:** Combining `managed_memory=True` and `pool_allocator=True` is supported, but size the pool explicitly and benchmark it for the workload.

### Troubleshooting

- CUDA out-of-memory (OOM) while using the pool allocator → switch to [Managed Memory](#managed-memory) or reduce dataset size.
- Very slow runtime with managed memory → reduce oversubscription or switch back to [Pool Allocator](#pool-allocator) if VRAM allows.

## Further Reading
For a more in-depth understanding of rmm and its functionalities, refer to the [RAPIDS Memory Manager documentation](https://docs.rapids.ai/api/rmm/stable/python/).


## System requirements and limits

rapids-singlecell performs most computations on the GPU.
Ensure your system has a CUDA-capable GPU with sufficient VRAM for your datasets.

- With an RTX 3090, analyzing around 200,000 cells is typically feasible.
- With an A100 80GB, analyses with 1,000,000+ cells are possible.

For larger datasets, use {mod}`~rmm` managed memory to oversubscribe GPU memory to host RAM (similar to SWAP).
This may introduce a performance penalty but can still outperform CPU-only runs. See the Managed Memory section above for how to enable it.

Limit note: CuPy sparse matrices normally use 32-bit `indices` and `indptr`,
which limits a single matrix or Dask block to at most 2**31-1
(2,147,483,647) explicitly stored values. rapids-singlecell kernels support
64-bit sparse indices, but high-level use still depends on CuPy preserving
those index buffers. Row-chunked Dask arrays avoid the per-block limit.
