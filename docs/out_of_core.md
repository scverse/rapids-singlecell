# Out-of-core with Dask (GPU)

Process datasets larger than GPU memory by chunking work with Dask while keeping arrays on the GPU via CuPy. Chunking also mitigates the CuPy sparse limit of `.nnz ≤ 2**31-1` by operating on smaller blocks.


## Start a Dask CUDA cluster

Choose one of these presets:

---

### A) NVLink / Performance preset (UCX + RMM pool, no managed memory)

Best when the data fits across GPUs and you want fast P2P.

```python
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

# Example: use 8 local GPUs
cluster = LocalCUDACluster(
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
    protocol="ucx",
    threads_per_worker=1,           # GPU-safe default
    rmm_pool_size="80%",            # per-worker pool; % of free VRAM at start
    rmm_managed_memory=False,       # avoid UM to maximize P2P
    rmm_allocator_external_lib="cupy",  # auto-patch CuPy to use RMM
)
client = Client(cluster)
```
### B) Capacity / Robustness preset (TCP + Managed memory)

Best when you need to stretch VRAM (slower P2P, but fewer OOMs).
```python
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

cluster = LocalCUDACluster(
    CUDA_VISIBLE_DEVICES="0,1",     # scale as needed
    protocol="tcp",                 # TCP is often more predictable with UVM
    threads_per_worker=1,
    rmm_managed_memory=True,        # allow oversubscription (paging)
    rmm_allocator_external_lib="cupy",
)
client = Client(cluster)
```

### Notes
* `threads_per_worker=1` is recommended for GPU workloads. More threads can be faster but often increase temporary allocations, causing VRAM spikes/overflows; some dask-cuda releases also showed leaks with multi-threaded workers. With row chunks around ~20,000, 4–5 threads can still work on many GPUs.
* For capacity over speed, enable RMM managed memory (see {doc}`memory_management`). For highest peer-to-peer (NVLink) performance, prefer the RMM pool allocator and avoid managed memory.
* Multi-GPU transport: use UCX (`protocol="ucx"`) to enable NVLink. UCX typically uses more memory; TCP is more stable but slower.
* UCX is not compatible with CUDA managed memory. For UCX/NVLink, disable managed memory. TCP can be used with managed memory.

## Loading AnnData lazily from Zarr (from the multi-GPU notebook)

Load `AnnData` from a Zarr store with `X` as a Dask array, and `obs/var` read eagerly. Chunk by rows.

```python
import anndata as ad
from packaging.version import parse as parse_version

if parse_version(ad.__version__) < parse_version("0.12.0rc1"):
    from anndata.experimental import read_elem_as_dask as read_dask
else:
    from anndata.experimental import read_elem_lazy as read_dask

import zarr

SPARSE_CHUNK_SIZE = 20_000
data_pth = "zarr/cell_atlas.zarr"  # example zarr path

f = zarr.open(data_pth)
X = f["X"]
shape = X.attrs["shape"]

adata = ad.AnnData(
    X=read_dask(X, (SPARSE_CHUNK_SIZE, shape[1])),
    obs=ad.io.read_elem(f["obs"]),
    var=ad.io.read_elem(f["var"]),
)
```

## Example: out-of-core preprocessing pipeline

```python
import rapids_singlecell as rsc
rsc.get.anndata_to_GPU(adata)
# Normalize and transform
rsc.pp.normalize_total(adata)
rsc.pp.log1p(adata)

# HVG selection
rsc.pp.highly_variable_genes(adata)
adata = adata[:, adata.var["highly_variable"]].copy()

# Scale and PCA
rsc.pp.scale(adata, zero_center=True, max_value=10)
rsc.pp.pca(adata, n_comps=50)
```

Most functions operate lazily; use `.compute()` only when you need concrete values on the client. Operations with reductions (e.g., scaling, HVG selection, PCA) synchronize and may call `compute()` internally.

## Computing results explicitly

```python
# Dense dask+cupy matrix → cupy
X_gpu = adata.X.compute()

```

## Persist and chunk sizes

- Persist after major transformations or filtering to materialize results in worker memory and shorten later graphs.
- Recompute chunk sizes to help Dask plan evenly across workers.

```python
# After filtering or transformations
adata.X = adata.X.persist()
adata.X.compute_chunk_sizes()
```

Persisting loads data into GPU memory across workers. This can quickly cause OOM if the dataset does not fit. On sufficiently large clusters, persisting can be extremely fast and effective.


## Multi-GPU notes

- Use `LocalCUDACluster(CUDA_VISIBLE_DEVICES="0,1,2,3")` to scale across GPUs.
- Ensure chunks are large enough to amortize scheduling but small enough to fit per-worker VRAM.
- Combine with RMM pool allocator for speed, or managed memory for capacity (see {doc}`memory_management`).
- NVLink: peer-to-peer performance is best with the RMM pool allocator. Managed memory can reduce or prevent effective NVLink use.

## Functions that support Dask

The functions below are implemented to run on Dask‑backed `AnnData` with GPU arrays. Most steps are lazy; reduction steps may synchronize internally. This covers the most common out‑of‑core workflows and will expand over time.

- {func}`~.pp.calculate_qc_metrics`
- {func}`~.pp.normalize_total`
- {func}`~.pp.log1p`
- {func}`~.pp.highly_variable_genes` (flavors: `seurat`, `cell_ranger`, `poisson_gene_selection`)
- {func}`~.pp.scale`
- {func}`~.pp.regress_out`
- {func}`~.pp.pca`
- {func}`~.tl.score_genes`
- {func}`~.tl.louvain`
- {func}`~.tl.leiden`
- {func}`~.tl.rank_genes_groups` (methods: `logreg`, `t-test`, `t-test_overestim_var`; NOT `wilcoxon`)
- {func}`~rapids_singlecell.get.aggregate`

## Troubleshooting

- CUDA OOM while running: reduce chunk size, enable RMM managed memory, or filter earlier.
- VRAM spikes or leaks: set `threads_per_worker=1`; limit task concurrency; consider TCP instead of UCX; restart workers to clear allocator state if needed.

## References

- [Dask-CUDA](https://docs.rapids.ai/api/dask-cuda/stable/)
- [Dask Array](https://docs.dask.org/en/stable/array.html)
- [CuPy Sparse](https://docs.cupy.dev/en/stable/reference/sparse.html)
