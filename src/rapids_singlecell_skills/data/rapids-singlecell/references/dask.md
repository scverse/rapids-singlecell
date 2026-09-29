# Dask and multi-GPU

In-memory single-GPU runs are fastest; use managed memory for moderate overflow, and Dask when data exceed host-backed managed memory or when several GPUs should share preprocessing.

## Cluster

```python
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

cluster = LocalCUDACluster(
    CUDA_VISIBLE_DEVICES="0,1",
    threads_per_worker=1,
    rmm_pool_size="80%",  # or rmm_managed_memory=True for capacity
    rmm_allocator_external_lib_list=["cupy"],
)
client = Client(cluster)
```

- Workers get RMM from the cluster arguments; `rmm.reinitialize` in the notebook process does not reach them.
- `protocol="ucx"` enables NVLink but cannot be combined with managed memory; TCP is the robust default.

## Load lazily

```python
import anndata as ad
import zarr
from anndata.experimental import read_elem_lazy

f = zarr.open(path, mode="r")
adata = ad.AnnData(
    X=read_elem_lazy(f["X"], chunks=(20_000, -1)),  # rows chunked, all genes per chunk
    obs=ad.io.read_elem(f["obs"]),
    var=ad.io.read_elem(f["var"]),
)
rsc.get.anndata_to_GPU(adata)
```

## Supported steps

- Dask-capable: `calculate_qc_metrics`, `filter_cells`, `filter_genes`, `normalize_total` (without `exclude_highly_expressed`), `log1p`, `highly_variable_genes` (not `pearson_residuals`), `scale`, `pca` (`covariance_eigh` only), `score_genes`, `rank_genes_groups` (`wilcoxon_binned` or t-test, not exact `wilcoxon`), `rsc.get.aggregate` and every `rsc.dcg` method.
- `rsc.tl.leiden(..., use_dask=True)` distributes clustering, but it only pays off above about 10 million cells.
- After PCA the embedding is small: `.compute()` it if `obsm["X_pca"]` is a Dask array, then run neighbors, Leiden and UMAP in memory.
- Stop and reduce the data for steps without Dask support; never materialize the full matrix or fall back to CPU silently.

## Execute

- `persist()` after filtering only if the result fits across workers, and call `adata.X.compute_chunk_sizes()` after row filtering.
- `compute()` only reduced results; it gathers everything onto the client GPU.
- On OOM, shrink chunks and task concurrency before adding workers.
- CuPy sparse blocks hold at most 2**31 - 1 nonzeros; row chunks avoid the limit.

## Multi-GPU without Dask

- `rank_genes_groups`, `rsc.gr.co_occurrence`, `rsc.gr.spatial_autocorr` and the `Distance` methods take `multi_gpu`.
- Configure all devices with `rmm.reinitialize(pool_allocator=True, devices=[0, 1])`, and restrict visible GPUs with `CUDA_VISIBLE_DEVICES` before the process starts.
