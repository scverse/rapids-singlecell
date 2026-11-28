# Out-of-core with Dask (GPU)

Process datasets larger than GPU memory by chunking work with Dask while keeping arrays on the GPU via CuPy. Chunking also mitigates the CuPy sparse limit of `.nnz ≤ 2**31-1` by operating on smaller blocks.

## Start a Dask CUDA cluster

Choose one of these presets:

--------------------------------------------------------------------------------

### A) NVLink / Performance preset (UCX + RMM pool, no managed memory)

Best when the data fits across GPUs and you want fast P2P.

```python
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

# Example: use 8 local GPUs
cluster = LocalCUDACluster(
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
    protocol="ucx",
    threads_per_worker=1,  # GPU-safe default
    rmm_pool_size="80%",  # per-worker pool; % of free VRAM at start
    rmm_managed_memory=False,  # avoid UM to maximize P2P
    rmm_allocator_external_lib_list=["cupy"],  # auto-patch CuPy to use RMM
)
client = Client(cluster)
```

### B) Capacity / Robustness preset (TCP + Managed memory)

Best when you need to stretch VRAM (slower P2P, but fewer OOMs).

```python
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

cluster = LocalCUDACluster(
    CUDA_VISIBLE_DEVICES="0,1",  # scale as needed
    protocol="tcp",  # TCP is often more predictable with UVM
    threads_per_worker=1,
    rmm_managed_memory=True,  # allow oversubscription (paging)
    rmm_allocator_external_lib_list=["cupy"],
)
client = Client(cluster)
```

### Notes

- `threads_per_worker=1` is recommended for GPU workloads. More threads can be faster but often increase temporary allocations, causing VRAM spikes/overflows; some dask-cuda releases also showed leaks with multi-threaded workers. With row chunks around ~20,000, 4–5 threads can still work on many GPUs.
- For capacity over speed, enable RMM managed memory (see {doc}`memory_management`). For highest peer-to-peer (NVLink) performance, prefer the RMM pool allocator and avoid managed memory.
- Multi-GPU transport: use UCX (`protocol="ucx"`) to enable NVLink. UCX typically uses more memory; TCP is more stable but slower.
- UCX is not compatible with CUDA managed memory. For UCX/NVLink, disable managed memory. TCP can be used with managed memory.

## Loading AnnData lazily from Zarr (from the multi-GPU notebook)

Load `AnnData` from a Zarr store with dense or CSR-encoded `X` as a Dask array, and `obs/var` read eagerly. Chunk by rows, keeping all genes in each chunk.

```python
import anndata as ad
import zarr
from anndata.experimental import read_elem_lazy

SPARSE_CHUNK_SIZE = 20_000
data_pth = "zarr/cell_atlas.zarr"  # example zarr path

f = zarr.open(data_pth, mode="r")
X = f["X"]

adata = ad.AnnData(
    X=read_elem_lazy(X, chunks=(SPARSE_CHUNK_SIZE, -1)),
    obs=ad.io.read_elem(f["obs"]),
    var=ad.io.read_elem(f["var"]),
)
```

## Example: out-of-core preprocessing pipeline

```python
import rapids_singlecell as rsc

rsc.get.anndata_to_GPU(adata)
# Normalize and transform
rsc.pp.normalize_total(adata, target_sum=10_000)
rsc.pp.log1p(adata)

# Rank all genes for each cell type against the remaining cells
rsc.tl.rank_genes_groups(adata, "cell_type", method="wilcoxon", use="X", tie_correct=True, pts=True)

# HVG selection
rsc.pp.highly_variable_genes(adata)
adata = adata[:, adata.var["highly_variable"]].copy()

# Scale and PCA
rsc.pp.scale(adata, zero_center=True, max_value=10)
rsc.pp.pca(adata, n_comps=50)
```

Most functions operate lazily; use `.compute()` only when you need concrete values on the client. Operations with reductions (e.g., scaling, HVG selection, PCA) synchronize and may call `compute()` internally.

## Computing results explicitly

Calling `.compute()` gathers the entire result onto the client, where it must fit in GPU memory.

```python
# Dense dask+cupy matrix → cupy
X_gpu = adata.X.compute()
```

## Persist and chunk sizes

- Persist a transformed matrix only if it fits across workers. For out-of-core Wilcoxon, pass the lazy matrix directly.
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
- {func}`~.pp.filter_cells`
- {func}`~.pp.filter_genes`
- {func}`~.pp.normalize_total`
- {func}`~.pp.log1p`
- {func}`~.pp.sqrt`
- {func}`~.pp.highly_variable_genes` (flavors: `seurat`, `cell_ranger`, `seurat_v3`, `seurat_v3_paper`, `poisson_gene_selection`)
- {func}`~.pp.scale`
- {func}`~.pp.regress_out`
- {func}`~.pp.pca`
- {func}`~.tl.score_genes`
- {func}`~.tl.score_genes_cell_cycle`
- `tl.louvain` (deprecated; set `use_dask=True` to distribute graph clustering)
- {func}`~.tl.leiden` (set `use_dask=True` to distribute graph clustering)
- {func}`~.tl.rank_genes_groups` (all methods, including exact `wilcoxon`)
- {func}`~rapids_singlecell.get.aggregate`
- {func}`~rapids_singlecell.dcg.aucell`, {func}`~rapids_singlecell.dcg.gsea`, {func}`~rapids_singlecell.dcg.mlm`, {func}`~rapids_singlecell.dcg.ora`, {func}`~rapids_singlecell.dcg.ulm`, {func}`~rapids_singlecell.dcg.waggr` and {func}`~rapids_singlecell.dcg.zscore`
- {func}`~rapids_singlecell.dcg.gsva` (computes Dask input into memory first)

For Dask inputs, {func}`~.pp.normalize_total` does not support
`exclude_highly_expressed=True`. {func}`~.pp.pca` uses the
`covariance_eigh` solver by default; `chunked=True`, `lanczos`, and `randomized` are not
supported.

(out-of-core-wilcoxon)=
## Exact Wilcoxon on log-normalized dense or CSR data

Use row chunks such as `chunks=(100_000, -1)`. The example above runs `normalize_total` with a fixed target and then `log1p`. Skip these calls if the input is already log-normalized. For local sharded Zstd stores, import `zarr_kvikio.dask`, set `zarr.config.set({"kvikio.fetch": "kvikio", "kvikio.decoders": 2, "kvikio.max_decode_bytes": 2**30})` and call `zarr_kvikio.dask.enable(client)` before reading.
Calling `rank_genes_groups` computes the lazy reads and any normalization in the input graph. Dask `map_blocks` collects expression values and group labels for a bounded set of genes. The GPU ranks each gene across the cells in the comparison, including implicit CSR zeros, and gives equal values their average rank. CSR chunks can store repeated values once in a lossless lookup table. When host RAM permits, workers keep packed records for later genes to avoid rereading most chunks. If the first four full-width chunks predict that the records would exceed the host budget, workers switch to smaller gene windows. With host buffering, each worker packs one CSR chunk at a time and waits for peer transfers to finish before reading the next. Choose a row chunk whose read and normalization temporaries fit in a single worker’s GPU pool. AnnData 0.13.4 can make an extra copy of a full-width CSR block during a Zarr read. The Replogle benchmark with 100,000-cell chunks and 10 GB pools used the [AnnData reader fix](https://github.com/scverse/anndata/pull/2676).
P-values use the two-sided normal approximation. Set `return_u_values=True` for Mann–Whitney U in `scores` and `pts=True` for expression fractions. The client holds the cell labels and the group-by-gene results in `adata.uns["rank_genes_groups"]`. Inputs must have fewer than `2**31` cells. Float64 U can round half-integers above `2**52`.

For a 10 GB pool per GPU, set `rmm_pool_size="10 GB"` and `rmm_maximum_pool_size="10 GB"` on `LocalCUDACluster`, with CuPy using RMM. The automatic GPU sort budget is the smaller of 25% of device memory and 30% of the pool maximum (up to 3 GB here). CUDA and I/O can allocate outside the pool. When space permits, workers can keep packed records in host RAM, using up to a quarter of each worker’s host memory allowance subject to available node RAM.
The 11.44-million-cell atlas used two workers with `protocol="ucx"`, `threads_per_worker=7` and 82 GiB per pool. To tune the automatic budgets, set `rapids-singlecell.wilcoxon.batch-size` for GPU sorting or `rapids-singlecell.wilcoxon.host-batch-size` for host records (`0` keeps them on the GPU). Host arrays and local schedulers use `array.chunk-size`.

## Troubleshooting

- CUDA OOM while running: reduce chunk size, enable RMM managed memory, or filter earlier.
- VRAM spikes or leaks: set `threads_per_worker=1`; limit task concurrency; consider TCP instead of UCX; restart workers to clear allocator state if needed.

## References

- [Dask-CUDA](https://docs.rapids.ai/api/dask-cuda/stable/)
- [Dask Array](https://docs.dask.org/en/stable/array.html)
- [CuPy Sparse](https://docs.cupy.dev/en/stable/reference/sparse.html)
