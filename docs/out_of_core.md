# Out-of-core with Dask (GPU)

Process datasets larger than GPU memory by chunking work with Dask while keeping arrays on the GPU via CuPy. Chunking also mitigates the CuPy sparse limit of `.nnz ≤ 2**31-1` by operating on smaller blocks.


## Quick start

{func}`rapids_singlecell.dask.start_cluster` starts a local Dask cluster with one worker per GPU and settings that work out of the box,
and returns a regular Dask {class}`~distributed.Client` for it;
and {func}`rapids_singlecell.io.read_lazy` loads an AnnData zarr store with `X` as a Dask array of GPU chunks:

```python
import rapids_singlecell as rsc

client = rsc.dask.start_cluster()  # all visible GPUs
adata = rsc.io.read_lazy("data.zarr")  # uses the cluster

rsc.pp.calculate_qc_metrics(adata)
rsc.pp.normalize_total(adata, target_sum=1e4)
rsc.pp.log1p(adata)
rsc.pp.highly_variable_genes(adata, flavor="cell_ranger", n_top_genes=2000)
adata = adata[:, adata.var["highly_variable"]].copy()
adata.X = adata.X.persist()  # keep the (much smaller) HVG matrix in GPU memory

rsc.pp.scale(adata, zero_center=False, max_value=10)
adata.X = adata.X.persist()
rsc.pp.pca(adata, n_comps=100)
adata.obsm["X_pca"] = rsc.get.X_to_CPU(adata.obsm["X_pca"]).compute()
client.restart()  # frees the workers' GPU memory for the steps in this process (and drops the persisted adata.X)

rsc.pp.neighbors(adata)
rsc.tl.umap(adata)
...
client.close()  # once you no longer need the cluster
client.cluster.close()
```

* {func}`~rapids_singlecell.dask.start_cluster` uses 4 threads per worker, UCX if available (else TCP), and RMM's asynchronous allocator
  (which does not fragment with large chunks), limited to 75% of each GPU, which leaves room for this process (e.g. the PCA) on the same GPU.
  It warns if a GPU is already busy (e.g. with another cluster). Every setting can be overridden,
  and the cluster stays usable for anything else you do with Dask.
  For capacity over speed, pass `rmm_managed_memory=True`; it then uses TCP, as UCX does not support managed memory.
* {func}`~rapids_singlecell.io.read_lazy` picks the chunk size (`chunks="auto"`) and, if possible,
  opens `X` to be read straight into GPU memory (see below). It needs no client:
  like any Dask array, `X` is computed on the cluster you started.
* Persist after steps that shrink the data: without it, every later step recomputes its input from disk
  (e.g. {func}`~rapids_singlecell.pp.pca` reads it twice). After filtering cells, also call
  `adata.X.compute_chunk_sizes()` if the chunk sizes became unknown (see "Persist and chunk sizes" below).

In a script (rather than a notebook), start the cluster under `if __name__ == "__main__":`.
Dask starts its workers by running the script again; without the guard, the workers fail to start
and {func}`~rapids_singlecell.dask.start_cluster` raises an error after about a minute.

The rest of this page shows how to set up the same things yourself.

## Reading data straight into GPU memory

By default, the compressed chunks of `X` are decompressed on the CPU and then copied to the GPU.
With [KvikIO](https://docs.rapids.ai/api/kvikio/stable/) and [nvCOMP](https://developer.nvidia.com/nvcomp) installed,
they are read straight into GPU memory and decompressed there:

```bash
pip install 'rapids-singlecell[io-cu13]'  # or [io-cu12]; `rapids-singlecell-cu13[io]` for the CUDA wheels
```

zarr then finds a codec pipeline that reads into GPU memory, `rapids_singlecell_zarr.KvikioCodecPipeline`,
which {func}`~rapids_singlecell.io.enable` switches on. It works with any store, and the GPU decodes chunks that are
zstd-compressed (or uncompressed), ideally small (nvCOMP decodes every chunk with one thread block) and in shards
(few files per read). Other chunks, e.g. blosc-compressed ones like in most existing stores, are decompressed on the CPU
and copied to the GPU, with the codec pipeline zarr was configured with before (e.g. [zarrs](https://zarrs-python.readthedocs.io)),
or else zarr's fast fused pipeline.
{func}`~rapids_singlecell.io.check` tells you whether a store is ready,
and {func}`~rapids_singlecell.io.convert_zarr` rewrites it with the same content:

```python
# versions, GPUDirect Storage status, and what to change in the store
rsc.io.check("data.zarr")
# zstd, 256 KiB chunks in shards, uint16 indices for < 65,536 genes
rsc.io.convert_zarr("data.zarr", "data_gpu.zarr")
```

{func}`~rapids_singlecell.dask.start_cluster` switches GPU reads on for its workers.
With your own cluster, call `rsc.io.enable(client)`, and everything else stays plain Dask and anndata:

```python
cluster = LocalCUDACluster(...)
client = Client(cluster)
rsc.io.enable(client)  # the workers read zarr chunks into GPU memory

f = zarr.open_group("data_gpu.zarr", mode="r")
adata = ad.AnnData(
    X=read_elem_lazy(f["X"], chunks=(100_000, -1), meta=cpx.csr_matrix((0, 0))),
    obs=ad.io.read_elem(f["obs"]),
    var=ad.io.read_elem(f["var"]),
)
```

Without a cluster, `rsc.io.enable()` switches GPU reads on for this process.
Both take effect right away; used as a context manager (`with rsc.io.enable(client): ...`),
they are undone at the end of the block, on the workers too.

GPU reads work with or without [GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/) (GDS):
without it, KvikIO reads through the page cache and a bounce buffer, which is still fast for cached data.

For example, on an 11.4 million cell atlas (19.5 billion non-zeros) on two GPUs, the steps from
{func}`~rapids_singlecell.pp.calculate_qc_metrics` to {func}`~rapids_singlecell.pp.pca` above took 26 s with GPU reads
(100,000 cells per chunk), and about 95 s decompressing on the CPU (20,000 cells per chunk, the fastest setting there),
with the subsetting improvements of [anndata#2675](https://github.com/scverse/anndata/pull/2675) and
[anndata#2676](https://github.com/scverse/anndata/pull/2676).

## Start a Dask CUDA cluster yourself

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
* `threads_per_worker=1` uses the least GPU memory. More threads overlap reading chunks with computing on them, which is faster,
  but increase temporary allocations, which can cause VRAM spikes/overflows; some dask-cuda releases also showed leaks with multi-threaded workers.
  {func}`~rapids_singlecell.dask.start_cluster` uses 4, which worked well in benchmarks; lower it if workers run out of memory.
* For capacity over speed, enable RMM managed memory (see {doc}`memory_management`). For highest peer-to-peer (NVLink) performance, prefer the RMM pool allocator and avoid managed memory.
* Multi-GPU transport: use UCX (`protocol="ucx"`) to enable NVLink. UCX typically uses more memory; TCP is more stable but slower.
* UCX is not compatible with CUDA managed memory. For UCX/NVLink, disable managed memory. TCP can be used with managed memory.

## Loading AnnData lazily from Zarr (from the multi-GPU notebook)

{func}`rapids_singlecell.io.read_lazy` does this for you. To do it yourself:

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

Calling `.compute()` gathers the entire result onto the client, where it must fit in GPU memory.

```python
# Dense dask+cupy matrix → cupy
X_gpu = adata.X.compute()
```

## Persist and chunk sizes

- Persist after major transformations or filtering to materialize results in worker memory and shorten later graphs.
- Recompute chunk sizes after filtering cells (when they become unknown, `nan`) to help Dask plan evenly across workers.

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
- {func}`~.tl.louvain` (set `use_dask=True` to distribute graph clustering)
- {func}`~.tl.leiden` (set `use_dask=True` to distribute graph clustering)
- {func}`~.tl.rank_genes_groups` (methods: `logreg`, `t-test`, `t-test_overestim_var`, `wilcoxon_binned`; not exact `wilcoxon`)
- {func}`~rapids_singlecell.get.aggregate`
- {func}`~rapids_singlecell.dcg.aucell`, {func}`~rapids_singlecell.dcg.gsea`, {func}`~rapids_singlecell.dcg.mlm`, {func}`~rapids_singlecell.dcg.ora`, {func}`~rapids_singlecell.dcg.ulm`, {func}`~rapids_singlecell.dcg.waggr` and {func}`~rapids_singlecell.dcg.zscore`
- {func}`~rapids_singlecell.dcg.gsva` (computes Dask input into memory first)

For Dask inputs, {func}`~.pp.normalize_total` does not support
`exclude_highly_expressed=True`. {func}`~.pp.pca` uses the
`covariance_eigh` solver by default; `chunked=True`, `lanczos`, and `randomized` are not
supported.

## Troubleshooting

- CUDA OOM while running: reduce chunk size, enable RMM managed memory, or filter earlier.
- VRAM spikes or leaks: lower `threads_per_worker` (down to 1); limit task concurrency; consider TCP instead of UCX; restart workers to clear allocator state if needed.

## References

- [Dask-CUDA](https://docs.rapids.ai/api/dask-cuda/stable/)
- [Dask Array](https://docs.dask.org/en/stable/array.html)
- [CuPy Sparse](https://docs.cupy.dev/en/stable/reference/sparse.html)
