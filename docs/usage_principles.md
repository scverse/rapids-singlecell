# Usage Principles

## Import

```python
import rapids_singlecell as rsc
```

## Workflow

The workflow of *rapids-singlecell* is basically the same as *scanpy's*.
The main difference is the speed at which *rsc* can analyze the data. For more information please checkout the notebooks and API documentation.

### AnnData setup

{class}`~anndata.AnnData` supports GPU-enabled cupy arrays and sparse matrices.
Rapids-singlecell leverages this capability to perform analyses directly on {class}`~anndata.AnnData` objects with cupy arrays.

To get your {class}`~anndata.AnnData` object onto the GPU via cupy, you can move {attr}`~anndata.AnnData.X` or each {attr}`~anndata.AnnData.layers` to a GPU based matrix.

```python
adata.X = cpx.scipy.sparse.csr_matrix(adata.X)  # moves `.X` to the GPU
adata.X = adata.X.get() # moves `.X` back to the CPU
```

You can also use {mod}`rapids_singlecell.get` to move arrays and matrices.

```python
rsc.get.anndata_to_GPU(adata) # moves `.X` to the GPU
rsc.get.anndata_to_CPU(adata) # moves `.X` to the CPU
```

### Preprocessing

The preprocessing can be handled by the functions in {mod}`~.pp`. They offer accelerated versions of functions within {mod}`scanpy.pp`.

Example:

```python
rsc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor="seurat_v3", batch_key= "PatientNumber", layer = "counts")
adata = adata[:,adata.var["highly_variable"]==True]
rsc.pp.regress_out(adata,keys=["n_counts", "percent_MT"])
rsc.pp.scale(adata,max_value=10)
```

### Tools

The functions provided in {mod}`~.tl` are designed to as near drop-in replacements for the functions in {mod}`scanpy.tl`, but offer significantly improved performance.
Consequently, you can continue to use scanpy's plotting API.

Example:

```python
rsc.tl.tsne(adata)
sc.pl.tsne(adata, color="leiden")
```

### Decoupler-GPU

`dcg` offers accelerated drop in replacements for {func}`~rapids_singlecell.dcg.mlm`, {func}`~rapids_singlecell.dcg.ulm` and {func}`~rapids_singlecell.dcg.aucell` like:

```python
import decoupler as dc

model = dc.op.resource("PanglaoDB", organism="human")
rsc.dcg.ulm(adata, model , tmin=3)
acts_mlm = dc.pp.get_obsm(adata, key="score_ulm")
sc.pl.umap(acts_mlm, color=['NK cells'], cmap='coolwarm', vcenter=0)
```

### Pertpy-compatible API (use `rsc.ptg`)

rapids_singlecell exposes a pertpy-compatible, GPU-accelerated API under {mod}`rapids_singlecell.ptg` like:

```python
from rapids_singlecell import ptg

distance = ptg.Distance(metric="edistance", obsm_key="X_pca")
result = distance.pairwise(adata, groupby="perturbation")
res, res_var = distance.pairwise(
	adata, groupby="perturbation", bootstrap=True, n_bootstrap=100, multi_gpu=None
)
```

### Squidpy GPU helpers (use `rsc.squidpy_gpu`)

rapids_singlecell includes GPU-accelerated implementations of common `squidpy` workflows under {mod}`rapids_singlecell.gr` like:

```python
from rapids_singlecell import squidpy_gpu as sqg

sqg.spatial_autocorr(
	adata,
	connectivity_key="spatial_connectivities",
	mode="moran",
	n_perms=500,
)
sqg.co_occurrence(adata, cluster_key="labels", interval=50)
sqg.ligrec(adata, cluster_key="labels", n_perms=1000)
```
