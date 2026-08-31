# GPU configuration

Most rapids-singlecell analyses use a single GPU.
On systems with several GPUs, you can assign different GPUs to independent analyses or use several GPUs for one supported operation.

## Quick guide

| What you want to do | Recommended setup |
| --- | --- |
| Run one analysis on one GPU | Start Python with `CUDA_VISIBLE_DEVICES=0`. |
| Run two independent scripts on two GPUs | Start one with `CUDA_VISIBLE_DEVICES=0` and the other with `CUDA_VISIBLE_DEVICES=1`. |
| Run two notebooks on two GPUs | Set a different `CUDA_VISIBLE_DEVICES` value in the first cell of each fresh kernel. |
| Keep a small display GPU out of an analysis | Exclude it from `CUDA_VISIBLE_DEVICES`, for example by starting with `CUDA_VISIBLE_DEVICES=1`. |
| Use several GPUs for one supported RSC operation | Make only the intended GPUs visible and pass `multi_gpu=True` or an explicit device list. |
| Process an out-of-core or distributed dataset | Use Dask-CUDA; see {doc}`out_of_core`. |

> Assign GPUs before CUDA is initialized in the Python process or notebook kernel.
> Use an RSC function's `multi_gpu` argument to control work inside that process.

`CUDA_VISIBLE_DEVICES` accepts integer IDs such as `0` and `1`.
GPU UUIDs are also supported when a stable physical identity is important.

## Check which GPUs Python can see

Run this near the beginning of a script or notebook:

```python
import cupy as cp

print("Number of visible GPUs:", cp.cuda.runtime.getDeviceCount())
print("Current GPU:", cp.cuda.Device().id)
```

If you start Python with `CUDA_VISIBLE_DEVICES=1`, that physical GPU is logical device `0` inside Python.
Seeing `Current GPU: 0` is therefore expected.

## Avoid using a small display GPU

Most single-GPU RSC operations allocate temporary arrays on the current CUDA device or on the device that owns the input array.
A new Python process normally starts with logical device `0` as its current device.
If you call `rsc.get.anndata_to_GPU(adata)` without selecting or masking a GPU first, `adata.X` normally moves to logical device `0`, and downstream temporary arrays usually follow it.

This can be a problem when physical GPU `0` is a small display card and another GPU has more memory.
The analysis can run out of memory on the display card even when the larger GPU is empty.

Exclude the display GPU when starting the process:

```bash
# Use physical GPU 1 and hide physical GPU 0 from this analysis.
CUDA_VISIBLE_DEVICES=1 python analysis.py
```

Physical GPU `1` becomes logical device `0` inside this process.
RSC and CuPy can use it normally without changes to the analysis code.

For a multi-GPU analysis, list only the compute GPUs:

```bash
# Physical GPU 0 is the display card; GPUs 1 and 2 are compute GPUs.
CUDA_VISIBLE_DEVICES=1,2 python analysis.py
```

The compute GPUs become logical devices `0` and `1` inside that process.
An operation that uses all visible GPUs can then use both compute cards without touching the display card.

Selecting a current device with `cp.cuda.Device(1).use()` is possible, but masking is safer.
Another library or an RSC operation that uses all visible GPUs can still access the display GPU if it remains visible.

## Run independent analyses on separate GPUs

Start one process for each analysis:

```bash
# First terminal or job
CUDA_VISIBLE_DEVICES=0 python analysis_a.py
```

```bash
# Second terminal or job
CUDA_VISIBLE_DEVICES=1 python analysis_b.py
```

Both scripts use logical device `0` internally because each process sees only its assigned physical GPU.
There is no need to change the RSC code between the two scripts.

GPU isolation does not isolate CPU cores, host RAM, or storage bandwidth.
Use different output and temporary paths, and make sure the machine has enough host memory for both analyses.

On a managed cluster, prefer the scheduler's GPU assignment mechanism.
Slurm, Kubernetes, and similar platforms normally set device visibility for the job.
Do not replace a scheduler-provided `CUDA_VISIBLE_DEVICES` unless the platform documentation tells you to.

## Run Jupyter notebooks on separate GPUs

Each Jupyter notebook normally has its own Python kernel process.
In a fresh kernel, the first cell can set GPU visibility before importing CuPy, rapids-singlecell, or another CUDA library.

In notebook A, make this the first cell:

```python
%env CUDA_VISIBLE_DEVICES=0
```

In notebook B, make this the first cell:

```python
%env CUDA_VISIBLE_DEVICES=1
```

Verify the assignment in the next cell:

```python
import cupy as cp

assert cp.cuda.runtime.getDeviceCount() == 1
assert cp.cuda.Device().id == 0
```

Restart each kernel before using this setup, and always run the visibility cell before any CUDA import or operation.
Changing the environment variable after CUDA was initialized does not change which GPUs the kernel can see.

Use the IPython `%env` command rather than a shell command such as `!export CUDA_VISIBLE_DEVICES=0`.
Commands prefixed with `!` run in a child shell, so their environment changes do not propagate back to the notebook kernel.

## Use several GPUs for one RSC operation

Some RSC functions can divide one computation across several GPUs.
First make only the intended GPUs visible:

```bash
CUDA_VISIBLE_DEVICES=0,1 python analysis.py
```

Then request multi-GPU execution from a supported function:

```python
import rapids_singlecell as rsc

rsc.gr.co_occurrence(
    adata,
    cluster_key="cluster",
    multi_gpu=True,
)
```

You can also select logical devices explicitly:

```python
rsc.gr.co_occurrence(
    adata,
    cluster_key="cluster",
    multi_gpu=[0, 1],
)
```

The meaning of `multi_gpu=None` and `multi_gpu=False` is function-specific, so consult the function's API documentation.
Use `multi_gpu=True` when all visible GPUs should participate, and use an explicit list when the exact device set matters.

Multi-GPU execution requires splitting work and transferring data to each GPU.
For smaller datasets, one GPU can be faster.

## Use Dask for distributed or out-of-core data

Dask-CUDA normally starts one worker process per GPU.
This differs from a single Python process using an RSC function's `multi_gpu` argument.

Use Dask when data are already Dask-backed, do not fit on one GPU, or need a distributed pipeline.
See {doc}`out_of_core` for supported functions and cluster configuration.

## AnnData `.raw` uses CPU memory

AnnData stores {attr}`~anndata.AnnData.raw` in CPU memory, even when `.X` was already moved to the GPU:

```python
rsc.get.anndata_to_GPU(adata)
adata.raw = adata.copy()

print(type(adata.X))  # CuPy or cupyx: GPU
print(type(adata.raw.X))  # NumPy or SciPy: CPU
```

This behavior matters because an RSC function can choose a different execution path for CPU input than for GPU-resident input.

For example, exact Wilcoxon with `use_raw=True` receives CPU input.
If several GPUs are visible, `multi_gpu=False` restricts this call to the current device:

```python
rsc.tl.rank_genes_groups(
    adata,
    groupby="leiden",
    method="wilcoxon",
    use_raw=True,
    multi_gpu=False,
)
```

A single-element list instead selects that exact logical device.
For example, `multi_gpu=[0]` runs Wilcoxon on logical device `0` even when device `1` is current.
If `adata.X` already created a context on device `1`, both devices can appear in `nvidia-smi`, although Wilcoxon runs only on device `0`.

Alternatively, use `use_raw=False` when `adata.X` contains the expression data intended for differential-expression testing.

## Troubleshooting unexpected GPU use

1. Check `cp.cuda.runtime.getDeviceCount()` and `cp.cuda.Device().id`.
2. Check whether the function uses `.X`, `.raw.X`, a layer, or an `.obsm` matrix, and whether that input is on the CPU or GPU.
3. Check the function's `multi_gpu` argument because `None` can mean all visible GPUs.
4. Confirm that `CUDA_VISIBLE_DEVICES` was set before Python or the Jupyter kernel initialized CUDA.
5. Restart a notebook kernel after changing its GPU assignment.
6. Remember that a process listed by `nvidia-smi` may only have an idle CUDA context.

For independent jobs, give each process or kernel only its assigned GPU.

## Advanced: current devices, arrays, and RMM

The controls involved in GPU execution have different purposes:

| Control | Scope | Purpose |
| --- | --- | --- |
| Scheduler or container assignment | Job or container | Enforces the platform's GPU allocation. |
| `CUDA_VISIBLE_DEVICES` | Python process or Jupyter kernel | Determines which GPUs CUDA libraries can see. |
| A function's `multi_gpu` argument | One RSC operation | Selects visible GPUs for that operation. |
| {class}`cupy.cuda.Device` | Current host thread | Selects the default logical device for new CUDA work. |
| `rmm.reinitialize(devices=...)` | RMM memory resources | Configures allocators but does not hide or select GPUs for algorithms. |

CuPy creates new arrays on the current device:

```python
import cupy as cp

gpu_device = 1
cp.cuda.Device(gpu_device).use()
X = cp.asarray(host_X)

assert X.device.id == gpu_device
```

Changing the current device later does not move `X`.
The array remains owned by the device where it was allocated.
Multi-GPU algorithms must explicitly copy or shard data across their selected devices.

A context manager temporarily changes the current device and restores the previous device afterward:

```python
with cp.cuda.Device(gpu_device):
    result = cp.sum(X)
```

Creating a CUDA context is enough for the process to appear on a GPU in `nvidia-smi`.
Contexts normally remain until the process or notebook kernel exits.

rapids-singlecell integrates with the RAPIDS Memory Manager ({mod}`rmm`).
RMM controls how GPU memory is allocated, but it does not select or hide GPUs.

Most users do not need to reinitialize RMM to choose a GPU.
Configure RMM only when changing memory policy, such as enabling a pool or managed memory, and do so before creating GPU arrays.
See {doc}`memory_management` for allocator configuration.
