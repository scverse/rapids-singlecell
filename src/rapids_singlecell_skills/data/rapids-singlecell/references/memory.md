# Memory management

Explicit RMM configuration is optional: importing RSC already installs an RMM-backed CuPy allocator. Configure RMM yourself only to select one of the deliberate routes below.

Do not fetch the RSC [memory guide](https://rapids-singlecell.readthedocs.io/en/latest/memory_management.html) by default: the routes below cover routine configuration. Open it only when a route is ambiguous or when current transport compatibility matters.

This file deliberately duplicates only decisions that must be made before RSC can be imported and introspected. The disposable preflight is the executable authority; if it disagrees with this checklist, stop and use the memory guide.

## Choose one route

| Situation | RMM configuration |
|---|---|
| No deliberate requirement | None; use the allocator RSC installs on import |
| Data and intermediates fit in VRAM | `managed_memory=False`, `pool_allocator=True` |
| Deliberate single-GPU oversubscription | `managed_memory=True`, `pool_allocator=False` |
| Deliberately sized managed pool | Both `True`, with explicit initial and maximum pool sizes |
| Dask or multi-GPU | Configure RMM on each worker; see `dask.md` |

Managed memory is the oversubscription toggle. It trades speed for capacity. A managed pool oversubscribes only if its maximum can grow beyond VRAM; do not select it without sizing deliberately. Check current transport compatibility in the memory guide.

## Initialize safely

- Run `rapids-singlecell-check-kernel` before Jupyter, adding `--mode managed` when that is the intended route. The check is disposable.
- When you do select a route, configure RMM in the notebook before importing CuPy or RSC and before creating GPU arrays. RSC installs an allocator on import, so an explicit choice must come first.
- Point CuPy at `rmm_cupy_allocator`. Never reinitialize while live RMM allocations exist, and never assume configuration crosses process boundaries.
- For Dask, configure `LocalCUDACluster`; do not call client-process `rmm.reinitialize` as a substitute for worker configuration.

## Plan capacity

Estimate dense representations, graphs, and temporary workspaces as well as the input matrix. Preserve counts, but plan residency from the next consumer: moving `X` does not imply that layers moved. If GPU preprocessing consumes a layer, move the required data once using the active `anndata_to_GPU` signature; `convert_all` includes supported layers but can enlarge the working set. Inspect the matching `anndata_to_CPU` signature rather than assuming every AnnData slot moves.

If a pool OOMs, reassess the working set and choose the managed-memory or Dask route deliberately. If managed memory thrashes, reduce the working set or use Dask instead of silently falling back to CPU computation. Verify representation limits against the active stack; do not preserve version-specific limits here.
