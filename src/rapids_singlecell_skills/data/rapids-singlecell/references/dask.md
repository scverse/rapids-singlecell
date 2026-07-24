# Dask execution

Use the current RSC [out-of-core guide](https://rapids-singlecell.readthedocs.io/en/latest/out_of_core.html)
for examples and the active API support list. Use Dask only for capacity,
row-chunking, or supported multi-GPU work; it is not the default in-memory route.

This file deliberately duplicates only pre-import orchestration constraints.
Treat the worker preflight and active guide as authoritative for dependency and
transport details.

## Configure workers

Create and configure GPU workers before the client. Configure RMM and CuPy on
each worker according to the active guide; client-process `rmm.reinitialize`
does not configure workers. Select transport, allocator, and concurrency only
after checking their current compatibility and measuring the workload.

## Shape the data

- Follow the active guide for supported block representations, conversion calls,
  and chunk layout. Do not infer backend support from an API name or annotation.
- Size chunks from measured worker peak memory and record the chosen layout.
  Recompute the estimate after filtering when later planning depends on it.
- Treat the co-released out-of-core support list as authoritative; API presence
  or a docstring alone does not establish Dask support. Use the live API helper
  for the selected signature and a scoped probe when behavior matters.
- Stop there for unsupported graphs or embeddings. Materialize only an
  intentionally reduced object that fits the target device, or stop; never hide
  a CPU fallback.

Do not call `.compute()` on the full expression matrix merely to build a graph or
plot. Estimate the result before `.compute()` or `.persist()`; persist only data
that fits across workers.

On OOM, reduce chunks and task concurrency before adding workers. Re-check the
active guide before changing transport or allocator mode, and return only
reduced results.
