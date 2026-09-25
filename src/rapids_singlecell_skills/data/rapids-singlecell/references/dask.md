# Dask execution

Use Dask only for capacity, row-chunking, or supported multi-GPU work; it is not the default in-memory route.

Do not fetch the RSC [out-of-core guide](https://rapids-singlecell.readthedocs.io/en/latest/out_of_core.html) by default: the constraints below plus live API introspection cover routine planning. Open it only when you need worked examples or the current Dask support list.

This file deliberately duplicates only pre-import orchestration constraints. Treat the worker preflight and the out-of-core guide as authoritative for dependency and transport details.

## Configure workers

Multi-GPU is not a flag on a call: it comes from the cluster. Stand up a `LocalCUDACluster` across the devices you intend to use, attach a client, and pass Dask-backed arrays — a method either supports that path or it does not, so confirm support before assuming a call scales out. A few methods shard internally across visible devices without Dask; check the live signature rather than inferring from the name.

Create and configure GPU workers before the client. Configure RMM and CuPy on each worker; client-process `rmm.reinitialize` does not configure workers. Select transport, allocator, and concurrency only after checking their current compatibility and measuring the workload.

## Shape the data

- Follow the out-of-core guide for supported block representations, conversion calls, and chunk layout. Do not infer backend support from an API name or annotation.
- Size chunks from measured worker peak memory and record the chosen layout. Recompute the estimate after filtering when later planning depends on it.
- Treat the co-released out-of-core support list as authoritative; API presence or a docstring alone does not establish Dask support. Confirm the selected signature with `python -m rapids_singlecell_skills.api describe <symbol>`, and add a scoped probe when behavior matters.
- Stop there for unsupported graphs or embeddings. Materialize only an intentionally reduced object that fits the target device, or stop; never hide a CPU fallback.

## Compute late

Keep the graph lazy and call `.compute()` at the last possible step, so no intermediate result is materialized on the way there. Do not call `.compute()` on the full expression matrix merely to build a graph or plot. Estimate the result before `.compute()` or `.persist()`; persist only data that fits across workers.

On OOM, reduce chunks and task concurrency before adding workers. Re-check the out-of-core guide before changing transport or allocator mode, and return only reduced results.
