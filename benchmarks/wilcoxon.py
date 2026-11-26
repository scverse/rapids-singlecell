r"""Time exact Wilcoxon from raw local AnnData Zarr through the complete ranked result.

Install this checkout, dask-cuda and zarr-kvikio[cuda13,dask]. To reproduce the
batch-decoder benchmark, use scverse/zarr-kvikio revision 2475caa. For example:

    CUDA_VISIBLE_DEVICES=0,1 python benchmarks/wilcoxon.py cell_atlas_kvikio.zarr cell_type \
        --reference "amacrine cell" --trials 3 --advise-cold --rmm-pool-size 82GiB \
        --device-memory-limit 64GiB --worker-memory-limit 96GiB

Use --reference rest to compare every group against all other cells. The JSON lines include
cluster settings and per-trial wall times. Cluster startup, metadata and cache advice
are outside the timer. POSIX_FADV_DONTNEED is advisory.
"""

from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import anndata as ad
import pandas as pd
import zarr
import zarr_kvikio
from dask_cuda import LocalCUDACluster
from distributed import Client
from zarr_kvikio.dask import enable

import rapids_singlecell as rsc


def _gpu_name():
    from cupy.cuda import runtime

    return runtime.getDeviceProperties(runtime.getDevice())["name"].decode()


def _run(args):
    if args.row_chunk < 1 or args.trials < 1:
        raise ValueError("--row-chunk and --trials must be positive")
    with (
        LocalCUDACluster(
            protocol=args.protocol,
            threads_per_worker=args.threads,
            rmm_pool_size=args.rmm_pool_size,
            rmm_maximum_pool_size=args.rmm_pool_size,
            rmm_allocator_external_lib_list=["cupy"],
            device_memory_limit=args.device_memory_limit,
            memory_limit=args.worker_memory_limit,
            dashboard_address=None,
        ) as cluster,
        Client(cluster) as client,
        zarr.config.set({"kvikio.fetch": "kvikio"}),
        enable(client),
    ):
        root = zarr.open_group(args.store, mode="r")
        x = root["X"]
        if isinstance(x, zarr.Array):
            arrays = [x]
        elif x.attrs.get("encoding-type") == "csr_matrix":
            arrays = [x[key] for key in ("data", "indices", "indptr")]
        else:
            raise ValueError("Expected a dense array or CSR matrix in X")
        layouts = {a.path: zarr_kvikio.describe(a) for a in arrays}
        if any(not layout["gpu_readable"] for layout in layouts.values()):
            raise ValueError(f"X needs a GPU-readable local Zarr v3 layout: {layouts}")
        matrix = rsc.get.X_to_GPU(ad.experimental.read_elem_lazy(x, chunks=(args.row_chunk, -1)))
        labels = pd.Categorical(ad.io.read_elem(root["obs"][args.groupby])).remove_unused_categories()
        genes = ad.io.read_elem(root["var"][root["var"].attrs["_index"]])
        adata = ad.AnnData(matrix, obs=pd.DataFrame({args.groupby: labels}), var=pd.DataFrame(index=genes))
        metadata = {
            **vars(args),
            "store": str(args.store.resolve()),
            "shape": matrix.shape,
            "row_blocks": matrix.numblocks[0],
            "groups": len(labels.categories),
            "gpus": list(client.run(_gpu_name).values()),
            "gds_capability": zarr_kvikio.check()["gds"],
            "layouts": layouts,
            "versions": {p: version(p) for p in ("rapids-singlecell", "anndata", "zarr", "zarr-kvikio", "dask")},
        }
        print(json.dumps(metadata), flush=True)
        for trial in range(1, args.trials + 1):
            adata.X = matrix
            adata.uns.pop("log1p", None)
            if args.advise_cold:
                for a in arrays:
                    for path in (args.store / a.path).rglob("*"):
                        if path.is_file() and path.name != "zarr.json":
                            with path.open("rb", buffering=0) as f:
                                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            start = perf_counter()
            if not args.raw_counts:
                rsc.pp.normalize_total(adata, target_sum=10_000, use="X")
                rsc.pp.log1p(adata, use="X")
            result = rsc.tl.rank_genes_groups(
                adata,
                args.groupby,
                use="X",
                method="wilcoxon",
                reference=args.reference,
                tie_correct=True,
                use_continuity=False,
                mean_in_log_space=True,
                skip_empty_groups=True,
                return_u_values=args.return_u_values,
                n_genes=adata.n_vars,
                pts=True,
                out=True,
            )
            seconds = perf_counter() - start
            scores = result["scores"]
            print(
                json.dumps(
                    {"trial": trial, "seconds": seconds, "genes": len(scores), "contrasts": len(scores.dtype.names)}
                ),
                flush=True,
            )
            del scores, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("store", type=Path)
    parser.add_argument("groupby", help="Categorical column in obs")
    parser.add_argument("--reference", default="rest")
    parser.add_argument("--row-chunk", type=int, default=100_000)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--raw-counts", action="store_true", help="Rank X without normalization or log1p")
    parser.add_argument("--return-u-values", action="store_true", help="Return U statistics in the scores field")
    parser.add_argument(
        "--advise-cold", action="store_true", help="Advise the OS to evict matrix files before each trial"
    )
    parser.add_argument("--protocol", default="ucx", choices=("ucx", "tcp"))
    parser.add_argument("--threads", type=int, default=4, help="Threads per GPU worker")
    parser.add_argument("--rmm-pool-size", help="Fixed per-worker RMM pool size, e.g. 82GiB")
    parser.add_argument("--device-memory-limit", default="default")
    parser.add_argument("--worker-memory-limit", default="auto")
    _run(parser.parse_args())
