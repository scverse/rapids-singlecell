from __future__ import annotations

import os
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr

from ._config import gpu_io_available

if TYPE_CHECKING:
    from zarr.storage import StoreLike

# nvCOMP decodes every zstd frame (= chunk) with one thread block, so large chunks decode slowly.
_MAX_GOOD_CHUNK_BYTES = 1 << 20


def check(path: StoreLike | None = None, *, verbose: bool = True) -> dict[str, Any]:
    """\
    Check whether zarr data can be read straight into GPU memory, and how fast.

    Reports the installed KvikIO and nvCOMP versions, whether GPUDirect Storage is active
    (from ``gdscheck``), and, given a store, whether its ``X`` can be decoded on the GPU
    and how to improve its layout (see :func:`~rapids_singlecell.io.convert_zarr`).

    Parameters
    ----------
    path
        An AnnData zarr store to check.
    verbose
        Print the report.

    Returns
    -------
    The report as a dictionary.
    """
    report: dict[str, Any] = {
        "gpu_io_available": gpu_io_available(),
        "versions": {
            pkg: _version(pkg) for pkg in ("kvikio", "nvidia-nvcomp", "zarr", "cupy")
        },
        "gds": _gds_status(),
    }
    if path is not None:
        report["store"] = check_store(path)
    if verbose:
        _print(report)
    return report


def check_store(path: StoreLike) -> dict[str, Any]:
    """Describe the arrays of ``X`` and list what keeps them from being read fast on the GPU."""
    g = zarr.open_group(path, mode="r")
    x = g["X"]
    arrays = (
        {k: x[k] for k in ("data", "indices", "indptr")}
        if isinstance(x, zarr.Group)
        else {"X": x}
    )
    n_vars = _n_vars(x)
    info: dict[str, Any] = {"encoding": x.attrs.get("encoding-type"), "arrays": {}}
    problems: list[str] = []
    for name, arr in arrays.items():
        assert isinstance(arr, zarr.Array)
        desc = _describe(arr)
        info["arrays"][name] = desc
        problems += [f"`{name}`: {p}" for p in desc["problems"]]
        if (
            name == "indices"
            and n_vars is not None
            and np.dtype(desc["dtype"]).itemsize > 2
            and n_vars <= np.iinfo(np.uint16).max + 1
        ):
            problems.append(
                f"`indices`: stored as {desc['dtype']}, but uint16 suffices for {n_vars:,} columns"
            )
    info["gpu_readable"] = all(d["gpu_readable"] for d in info["arrays"].values())
    info["problems"] = problems
    return info


def _describe(arr: zarr.Array) -> dict[str, Any]:
    from rapids_singlecell_zarr._pipeline import KvikioCodecPipeline, _layout_of

    desc: dict[str, Any] = {
        "dtype": str(arr.dtype),
        "zarr_format": arr.metadata.zarr_format,
        "codecs": [type(c).__name__ for c in _codecs(arr)],
        "chunk_bytes": int(np.prod(arr.chunks)) * arr.dtype.itemsize,
        "sharded": arr.shards is not None,
    }
    problems = []
    layout = None
    if arr.metadata.zarr_format == 3:
        layout = _layout_of(KvikioCodecPipeline.from_codecs(arr.metadata.codecs))
    if layout is None:
        problems.append(
            f"codecs {desc['codecs']} cannot be decoded on the GPU (needs zarr v3 with zstd or no compression)"
        )
    if desc["chunk_bytes"] > _MAX_GOOD_CHUNK_BYTES:
        problems.append(
            f"chunks of {desc['chunk_bytes'] / 2**20:.1f} MiB decode slowly on the GPU (~256 KiB is best)"
        )
    if not desc["sharded"] and arr.nchunks > 1:
        problems.append(
            "not sharded: every chunk is a separate file, which slows down small reads"
        )
    desc["gpu_readable"] = layout is not None
    desc["problems"] = problems
    return desc


def _codecs(arr: zarr.Array) -> list[Any]:
    if arr.metadata.zarr_format == 3:
        from zarr.codecs import ShardingCodec

        # describe what is inside the shards
        return [
            c
            for codec in arr.metadata.codecs
            for c in (codec.codecs if isinstance(codec, ShardingCodec) else [codec])
        ]
    return [c for c in (*arr.filters, *arr.compressors) if c is not None]


def _n_vars(x: zarr.Array | zarr.Group) -> int | None:
    shape = x.attrs.get("shape")
    if isinstance(shape, list) and len(shape) == 2:
        return (
            int(shape[1])
            if x.attrs.get("encoding-type") == "csr_matrix"
            else int(shape[0])
        )
    return None


def _version(pkg: str) -> str | None:
    try:
        return version(pkg)
    except PackageNotFoundError:
        # pip names e.g. nvCOMP's bindings and CuPy by CUDA version
        for suffix in ("-cu13", "-cu12", "-cuda13x", "-cuda12x"):
            try:
                return version(pkg + suffix)
            except PackageNotFoundError:
                pass
        return None


def _gds_status() -> dict[str, Any]:
    exe = shutil.which("gdscheck") or next(
        (
            p
            for p in (
                Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
                / "gds/tools/gdscheck",
                Path("/usr/local/cuda/gds/tools/gdscheck"),
            )
            if p.exists()
        ),
        None,
    )
    if exe is None:
        return {"status": "unknown (gdscheck not found)"}
    try:
        out = subprocess.run(
            [str(exe), "-p"], capture_output=True, text=True, timeout=60, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return {"status": f"unknown ({e})"}
    fields = {}
    for line in out.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    nvme = fields.get("NVMe", "")
    p2pdma = fields.get("properties.use_pci_p2pdma", "").lower() == "true"
    # (not `in`: gdscheck also reports "Unsupported")
    direct = nvme.lower().startswith("supported") or p2pdma
    return {
        "status": "direct" if direct else "compatibility mode",
        "nvme": nvme or None,
        "use_pci_p2pdma": p2pdma,
    }


def _print(report: dict[str, Any]) -> None:
    versions = ", ".join(f"{k} {v}" for k, v in report["versions"].items() if v)
    print(f"Versions: {versions}")
    print(
        "GPU reads (KvikIO + nvCOMP): "
        + ("available" if report["gpu_io_available"] else "not installed")
    )
    gds = report["gds"]
    note = (
        ""
        if gds["status"] != "compatibility mode"
        else " (reads go through the page cache and a bounce buffer; still fast if the data is cached)"
    )
    print(f"GPUDirect Storage: {gds['status']}{note}")
    if (store := report.get("store")) is None:
        return
    print(f"X ({store['encoding']}):")
    for name, d in store["arrays"].items():
        print(
            f"  {name}: {d['dtype']}, zarr v{d['zarr_format']}, {', '.join(d['codecs'])}, "
            f"chunks of {d['chunk_bytes'] / 2**10:,.0f} KiB, {'sharded' if d['sharded'] else 'not sharded'}"
        )
    if store["problems"]:
        print("To read faster on the GPU:")
        for p in store["problems"]:
            print(f"  - {p}")
        print("  → rewrite the store with rapids_singlecell.io.convert_zarr(src, dst)")
    else:
        print("The store is laid out for fast GPU reads.")
