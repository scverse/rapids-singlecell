from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr

if TYPE_CHECKING:
    from zarr.storage import StoreLike


def check(path: StoreLike | None = None, *, verbose: bool = True) -> dict[str, Any]:
    """Check whether zarr data can be read straight into GPU memory, and how fast.

    Reports the installed KvikIO and nvCOMP versions, whether GPUDirect Storage is active
    (from ``gdscheck``), and, given a store, whether its ``X`` can be decoded on the GPU
    and how to improve its layout (see :func:`~rapids_singlecell.io.convert_zarr`).
    Checking a store's layout needs zarr-kvikio (the `io` extra).

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
    system = (
        _zarr_kvikio().check()
        if find_spec("zarr_kvikio") is not None
        else {"available": False, "versions": {}, "gds": {"status": "unknown"}}
    )
    report: dict[str, Any] = {
        "gpu_io_available": system["available"],
        "versions": system["versions"],
        "gds": system["gds"],
    }
    if path is not None:
        report["store"] = check_store(path) if find_spec("zarr_kvikio") is not None else None
    if verbose:
        _print(report)
    return report


def check_store(path: StoreLike) -> dict[str, Any]:
    """Describe the arrays of ``X`` and list what keeps them from being read fast on the GPU."""
    describe = _zarr_kvikio().describe
    g = zarr.open_group(path, mode="r")
    x = g["X"]
    arrays = {k: x[k] for k in ("data", "indices", "indptr")} if isinstance(x, zarr.Group) else {"X": x}
    n_vars = _n_vars(x)
    info: dict[str, Any] = {"encoding": x.attrs.get("encoding-type"), "arrays": {}}
    problems: list[str] = []
    for name, arr in arrays.items():
        assert isinstance(arr, zarr.Array)
        desc = describe(arr)
        info["arrays"][name] = desc
        problems += [f"`{name}`: {p}" for p in desc["problems"]]
        if (
            name == "indices"
            and n_vars is not None
            and np.dtype(desc["dtype"]).itemsize > 2
            and n_vars <= np.iinfo(np.uint16).max + 1
        ):
            problems.append(f"`indices`: stored as {desc['dtype']}, but uint16 suffices for {n_vars:,} columns")
    info["gpu_readable"] = all(d["gpu_readable"] for d in info["arrays"].values())
    info["problems"] = problems
    return info


def _zarr_kvikio() -> Any:
    try:
        import zarr_kvikio
    except ImportError as e:
        msg = (
            "This needs zarr-kvikio: `pip install 'rapids-singlecell[io-cu13]'` (or `[io-cu12]`), "
            "or just `pip install zarr-kvikio` where only the CPU is used."
        )
        raise ImportError(msg) from e
    return zarr_kvikio


def _n_vars(x: zarr.Array | zarr.Group) -> int | None:
    shape = x.attrs.get("shape")
    if isinstance(shape, list) and len(shape) == 2:
        return int(shape[1]) if x.attrs.get("encoding-type") == "csr_matrix" else int(shape[0])
    return None


def _print(report: dict[str, Any]) -> None:
    versions = ", ".join(f"{k} {v}" for k, v in report["versions"].items() if v)
    if versions:
        print(f"Versions: {versions}")
    print("GPU reads (KvikIO + nvCOMP): " + ("available" if report["gpu_io_available"] else "not installed"))
    gds = report["gds"]
    note = (
        ""
        if gds["status"] != "compatibility mode"
        else " (reads go through the page cache and a bounce buffer; still fast if the data is cached)"
    )
    print(f"GPUDirect Storage: {gds['status']}{note}")
    if "store" not in report:
        return
    if (store := report["store"]) is None:
        print("Checking the store's layout needs zarr-kvikio (the `io` extra).")
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
