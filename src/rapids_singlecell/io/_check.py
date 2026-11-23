from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import zarr
from zarr.storage import LocalStore

from ._config import zarr_kvikio_or_none

if TYPE_CHECKING:
    from zarr.storage import StoreLike


def check(path: StoreLike | None = None, *, verbose: bool = True) -> dict[str, Any]:
    """Check whether zarr data can be read straight into GPU memory, and how fast.

    Reports the installed KvikIO and nvCOMP versions, whether GPUDirect Storage is active
    (from ``gdscheck`` and KvikIO's compatibility mode), and, given a store, whether its ``X`` can be decoded on the GPU
    and how to improve its layout (see :func:`~rapids_singlecell.io.convert_zarr`).
    Checking a store's layout needs `zarr-kvikio <https://github.com/scverse/zarr-kvikio>`_
    (without its CUDA extra on the CPU).

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
    zk = zarr_kvikio_or_none()
    system: dict[str, Any] = (
        zk.check() if zk is not None else {"available": False, "versions": {}, "gds": {"status": "unknown"}}
    )
    report: dict[str, Any] = {
        "gpu_io_available": system["available"],
        "versions": {"zarr-kvikio": None if zk is None else zk.__version__, **system["versions"]},
        "gds": system["gds"],
    }
    if path is not None:
        report["store"] = check_store(path) if zk is not None else None
    if verbose:
        _print(report)
    return report


def check_store(path: StoreLike) -> dict[str, Any]:
    """Describe the arrays of ``X`` and list what keeps them from being read fast on the GPU."""
    describe = _zarr_kvikio().describe
    g = zarr.open_group(path, mode="r")
    x = g["X"]
    arrays = {k: x[k] for k in ("data", "indices", "indptr")} if isinstance(x, zarr.Group) else {"X": x}
    info: dict[str, Any] = {"encoding": x.attrs.get("encoding-type"), "arrays": {}}
    problems: list[str] = []
    for name, arr in arrays.items():
        assert isinstance(arr, zarr.Array)
        desc = describe(arr)
        info["arrays"][name] = desc
        problems += [f"`{name}`: {p}" for p in desc["problems"]]
    # (the GPU decodes chunks from local files only)
    info["gpu_readable"] = type(g.store) is LocalStore and all(d["gpu_readable"] for d in info["arrays"].values())
    info["problems"] = problems
    return info


def _zarr_kvikio() -> Any:
    if find_spec("zarr_kvikio") is None:
        msg = "This needs zarr-kvikio: `pip install zarr-kvikio` (with its `cuda13` or `cuda12` extra for GPU reads)."
        raise ImportError(msg)
    import zarr_kvikio  # (raises its own error if it fails to import, e.g. with an untested zarr)

    return zarr_kvikio


def _print(report: dict[str, Any]) -> None:
    versions = ", ".join(f"{k} {v}" for k, v in report["versions"].items() if v)
    if versions:
        print(f"Versions: {versions}")
    if report["gpu_io_available"]:
        status = "available"
    elif report["versions"]["zarr-kvikio"] is None:
        status = "zarr-kvikio not installed" if find_spec("zarr_kvikio") is None else "zarr-kvikio failed to import"
    else:
        status = "KvikIO or nvCOMP not installed"
    print(f"GPU reads (zarr-kvikio, KvikIO + nvCOMP): {status}")
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
        print("Checking the store's layout needs zarr-kvikio.")
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
