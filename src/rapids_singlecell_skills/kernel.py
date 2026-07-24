"""Check a RAPIDS-singlecell environment before starting a notebook."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from typing import Any, Literal

Mode = Literal["pool", "managed"]
Report = dict[str, Any]

_STEPS = ("rmm", "gpu", "cuda", "rsc", "extensions")


def _pass(report: Report, step: str, summary: str) -> None:
    report["checks"][step] = {"status": "pass", "summary": summary}


def _fail(report: Report, step: str, error: BaseException) -> None:
    message = " ".join(str(error).splitlines()).strip() or repr(error)
    report["checks"][step] = {
        "status": "fail",
        "summary": message,
        "error": {"type": type(error).__name__, "message": message},
    }


def _finish(report: Report, *, display: bool) -> Report:
    report["ready"] = all(
        report["checks"].get(step, {}).get("status") == "pass" for step in _STEPS
    )
    if display:
        print("RAPIDS-singlecell environment preflight")
        for step in _STEPS:
            check = report["checks"].get(step, {"status": "skip", "summary": "not run"})
            print(f"{step}: {check['status']} - {check['summary']}")
        print("READY" if report["ready"] else "NOT READY")
    return report


def _rmm_options(mode: Mode, device: int) -> dict[str, bool | int]:
    if mode not in {"pool", "managed"}:
        raise ValueError("mode must be 'pool' or 'managed'")
    if device < 0:
        raise ValueError("device must be non-negative")
    return {
        "pool_allocator": mode == "pool",
        "managed_memory": mode == "managed",
        "devices": device,
    }


def _init_rmm(mode: Mode, *, device: int) -> Any:
    options = _rmm_options(mode, device)
    loaded = [name for name in ("cupy", "rapids_singlecell") if name in sys.modules]
    if loaded:
        raise RuntimeError(f"already imported: {', '.join(loaded)}; start fresh")

    import rmm

    rmm.reinitialize(**options)

    import cupy as cp
    from rmm.allocators.cupy import rmm_cupy_allocator

    cp.cuda.set_allocator(rmm_cupy_allocator)
    cp.cuda.Device(device).use()
    if cp.cuda.get_allocator() is not rmm_cupy_allocator:
        raise RuntimeError("CuPy is not using the RMM allocator")
    return cp


def _extension_names(directory: Path) -> list[str]:
    return sorted(
        {
            path.name[: -len(suffix)]
            for suffix in EXTENSION_SUFFIXES
            for path in directory.glob(f"*{suffix}")
            if path.name[: -len(suffix)].endswith("_cuda")
        }
    )


def _preflight(mode: Mode = "pool", *, device: int = 0, display: bool = True) -> Report:
    report: Report = {"checks": {}}
    if "ipykernel" in sys.modules:
        _fail(
            report,
            "rmm",
            RuntimeError("run before notebook startup; active notebook kernel found"),
        )
        return _finish(report, display=display)

    try:
        cp = _init_rmm(mode, device=device)
        _pass(report, "rmm", mode)
    except Exception as error:  # noqa: BLE001 - this command diagnoses import failures
        _fail(report, "rmm", error)
        return _finish(report, display=display)

    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        if count == 0 or device >= count:
            raise RuntimeError(f"device {device} unavailable; {count} GPU(s) visible")
        _pass(report, "gpu", f"{count} visible; using device {device}")

        total = cp.arange(16, dtype=cp.int32).sum()
        cp.cuda.runtime.deviceSynchronize()
        observed = int(total.get())
        if observed != 120:
            raise RuntimeError(f"unexpected CuPy result: {observed}")
        _pass(report, "cuda", f"CuPy {cp.__version__}; synchronized computation")
    except Exception as error:  # noqa: BLE001 - this command diagnoses CUDA failures
        step = "gpu" if "gpu" not in report["checks"] else "cuda"
        _fail(report, step, error)
        return _finish(report, display=display)

    try:
        rsc = importlib.import_module("rapids_singlecell")
        _pass(report, "rsc", f"RAPIDS-singlecell {rsc.__version__}")
    except Exception as error:  # noqa: BLE001 - this command diagnoses import failures
        _fail(report, "rsc", error)
        return _finish(report, display=display)

    failures: dict[str, str] = {}
    modules: dict[str, Any] = {}
    try:
        directory = Path(rsc.__file__).resolve().parent / "_cuda"
        names = _extension_names(directory)
        if not names:
            raise RuntimeError("no installed native _cuda extensions found")
        for name in names:
            try:
                modules[name] = importlib.import_module(
                    f"rapids_singlecell._cuda.{name}"
                )
            except Exception as error:  # noqa: BLE001 - report every loader failure
                failures[name] = " ".join(str(error).splitlines())
        if failures:
            raise RuntimeError(f"failed to load: {', '.join(failures)}")
        norm = modules.get("_norm_cuda")
        if norm is None:
            raise RuntimeError("_norm_cuda is not installed")
        values = cp.asarray([[1.0, 3.0]], dtype=cp.float32)
        norm.mul_dense(
            values,
            nrows=1,
            ncols=2,
            target_sum=8.0,
            stream=cp.cuda.get_current_stream().ptr,
        )
        cp.cuda.runtime.deviceSynchronize()
        observed = cp.asnumpy(values).ravel().tolist()
        if observed != [2.0, 6.0]:
            raise RuntimeError(f"unexpected _norm_cuda result: {observed}")
        _pass(report, "extensions", f"{len(names)} loaded; _norm_cuda succeeded")
    except Exception as error:  # noqa: BLE001 - this command diagnoses kernel failures
        _fail(report, "extensions", error)

    return _finish(report, display=display)


def main(argv: list[str] | None = None) -> int:
    """Run the disposable preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pool", "managed"), default="pool")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = _preflight(args.mode, device=args.device, display=not args.json)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 1


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
