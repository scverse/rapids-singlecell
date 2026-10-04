"""Export the Harmony cuTile kernels: one cubin per kernel and GPU architecture.

Usage: python export_harmony_tile.py OUTPUT_DIR sm_XX
Writes OUTPUT_DIR/<symbol>_<sm_XX>.cubin for every kernel in KERNELS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cuda.tile as ct
from cuda.tile.compilation import (
    ArrayConstraint,
    CallingConvention,
    KernelSignature,
    ScalarConstraint,
    export_kernel,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harmony_tile import make_neg_rw, make_rtz

# Padded PC widths with a kernel; wider embeddings keep float32 R.
PC_WIDTHS = (64, 128)
# (name, row tile, occupancy), tuned on an RTX PRO 6000 (sm_120) at atlas
# shapes: R^T Z over all cells ("large") and per batch ("small"), and -R W.
RTZ_CONFIGS = (("large", 64, 2), ("small", 32, 1))
NEG_RW_CONFIG = (128, 2)


def _matrix(dtype, ndim=2):
    return ArrayConstraint(
        dtype,
        ndim=ndim,
        index_dtype=ct.int32,
        stride_lower_bound_incl=(0,) * (ndim - 1) + (None,),
        alias_groups=(),
        may_alias_internally=False,
        stride_constant=(None,) * (ndim - 1) + (1,),
        stride_divisible_by=(1,) * ndim,
        shape_divisible_by=(1,) * ndim,
        base_addr_divisible_by=16,
    )


def _signature(params, symbol):
    return KernelSignature(
        parameters=params, calling_convention=CallingConvention.cutile_python_v1()
    ).with_symbol(symbol)


def kernels():
    """(kernel, signature) for every exported symbol."""
    out = []
    for dp in PC_WIDTHS:
        for name, row_tile, occupancy in RTZ_CONFIGS:
            out.append(
                (
                    make_rtz(dp, row_tile, occupancy),
                    _signature(
                        [
                            _matrix(ct.bfloat16),
                            _matrix(ct.float32),
                            _matrix(ct.float32, 3),
                            ScalarConstraint(ct.int32),
                            ScalarConstraint(ct.int32),
                        ],
                        f"rsc_harmony_rtz_{name}_d{dp}",
                    ),
                )
            )
        out.append(
            (
                make_neg_rw(dp, *NEG_RW_CONFIG),
                _signature(
                    [_matrix(ct.bfloat16), _matrix(ct.float32), _matrix(ct.float32)],
                    f"rsc_harmony_neg_rw_d{dp}",
                ),
            )
        )
    return out


def main(output_dir: str, gpu_code: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for kernel, signature in kernels():
        export_kernel(
            kernel,
            [signature],
            f"{output_dir}/{signature.symbol}_{gpu_code}.cubin",
            gpu_code=gpu_code,
            output_format="cubin",
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
