"""Native metadata borrowing preserves allocation and stream ownership."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

from rapids_singlecell._cuda import _sparse2dense_cuda as native


class CustomArray(cp.ndarray):
    def __dlpack__(self, **kwargs):
        raise AssertionError("native metadata must use the CuPy descriptor")


@pytest.mark.parametrize("order", ["C", "F"])
def test_native_metadata_uses_cupy_descriptor_for_subclasses(order):
    data = cp.asarray([2.0, 4.0, 6.0]).view(type=CustomArray)
    output = cp.zeros((2, 2), dtype=np.float64, order=order).view(type=CustomArray)
    native.sparse2dense(
        cp.asarray([0, 2, 3], dtype=np.int32),
        cp.asarray([0, 1, 0], dtype=np.int32),
        data,
        out=output,
        major=2,
        minor=2,
        c_switch=order == "C",
        max_nnz=2,
    )
    cp.testing.assert_array_equal(output, [[2, 4], [6, 0]])
    # Borrowing must leave CuPy's exporter usable and its allocation owned here.
    capsule = cp.ndarray.__dlpack__(output, stream=-1)
    assert type(capsule).__name__ == "PyCapsule"
