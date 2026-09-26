"""Run scanpy's and squidpy's backend conformance suites against the RSC backend."""

from __future__ import annotations

import pytest

SCANPY_CONFORMANCE_BLOCKED = (
    "scanpy's conformance suite feeds CPU AnnData, which rsc.pp/tl reject, and its "
    "6-cell dataset crashes cuML's spectral UMAP initialisation"
)


@pytest.mark.parametrize(
    "module",
    [
        pytest.param(
            "scanpy.testing", marks=pytest.mark.skip(reason=SCANPY_CONFORMANCE_BLOCKED)
        ),
        "squidpy.testing.backend_conformance",
    ],
)
def test_conformance(module):
    testing = pytest.importorskip(module)
    if not hasattr(testing, "validate_backend"):
        pytest.skip(f"{module} has no backend conformance suite")

    results = testing.validate_backend("rapids-singlecell")
    for name, status in results.items():
        assert status == "PASSED", f"{name}: {status}"
