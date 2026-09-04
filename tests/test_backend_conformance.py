"""Run squidpy's backend conformance suite against the RSC backend."""

from __future__ import annotations

import pytest

validate_backend = pytest.importorskip(
    "squidpy.testing.backend_conformance"
).validate_backend


def test_conformance():
    results = validate_backend("rapids-singlecell")
    for name, status in results.items():
        assert status == "PASSED", f"{name}: {status}"
