"""Exercise backend selection without importing CuPy or initializing CUDA."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_PACKAGE = "_test_rsc_cuda_backend"
_CUDA_INIT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "rapids_singlecell"
    / "_cuda"
    / "__init__.py"
)
_EXPORTED = tuple(
    ast.literal_eval(node.value)
    for node in ast.parse(_CUDA_INIT.read_text()).body
    if isinstance(node, ast.Assign)
    and any(
        isinstance(target, ast.Name) and target.id == "__all__"
        for target in node.targets
    )
)[0]


@pytest.fixture
def load_cuda_package(monkeypatch, tmp_path):
    # Runtime preloading is orthogonal to routing and must not touch CUDA here.
    for name in ("librmm", "rapids_logger"):
        runtime = ModuleType(name)
        runtime.load_library = lambda: None
        monkeypatch.setitem(sys.modules, name, runtime)

    def load(backend=None):
        spec = importlib.util.spec_from_file_location(
            _PACKAGE, _CUDA_INIT, submodule_search_locations=[str(tmp_path)]
        )
        package = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, _PACKAGE, package)
        if backend is not None:
            monkeypatch.setitem(sys.modules, f"{_PACKAGE}._rust_cuda", backend)
        spec.loader.exec_module(package)
        return package

    yield load
    for name in tuple(sys.modules):
        if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
            sys.modules.pop(name)


def _rust_backend():
    backend = ModuleType(f"{_PACKAGE}._rust_cuda")
    backend.__all__ = _EXPORTED
    backend.__backend__ = "rust"
    backend.__file__ = "/fake/_rust_cuda.abi3.so"
    for name in _EXPORTED:
        child = ModuleType(name)
        child.__backend__ = "rust"
        setattr(backend, name, child)
    return backend


@pytest.mark.parametrize("stale_legacy_modules", [False, True])
def test_rust_attribute_and_dotted_imports_use_same_modules(
    load_cuda_package, monkeypatch, stale_legacy_modules
):
    backend = _rust_backend()
    if stale_legacy_modules:
        for name in _EXPORTED:
            fullname = f"{_PACKAGE}.{name}"
            monkeypatch.setitem(sys.modules, fullname, ModuleType(fullname))

    package = load_cuda_package(backend)

    for name in _EXPORTED:
        child = getattr(backend, name)
        assert getattr(package, name) is child
        assert importlib.import_module(f"{_PACKAGE}.{name}") is child
        assert child.__backend__ == "rust"
        assert child.__name__ == f"{_PACKAGE}.{name}"
        assert child.__package__ == _PACKAGE
        assert child.__file__ == backend.__file__


def test_missing_rust_backend_does_not_load_legacy_binaries(
    load_cuda_package, monkeypatch
):
    fullname = f"{_PACKAGE}._sparse2dense_cuda"
    legacy = ModuleType(fullname)
    monkeypatch.setitem(sys.modules, fullname, legacy)

    package = load_cuda_package()

    assert package._sparse2dense_cuda is None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(fullname)


def test_missing_extensions_remain_available_for_documentation(load_cuda_package):
    package = load_cuda_package()

    assert package._jaccard_cuda is None
    assert package._sparse2dense_cuda is None
    assert package._scale_cuda is None
    with pytest.raises(AttributeError):
        package.unknown_module


@pytest.mark.parametrize(
    "backend_source,exception_type,detail",
    [
        (
            "import _nonexistent_rsc_backend_dependency\n",
            ModuleNotFoundError,
            "_nonexistent_rsc_backend_dependency",
        ),
        (
            "raise ImportError('ABI version mismatch')\n",
            ImportError,
            "ABI version mismatch",
        ),
    ],
)
def test_present_rust_backend_import_errors_are_surfaced(
    load_cuda_package, tmp_path, backend_source, exception_type, detail
):
    (tmp_path / "_rust_cuda.py").write_text(backend_source)

    with pytest.raises(
        ImportError, match="Failed to load the Rust CUDA backend"
    ) as error:
        load_cuda_package()

    assert isinstance(error.value.__cause__, exception_type)
    assert detail in str(error.value)


def test_unknown_rust_exports_are_rejected(load_cuda_package):
    backend = _rust_backend()
    backend.__all__ = ("_unknown_cuda",)
    backend._unknown_cuda = ModuleType("_unknown_cuda")

    with pytest.raises(ImportError, match="module mismatch.*unknown=.*_unknown_cuda"):
        load_cuda_package(backend)

    assert f"{_PACKAGE}._unknown_cuda" not in sys.modules


def test_incomplete_backend_is_rejected_without_partial_registration(load_cuda_package):
    backend = _rust_backend()
    backend.__all__ = _EXPORTED[:-1]

    with pytest.raises(ImportError, match="module mismatch.*missing="):
        load_cuda_package(backend)

    assert all(f"{_PACKAGE}.{name}" not in sys.modules for name in _EXPORTED)
