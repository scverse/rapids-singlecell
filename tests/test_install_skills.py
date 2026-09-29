from __future__ import annotations

import importlib
import json
import re
import sys
import tomllib
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from types import ModuleType

import pytest

from rapids_singlecell_skills import install, kernel

ROOT = Path(__file__).parents[1]


SKILL = install.skill_source()
REFERENCES = ("conditions.md", "dask.md", "perturbation.md", "setup.md", "spatial.md")


def _skill_texts() -> dict[str, str]:
    return {
        path.relative_to(SKILL).as_posix(): path.read_text(encoding="utf-8")
        for path in SKILL.rglob("*.md")
    }


def test_skill_bundle_is_minimal() -> None:
    assert SKILL == (
        Path(install.__file__).resolve().parent / "data" / "rapids-singlecell"
    )
    files = {
        path.relative_to(SKILL).as_posix()
        for path in SKILL.rglob("*")
        if path.is_file()
    }
    assert files == {
        "SKILL.md",
        "agents/openai.yaml",
        *(f"references/{name}" for name in REFERENCES),
    }

    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert len(text.split()) <= 1200
    description = re.search(r"(?m)^description:\s*(.+)$", text)
    assert description is not None
    assert len(description.group(1).strip("\"'")) <= 500
    for name in REFERENCES:
        reference = (SKILL / "references" / name).read_text(encoding="utf-8")
        assert len(reference.split()) <= 800


def test_skill_links_resolve() -> None:
    linked = set()
    for name, text in _skill_texts().items():
        for target in re.findall(r"\]\(((?!https?://)[^)]+)\)", text):
            path = (SKILL / name).parent / target
            assert path.is_file(), f"{name} links to missing {target}"
            linked.add(path.resolve())
    assert linked == {(SKILL / "references" / name).resolve() for name in REFERENCES}


def test_skill_python_blocks_compile() -> None:
    for name, text in _skill_texts().items():
        for block in re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL):
            compile(block, name, "exec")


def test_skill_names_only_public_rsc_symbols() -> None:
    rsc = pytest.importorskip("rapids_singlecell")
    api = pytest.importorskip("rapids_singlecell_skills.api")
    contract = api._contract(rsc)
    namespaces = {symbol.split(".")[0] for symbol in contract}
    for name, text in _skill_texts().items():
        for symbol in re.findall(r"\brsc\.(\w+(?:\.\w+)*)", text):
            assert symbol in namespaces or symbol in contract, f"{name}: rsc.{symbol}"


@pytest.mark.parametrize(
    ("facade", "package"),
    [
        ("pp", "preprocessing"),
        ("tl", "tools"),
        ("gr", "squidpy_gpu"),
        ("dcg", "decoupler_gpu"),
        ("ptg", "pertpy_gpu"),
    ],
)
def test_facades_export_every_public_callable(facade: str, package: str) -> None:
    rsc = pytest.importorskip("rapids_singlecell")
    source = importlib.import_module(f"rapids_singlecell.{package}")
    module = getattr(rsc, facade)
    public = {
        name
        for name, value in vars(source).items()
        if not name.startswith("_")
        and callable(value)
        and not isinstance(value, ModuleType)
    }
    exported = set(module.__all__) | set(getattr(module, "__deprecated_exports__", {}))
    assert public <= exported


def test_api_index_finds_explicit_method_preferences() -> None:
    notes_path = ROOT / "src" / "rapids_singlecell_skills" / "api_notes.toml"
    with notes_path.open("rb") as handle:
        entries = tomllib.load(handle)["entries"]

    hvg_index = entries["pp.highly_variable_genes"]["index"]
    assert "poisson gene selection" in hvg_index["keywords"]

    dcg_index = entries["dcg.aucell"]["index"]
    assert "cell-level pathway activity" in dcg_index["keywords"]
    assert "decoupler" in dcg_index["keywords"]

    assert "squidpy" in entries["gr.calculate_niche_cellcharter"]["index"]["keywords"]
    assert "pertpy" in entries["ptg.Mixscape"]["index"]["keywords"]

    leiden = entries["tl.leiden"]
    assert "reproducible leiden" in leiden["index"]["keywords"]
    assert any("rng" in note["claim"] for note in leiden["notes"])


def test_parameter_choices_resolve_aliased_literals() -> None:
    """Choices must survive private aliases and PEP 695 `type` statements.

    `typing.get_type_hints` resolves a whole signature at once and dies on the first
    TYPE_CHECKING-only name, which would hide every parameter on these functions.
    """
    rsc = pytest.importorskip("rapids_singlecell")
    api = pytest.importorskip("rapids_singlecell_skills.api")

    # `flavor: flavors` -- a module-level alias, not an inline Literal
    hvg = api._parameter_choices(rsc.pp.highly_variable_genes)
    assert "seurat_v3" in hvg["flavor"]
    assert "poisson_gene_selection" in hvg["flavor"]

    # `method: _Method | None` -- a PEP 695 `type` alias inside a union
    ranked = api._parameter_choices(rsc.tl.rank_genes_groups)
    assert {"wilcoxon", "logreg", "t-test"} <= set(ranked["method"])

    # an inline Literal still works, and unresolvable annotations are skipped
    niche = api._parameter_choices(rsc.gr.calculate_niche_cellcharter)
    assert "variance" in niche["aggregation"]


def test_install_check_and_force(tmp_path: Path) -> None:
    destination = tmp_path / "rapids-singlecell"
    assert install.install_skill(destination=destination) == destination
    assert install.check_skill(destination=destination)[0]

    skill_file = destination / "SKILL.md"
    skill_file.write_text("modified\n", encoding="utf-8")
    assert not install.check_skill(destination=destination)[0]
    with pytest.raises(RuntimeError, match="destination differs"):
        install.install_skill(destination=destination)

    install.install_skill(destination=destination, force=True)
    assert install.check_skill(destination=destination)[0]


def test_install_refuses_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    destination = tmp_path / "rapids-singlecell"
    destination.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        install.install_skill(destination=destination, force=True)


@pytest.mark.parametrize(
    ("agent", "directory"),
    [("codex", ".codex"), ("claude", ".claude"), ("agents", ".agents")],
)
def test_default_agent_destinations(
    agent: str, directory: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    assert install._target(agent, None) == (
        tmp_path / directory / "skills" / "rapids-singlecell"
    )


def test_default_claude_science_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".claude-science"
    root.mkdir()
    (root / "active-org.json").write_text(
        json.dumps({"org_uuid": "test-org"}),
        encoding="utf-8",
    )

    assert install._target("claude-science", None) == (
        root / "orgs" / "test-org" / "skills" / "rapids-singlecell"
    )


def test_claude_science_requires_safe_active_org(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".claude-science"
    root.mkdir()

    with pytest.raises(ValueError, match="active organization"):
        install._target("claude-science", None)

    (root / "active-org.json").write_text(
        json.dumps({"org_uuid": "../escape"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid Claude Science org_uuid"):
        install._target("claude-science", None)


def test_custom_agent_requires_destination(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provide --dest"):
        install.install_skill("other")
    destination = tmp_path / "custom"
    assert install.install_skill("other", destination=destination) == destination


def test_installer_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    destination = tmp_path / "rapids-singlecell"
    assert install.main(["--dest", str(destination)]) == 0
    assert install.main(["--check", "--dest", str(destination)]) == 0
    assert "matches the active package" in capsys.readouterr().out


def test_managed_memory_toggle() -> None:
    assert kernel._rmm_options("managed", 2) == {
        "pool_allocator": False,
        "managed_memory": True,
        "devices": 2,
    }


def test_managed_pool_route_is_checkable() -> None:
    assert kernel._rmm_options(
        "managed-pool", 0, initial_pool_size="1GiB", maximum_pool_size="8GiB"
    ) == {
        "pool_allocator": True,
        "managed_memory": True,
        "devices": 0,
        "initial_pool_size": "1GiB",
        "maximum_pool_size": "8GiB",
    }
    assert "api" in kernel._STEPS


def test_preflight_reports_setup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("broken allocator")

    monkeypatch.delitem(sys.modules, "ipykernel", raising=False)
    monkeypatch.setattr(kernel, "_init_rmm", fail)
    report = kernel._preflight(display=False)
    assert not report["ready"]
    assert report["checks"]["rmm"]["status"] == "fail"


def test_preflight_refuses_notebook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ipykernel", ModuleType("ipykernel"))
    report = kernel._preflight(display=False)
    assert not report["ready"]
    assert "before notebook startup" in report["checks"]["rmm"]["summary"]


def test_extension_discovery(tmp_path: Path) -> None:
    suffix = EXTENSION_SUFFIXES[0]
    (tmp_path / f"_norm_cuda{suffix}").touch()
    (tmp_path / f"helper{suffix}").touch()
    assert kernel._extension_names(tmp_path) == ["_norm_cuda"]


def test_kernel_json_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = {"ready": True, "checks": {}}
    monkeypatch.setattr(kernel, "_preflight", lambda *args, **kwargs: report)
    assert kernel.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out) == report


def test_console_scripts_are_packaged() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    assert project["project"]["scripts"] == {
        "rapids-singlecell-install-skills": "rapids_singlecell_skills.install:main",
        "rapids-singlecell-check-kernel": "rapids_singlecell_skills.kernel:main",
    }
    assert (
        "src/rapids_singlecell_skills"
        in project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    )
