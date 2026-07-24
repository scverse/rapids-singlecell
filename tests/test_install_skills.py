from __future__ import annotations

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


def test_skill_bundle_is_minimal() -> None:
    source = install.skill_source()
    assert source == (
        Path(install.__file__).resolve().parent / "data" / "rapids-singlecell"
    )
    files = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
    }
    assert files == {
        "SKILL.md",
        "agents/openai.yaml",
        "references/dask.md",
        "references/memory.md",
        "references/notebooks.md",
        "references/setup.md",
        "references/spatial.md",
    }

    text = (source / "SKILL.md").read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 100
    assert len(text.split()) <= 850
    description = re.search(r"(?m)^description:\s*(.+)$", text)
    assert description is not None
    assert len(description.group(1).strip("\"'")) <= 200
    normalized_text = " ".join(text.split())
    for phrase in (
        "rapids-singlecell-check-kernel",
        "managed_memory=oversubscribe",
        "references/memory.md",
        "references/dask.md",
        "references/notebooks.md",
        "references/setup.md",
        "references/spatial.md",
        "executed `.ipynb`",
    ):
        assert phrase in normalized_text

    for name in ("memory.md", "dask.md", "notebooks.md", "setup.md", "spatial.md"):
        reference = (source / "references" / name).read_text(encoding="utf-8")
        assert len(reference.splitlines()) <= 100
        assert len(reference.split()) <= 900


def _markdown_bullets(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").casefold()
    return [
        " ".join(match.split())
        for match in re.findall(
            r"(?ms)^- (.*?)(?=^- |^## |\Z)",
            text,
        )
    ]


def _notebook_contract_bullets() -> list[str]:
    return _markdown_bullets(install.skill_source() / "references" / "notebooks.md")


def _spatial_contract_bullets() -> list[str]:
    return _markdown_bullets(install.skill_source() / "references" / "spatial.md")


def _covers(bullets: list[str], *terms: str) -> bool:
    return any(all(term.casefold() in bullet for term in terms) for bullet in bullets)


def test_core_preserves_cold_run_recovery_contract() -> None:
    bullets = _markdown_bullets(install.skill_source() / "SKILL.md")

    assert _covers(
        bullets,
        "_check_gpu_x",
        "cupy",
        "layer",
        "rsc.get.anndata_to_gpu",
        "convert_all=true",
        "transitional",
    )
    assert _covers(
        bullets,
        "squidpy",
        "physical graph",
        "live rsc cannot",
        "attribute",
        "references/spatial.md",
    )
    assert _covers(
        bullets,
        "critically evaluate every output",
        "group counts",
        "figure pixels",
        "megapixels",
        "stop signals",
        "rerun",
    )
    assert _covers(
        bullets,
        "requested analysis choices",
        "push back with evidence",
        "valid alternative",
    )
    assert _covers(
        bullets,
        "sc.pl",
        "sq.pl",
        "only",
        "canonical `decoupler` plotting does not express",
    )


def test_core_dispatches_pathway_work_by_analysis_unit() -> None:
    bullets = _markdown_bullets(install.skill_source() / "SKILL.md")

    assert _covers(
        bullets,
        "canonical",
        "decoupler",
        "resources",
        "pathway-native plots",
        "`rsc.dcg`",
        "per-cell scoring",
        "descriptively",
        "pseudobulk",
        "replicated cross-condition inference",
        "biological sample",
        "replication unit",
        "attribute each boundary",
    )
    assert _covers(
        bullets,
        "sc.pl",
        "sq.pl",
        "only",
        "canonical `decoupler` plotting does not express",
    )


def test_core_surfaces_neighboring_gpu_ports() -> None:
    bullets = _markdown_bullets(install.skill_source() / "SKILL.md")

    assert _covers(
        bullets,
        "`rsc.gr`",
        "squidpy-compatible",
        "`rsc.ptg`",
        "pertpy-compatible",
        "`rsc.dcg`",
        "decoupler-compatible",
        "ecosystem",
        "method names",
        "describe",
        "public symbol",
    )


def test_setup_api_discovery_degrades_without_helper() -> None:
    setup = (
        (install.skill_source() / "references" / "setup.md")
        .read_text(encoding="utf-8")
        .casefold()
    )
    discovery = " ".join(setup.split("## discover the live api", maxsplit=1)[1].split())

    for phrase in (
        "rapids_singlecell_skills.api search",
        "rapids_singlecell_skills.api describe",
        "unavailable",
        "inspect.signature",
        "inspect.getdoc",
        "help(call)",
        "official documentation",
        "implementation source",
        "license compatibility",
    ):
        assert phrase in discovery


def test_setup_bounds_kernel_less_execution_fallback() -> None:
    bullets = _markdown_bullets(install.skill_source() / "references" / "setup.md")

    assert _covers(
        bullets,
        "kernel-less",
        "preflight passes",
        "startup logs",
        "denied jupyter/zmq socket",
        "not diagnostic",
        "import",
        "abi",
        "gpu",
        "oom",
    )
    assert _covers(
        bullets,
        "tested in-process notebook executor",
        "fresh disposable child",
        "not the agent process",
        "in order",
        "stop on first error",
        "counts",
        "streams",
        "rich displays",
        "figures",
        "tracebacks",
        "unsupported magics",
        "blockers",
        "`cuda_visible_devices`",
        "before any cuda import",
        "persisted outputs",
    )


def test_notebook_reference_preserves_notebook_contract() -> None:
    bullets = _notebook_contract_bullets()

    assert _covers(bullets, "one", "analysis task", "code cell")
    assert _covers(bullets, "plot", "next visible cell", "interpretation")
    assert _covers(
        bullets,
        "owning package",
        "canonical",
        "decoupler",
        "pathways",
        "sc.pl",
        "sq.pl",
        "custom plotting",
    )
    assert _covers(
        bullets,
        "resolve unexpected warnings",
        "understood routine",
        "without errors",
        "raw log streams",
    )
    assert _covers(
        bullets,
        "scaffolding",
        "outside the narrative",
        "compact table",
        "validation cell",
    )
    assert _covers(bullets, "execute every cell", "fresh kernel")
    assert _covers(
        bullets,
        "tentative labels",
        "partition changes",
        "never reuse cluster-id maps",
        "complete coverage",
        "`unknown`",
        "exclusion",
        "contradictory",
        "confidence",
        "source links",
    )
    assert _covers(
        bullets,
        "analysis preferences",
        "live rsc",
        "visible",
        "standing default",
        "explicit request",
        "scope",
        "conditions",
    )
    assert _covers(
        bullets,
        "executed",
        ".ipynb",
        ".md",
        "findings",
        "evidence",
        "limitations",
    )
    assert _covers(
        bullets,
        "requested choice",
        "data or design",
        "evidence",
        "concern",
        "valid alternative",
        "never silently",
    )


def test_notebook_reference_preserves_scverse_data_flow() -> None:
    bullets = _notebook_contract_bullets()

    assert _covers(
        bullets,
        "session_info2",
        "seed",
        "input provenance",
        "source revision",
    )
    assert _covers(
        bullets,
        "standard log-normalized workflow",
        "counts",
        "hvgs",
        "full object",
        "subset",
    )
    assert _covers(
        bullets,
        "gpu call",
        "layer",
        "moving `x`",
        "every layer",
    )
    assert _covers(
        bullets,
        "rsc.get.x_to_cpu",
        "rsc.get.anndata_to_cpu",
        "interop",
        "host-backed",
    )
    assert _covers(
        bullets,
        "non-rsc",
        "essential",
        "relevant skill",
        "attribution",
        "fallback",
    )


def test_notebook_reference_routes_decoupler_work() -> None:
    bullets = _notebook_contract_bullets()

    assert _covers(
        bullets,
        "canonical `decoupler`",
        "resources",
        "pathway-native plots",
        "per-cell scoring",
        "live `rsc.dcg`",
        "single sample",
        "descriptive",
        "pseudobulk",
        "replicated cross-condition inference",
        "source counts",
        "biological sample",
        "decoupler skill",
        "attribute each boundary",
    )


def test_spatial_reference_bounds_large_spatial_plots() -> None:
    bullets = _spatial_contract_bullets()

    assert _covers(
        bullets,
        "figsize",
        "dpi",
        "rasterized=true",
        "background",
        "analytical data",
    )
    assert _covers(
        bullets,
        "render time",
        "pixel count",
        "file size",
        "stop",
        "rerender",
    )


def test_notebook_reference_has_ordered_workflow_outline() -> None:
    text = (
        (install.skill_source() / "references" / "notebooks.md")
        .read_text(encoding="utf-8")
        .casefold()
    )
    outline = text.split("## follow this scaffold", maxsplit=1)[1].split(
        "\n## ", maxsplit=1
    )[0]
    items = [
        " ".join(match.split())
        for match in re.findall(
            r"(?ms)^\d+\. (.*?)(?=^\d+\. |\Z)",
            outline,
        )
    ]

    assert len(items) >= 10
    assert any("markdown" in item for item in items)
    assert any("code" in item for item in items)

    stages = (
        ("question", "unit of replication", "design"),
        ("runtime", "provenance", "session_info2"),
        ("load", "inspect", "count location"),
        ("preserve", "gpu residency"),
        ("qc", "filter"),
        ("preprocess", "hvgs", "normalize"),
        ("structure", "graph", "leiden"),
        ("annotate", "marker evidence"),
        ("spatial", "niche"),
        ("render", "sanity-check"),
        ("export", "anndata", "findings report"),
    )
    indices = [
        next(
            index
            for index, item in enumerate(items)
            if all(term in item for term in terms)
        )
        for terms in stages
    ]
    assert indices == sorted(indices)
    structure = next(
        item
        for item in items
        if all(term in item for term in ("structure", "graph", "leiden"))
    )
    for term in (
        "pca and neighbors in float32",
        "`random_state`",
        '`dtype="float64"`',
        "observation order",
        "parameters",
        "package versions",
    ):
        assert term in structure


def test_api_index_finds_explicit_method_preferences() -> None:
    notes_path = ROOT / "src" / "rapids_singlecell_skills" / "api_notes.toml"
    with notes_path.open("rb") as handle:
        entries = tomllib.load(handle)["entries"]

    hvg_index = entries["pp.highly_variable_genes"]["index"]
    assert "poisson gene selection" in hvg_index["keywords"]

    dcg_index = entries["dcg.aucell"]["index"]
    assert "cell-level pathway activity" in dcg_index["keywords"]
    assert "decoupler" in dcg_index["keywords"]

    assert "squidpy" in entries["gr.calculate_niche"]["index"]["keywords"]
    assert "pertpy" in entries["ptg.Mixscape"]["index"]["keywords"]

    leiden = entries["tl.leiden"]
    assert "reproducible leiden" in leiden["index"]["keywords"]
    assert any(
        note["kind"] == "snapshot"
        and "dtype='float64'" in note["claim"]
        and "random_state" in note["claim"]
        and "input graph" in note["claim"]
        for note in leiden["notes"]
    )


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
