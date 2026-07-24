from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from importlib import resources
from pathlib import Path

SKILL_NAME = "rapids-singlecell"
_DEFAULT_ROOTS = {
    "codex": ("CODEX_HOME", "~/.codex"),
    "claude": ("CLAUDE_CONFIG_DIR", "~/.claude"),
    "agents": (None, "~/.agents"),
}


def _claude_science_parent() -> Path:
    root = Path("~/.claude-science").expanduser()
    active_org_path = root / "active-org.json"
    try:
        active_org = json.loads(active_org_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "cannot resolve the Claude Science active organization from "
            f"{active_org_path}; provide --dest"
        ) from error

    org_uuid = active_org.get("org_uuid") if isinstance(active_org, dict) else None
    if (
        not isinstance(org_uuid, str)
        or not org_uuid
        or Path(org_uuid).name != org_uuid
        or org_uuid in {".", ".."}
    ):
        raise ValueError(
            f"invalid Claude Science org_uuid in {active_org_path}; provide --dest"
        )
    return root / "orgs" / org_uuid / "skills"


def skill_source() -> Path:
    """Return the package-owned skill directory."""
    source = Path(
        str(
            resources.files("rapids_singlecell_skills")
            .joinpath("data")
            .joinpath(SKILL_NAME)
        )
    )
    if not (source / "SKILL.md").is_file():
        raise RuntimeError(f"packaged skill is missing: {source}")
    return source


def _default_parent(agent: str) -> Path:
    if agent == "claude-science":
        return _claude_science_parent()
    try:
        variable, fallback = _DEFAULT_ROOTS[agent]
    except KeyError as error:
        raise ValueError(
            f"agent {agent!r} has no default skill directory; provide --dest"
        ) from error
    root = os.environ.get(variable, fallback) if variable is not None else fallback
    return Path(root).expanduser() / "skills"


def _target(agent: str, destination: Path | None) -> Path:
    if destination is not None:
        return destination.expanduser()
    return _default_parent(agent) / SKILL_NAME


def _snapshot(root: Path) -> dict[Path, bytes | None]:
    if root.is_symlink():
        raise RuntimeError(f"refusing symlinked skill directory: {root}")
    if not root.is_dir():
        raise RuntimeError(f"skill path is not a directory: {root}")

    snapshot: dict[Path, bytes | None] = {}
    for item in root.rglob("*"):
        relative = item.relative_to(root)
        if item.is_symlink():
            raise RuntimeError(f"refusing symlink in skill directory: {item}")
        if item.is_dir():
            snapshot[relative] = None
        elif item.is_file():
            snapshot[relative] = item.read_bytes()
        else:
            raise RuntimeError(f"refusing special file in skill directory: {item}")
    return snapshot


def _matches(source: Path, target: Path) -> bool:
    return _snapshot(source) == _snapshot(target)


def _looks_like_existing_skill(target: Path) -> bool:
    skill_file = target / "SKILL.md"
    try:
        text = skill_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        text = ""
    if "name: rapids-singlecell" in text:
        return True
    anchors = (
        target / "agents" / "openai.yaml",
        target / "references" / "dask.md",
        target / "references" / "memory.md",
    )
    return skill_file.is_file() and all(path.is_file() for path in anchors)


def check_skill(
    agent: str = "codex", *, destination: Path | None = None
) -> tuple[bool, str]:
    """Check whether an agent copy exactly matches the package-owned skill."""
    source = skill_source()
    target = _target(agent, destination)
    if not target.exists() and not target.is_symlink():
        return False, f"skill is not installed at {target}"
    try:
        matches = _matches(source, target)
    except (OSError, RuntimeError) as error:
        return False, str(error)
    if not matches:
        return False, f"skill differs from the active package: {target}"
    return True, f"skill matches the active package: {target}"


def install_skill(
    agent: str = "codex",
    *,
    destination: Path | None = None,
    force: bool = False,
) -> Path:
    """Copy the package-owned skill into an agent's skill directory."""
    source = skill_source()
    target = _target(agent, destination)

    if target.is_symlink():
        raise RuntimeError(f"refusing to replace symlinked skill directory: {target}")
    if target.exists():
        if target.is_dir() and _matches(source, target):
            return target
        if not force:
            raise RuntimeError(f"destination differs; rerun with --force: {target}")
        if target.is_dir():
            if not any(target.iterdir()):
                target.rmdir()
            elif _looks_like_existing_skill(target):
                shutil.rmtree(target)
            else:
                raise RuntimeError(
                    "refusing --force for a directory that is not recognizably "
                    f"the {SKILL_NAME} skill: {target}"
                )
        else:
            raise RuntimeError(f"refusing --force for non-directory target: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rapids-singlecell-install-skills",
        description="Install the RAPIDS-singlecell skill bundled with this package.",
    )
    parser.add_argument("--agent", default="codex")
    parser.add_argument("--dest", type=Path, help="exact destination directory")
    parser.add_argument("--force", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true")
    action.add_argument("--print-path", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.print_path:
            print(skill_source())
            return 0
        if args.check:
            if args.force:
                parser.error("--check cannot be combined with --force")
            ok, message = check_skill(args.agent, destination=args.dest)
            print(message)
            return 0 if ok else 1
        target = install_skill(args.agent, destination=args.dest, force=args.force)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Installed RAPIDS-singlecell skill at {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
