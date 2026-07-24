"""Query a live, provenance-aware RAPIDS-singlecell API view."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import platform
import re
import subprocess
import sys
import textwrap
import tomllib
from importlib import resources
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import unquote, urlparse

_DISTRIBUTION = "rapids-singlecell"
_MAX_RESULTS = 25
_MIN_SCORE = 8
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NAME_RE = re.compile(r"^[A-Za-z_]\w*$")
_SYMBOL_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")
_SPHINX_ROLE_RE = re.compile(
    r":(?P<role>[A-Za-z0-9_]+)(?::[A-Za-z0-9_]+)?:`(?P<value>[^`]+)`"
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "can",
    "cell",
    "cells",
    "compute",
    "data",
    "dataset",
    "for",
    "find",
    "from",
    "how",
    "in",
    "my",
    "of",
    "on",
    "or",
    "perform",
    "please",
    "run",
    "should",
    "single",
    "the",
    "to",
    "use",
    "using",
    "with",
}
_ENTRY_KEYS = {"index", "notes"}
_INDEX_KEYS = {"keywords", "related"}
_NOTE_KEYS = {"as_of", "claim", "kind", "source"}


class ApiError(RuntimeError):
    """Report an unavailable runtime or invalid hand-authored layer."""


class QueryError(ValueError):
    """Report an invalid search or description request."""


def _resource_path(name: str) -> Path:
    resource = resources.files("rapids_singlecell_skills").joinpath(name)
    return Path(str(resource))


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ApiError(f"{field} must be a list of non-empty strings")
    if len(set(value)) != len(value):
        raise ApiError(f"{field} contains duplicate values")
    return value


def _load_hand_layer() -> dict[str, dict[str, Any]]:
    path = _resource_path("api_notes.toml")
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ApiError(f"cannot read API hand layer at {path}: {error}") from error

    if set(payload) != {"entries", "schema"} or payload.get("schema") != 1:
        raise ApiError(f"unsupported API hand-layer schema at {path}")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ApiError("API hand layer must contain an entries table")

    for symbol, entry in entries.items():
        if not isinstance(symbol, str) or not _SYMBOL_RE.fullmatch(symbol):
            raise ApiError(f"invalid hand-layer symbol: {symbol!r}")
        if not isinstance(entry, dict) or not entry or not set(entry) <= _ENTRY_KEYS:
            raise ApiError(f"invalid hand-layer fields for {symbol}")

        if "index" in entry:
            index = entry["index"]
            if not isinstance(index, dict) or set(index) != _INDEX_KEYS:
                raise ApiError(f"{symbol}.index must contain keywords and related")
            _string_list(index["keywords"], field=f"{symbol}.index.keywords")
            related = _string_list(index["related"], field=f"{symbol}.index.related")
            if any(not _SYMBOL_RE.fullmatch(item) for item in related):
                raise ApiError(f"{symbol}.index.related contains an invalid symbol")

        notes = entry.get("notes", [])
        if not isinstance(notes, list):
            raise ApiError(f"{symbol}.notes must be an array of tables")
        for position, note in enumerate(notes):
            label = f"{symbol}.notes[{position}]"
            if not isinstance(note, dict) or not set(note) <= _NOTE_KEYS:
                raise ApiError(f"{label} contains invalid fields")
            kind = note.get("kind")
            if kind == "runtime":
                raise ApiError(f"{label}: kind='runtime' is forbidden; write a probe")
            if kind not in {"durable", "snapshot"}:
                raise ApiError(f"{label}.kind must be durable or snapshot")
            for field in ("claim", "source"):
                value = note.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ApiError(f"{label}.{field} must be a non-empty string")
            source = " ".join(note["source"].casefold().split())
            if source in {
                "a model",
                "language model",
                "language model memory",
                "llm",
                "llm memory",
                "model",
                "model memory",
                "model's memory",
                "the model",
                "the model's memory",
            }:
                raise ApiError(f"{label}.source cannot cite model memory")
            as_of = note.get("as_of")
            if kind == "snapshot" and not as_of:
                raise ApiError(f"{label}: snapshot notes require as_of")
            if as_of is not None and (
                not isinstance(as_of, dict)
                or not as_of
                or any(
                    not isinstance(key, str) or not isinstance(value, str) or not value
                    for key, value in as_of.items()
                )
            ):
                raise ApiError(f"{label}.as_of must map packages to versions")
    return entries


def _load_rsc() -> Any:
    try:
        return importlib.import_module("rapids_singlecell")
    except Exception as error:
        raise ApiError(
            "cannot import the active rapids_singlecell package; activate its "
            "environment and run rapids-singlecell-check-kernel first: "
            f"{type(error).__name__}: {error}"
        ) from error


def _distribution_version(module_name: str) -> tuple[str, str] | None:
    distributions = importlib.metadata.packages_distributions().get(module_name, ())
    found: list[tuple[str, str]] = []
    for distribution in distributions:
        try:
            found.append((distribution, importlib.metadata.version(distribution)))
        except importlib.metadata.PackageNotFoundError:
            continue
    if not found:
        return None
    return sorted(found)[0]


def _editable_source_root(module: Any) -> Path | None:
    source = Path(module.__file__).resolve()
    distributions = importlib.metadata.packages_distributions().get(
        "rapids_singlecell", ()
    )
    for name in sorted(distributions):
        try:
            direct_url = importlib.metadata.distribution(name).read_text(
                "direct_url.json"
            )
        except importlib.metadata.PackageNotFoundError:
            continue
        if direct_url is None:
            continue
        try:
            payload = json.loads(direct_url)
            parsed = urlparse(payload["url"])
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        if payload.get("dir_info", {}).get("editable") is not True:
            continue
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            continue
        root = Path(unquote(parsed.path)).resolve()
        if source.is_relative_to(root):
            return root

    for repository in source.parents:
        expected = repository / "src" / "rapids_singlecell"
        if (repository / ".git").exists() and expected.resolve() == source.parent:
            return repository
    return None


def _source_state(module: Any) -> tuple[str, bool] | None:
    repository = _editable_source_root(module)
    if repository is None:
        return None
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = revision.stdout.strip()
    return (value, bool(status.stdout.strip())) if value else None


def _built_against(rsc: Any) -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "rsc": str(getattr(rsc, "__version__", "unknown")),
    }
    if distribution := _distribution_version("rapids_singlecell"):
        versions["rsc_distribution"] = f"{distribution[0]}=={distribution[1]}"
    if source_state := _source_state(rsc):
        versions["rsc_source_editable"] = True
        versions["rsc_source_revision"] = source_state[0]
        versions["rsc_source_dirty"] = source_state[1]

    for label, module_name in (
        ("anndata", "anndata"),
        ("cuml", "cuml"),
        ("cupy", "cupy"),
        ("cuvs", "cuvs"),
        ("dask", "dask"),
        ("rmm", "rmm"),
        ("scanpy", "scanpy"),
    ):
        if distribution := _distribution_version(module_name):
            versions[label] = distribution[1]
            continue
        loaded = sys.modules.get(module_name)
        version = getattr(loaded, "__version__", None) if loaded is not None else None
        if version is not None:
            versions[label] = str(version)
    return versions


def _freshness_diagnostics(built_against: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    distribution = built_against.get("rsc_distribution")
    runtime = built_against.get("rsc")
    if isinstance(distribution, str) and "==" in distribution:
        distribution_version = distribution.split("==", maxsplit=1)[1]
        if runtime != distribution_version:
            diagnostics.append(
                {
                    "kind": "generated",
                    "severity": "warning",
                    "message": (
                        f"imported RSC reports {runtime}, but installed distribution "
                        f"metadata reports {distribution_version}"
                    ),
                }
            )
    if built_against.get("rsc_source_dirty") is True:
        diagnostics.append(
            {
                "kind": "generated",
                "severity": "warning",
                "message": (
                    "the imported RSC source tree is dirty; its revision alone does "
                    "not identify the observed code"
                ),
            }
        )
    return diagnostics


def _exports(module: Any, *, label: str) -> tuple[str, ...]:
    exported = getattr(module, "__all__", None)
    if not isinstance(exported, (list, tuple)) or not exported:
        raise ApiError(f"{label} must publish a non-empty __all__ contract")
    if any(not isinstance(name, str) or not name for name in exported):
        raise ApiError(f"{label}.__all__ must contain only non-empty strings")
    if len(set(exported)) != len(exported):
        raise ApiError(f"{label}.__all__ contains duplicate exports")
    return tuple(exported)


def _facades(rsc: Any) -> list[tuple[str, ModuleType]]:
    facades: list[tuple[str, ModuleType]] = []
    for name in _exports(rsc, label="rapids_singlecell"):
        value = getattr(rsc, name, None)
        if not isinstance(value, ModuleType):
            raise ApiError(f"invalid rapids_singlecell.__all__ entry: {name!r}")
        _exports(value, label=f"rsc.{name}")
        facades.append((name, value))
    return sorted(facades)


def _resolve(rsc: Any, symbol: str) -> Any:
    value = rsc
    try:
        for part in symbol.split("."):
            if part.startswith("_"):
                raise AttributeError(part)
            value = getattr(value, part)
    except AttributeError as error:
        raise QueryError(f"unknown live RSC symbol: {symbol}") from error
    if not callable(value):
        raise QueryError(f"RSC symbol is not callable: {symbol}")
    return value


def _kind(value: Any) -> str:
    if inspect.isclass(value):
        return "class"
    if inspect.ismethod(value):
        return "method"
    if inspect.isfunction(value):
        return "function"
    return "callable"


_ADDRESS_RE = re.compile(r" at 0x[0-9a-fA-F]+>")


def _stable_text(value: str) -> str:
    value = value.replace("<object object", "<omitted")
    return _ADDRESS_RE.sub(">", value)


def _annotation(value: Any) -> str:
    if value is inspect.Parameter.empty:
        return "unspecified"
    if isinstance(value, str):
        return value
    return _stable_text(inspect.formatannotation(value))


def _default(parameter: inspect.Parameter) -> str:
    if parameter.kind in {
        inspect.Parameter.VAR_KEYWORD,
        inspect.Parameter.VAR_POSITIONAL,
    }:
        return "not applicable"
    if parameter.default is inspect.Parameter.empty:
        return "required"
    return _stable_text(repr(parameter.default))


def _interface(
    rsc: Any, symbol: str, value: Any
) -> tuple[str, inspect.Signature, dict[str, Any]]:
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError) as error:
        raise ApiError(f"cannot inspect signature for {symbol}: {error}") from error
    parts = symbol.split(".")
    if len(parts) != 3:
        return _kind(value), signature, {}

    owner_symbol = ".".join(parts[:-1])
    owner = _resolve(rsc, owner_symbol)
    descriptor = inspect.getattr_static(owner, parts[-1])
    metadata = {"call": f"rsc.{symbol}", "owner": owner_symbol}
    if isinstance(descriptor, staticmethod):
        return "staticmethod", signature, metadata
    if isinstance(descriptor, classmethod):
        return "classmethod", signature, metadata

    parameters = list(signature.parameters.values())
    if not parameters or parameters[0].name not in {"self", "cls"}:
        raise ApiError(f"cannot identify the receiver for live method: {symbol}")
    signature = signature.replace(parameters=parameters[1:])
    class_name = parts[-2]
    instance = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
    constructor = inspect.signature(owner)
    required = [
        parameter.name
        for parameter in constructor.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}
    ]
    if required:
        instantiate = {
            "owner": owner_symbol,
            "required_parameters": ", ".join(required),
        }
    else:
        instantiate = {"call": f"{instance} = rsc.{owner_symbol}()"}
    metadata.update(
        {
            "call": f"{instance}.{parts[-1]}",
            "instantiate": instantiate,
        }
    )
    return "method", signature, metadata


def _summary(value: Any) -> str:
    doc = inspect.getdoc(value) or ""
    lines = doc.splitlines()
    stop = len(lines)
    for index in range(len(lines) - 1):
        underline = lines[index + 1].strip()
        if lines[index].strip() and len(underline) >= 3 and set(underline) == {"-"}:
            stop = index
            break
    first = "\n".join(lines[:stop]).split("\n\n", maxsplit=1)[0]
    return " ".join(first.split())


def _generated_record(
    rsc: Any,
    symbol: str,
    value: Any,
    *,
    aliases: list[str] | None = None,
) -> dict[str, Any]:
    kind, signature, interface = _interface(rsc, symbol, value)
    record: dict[str, Any] = {
        "aliases": aliases or [],
        "deprecated": getattr(value, "__deprecated__", None),
        "kind": kind,
        "module": getattr(value, "__module__", None) or type(value).__module__,
        "params": [parameter.name for parameter in signature.parameters.values()],
        "signature": _stable_text(str(signature)),
        "summary": _summary(value),
    }
    record.update(interface)
    return record


def _contract(rsc: Any) -> dict[str, Any]:
    contract: dict[str, Any] = {}
    for namespace, module in _facades(rsc):
        exported = _exports(module, label=f"rsc.{namespace}")
        public: dict[str, Any] = {}
        for name in exported:
            value = getattr(module, name, None)
            if name.startswith("_") or not callable(value):
                raise ApiError(f"invalid rsc.{namespace}.__all__ entry: {name!r}")
            public[name] = value
            contract[f"{namespace}.{name}"] = value

        class_names = {name for name, value in public.items() if inspect.isclass(value)}
        class_members = getattr(module, "__api_members__", {})
        if not isinstance(class_members, dict) or set(class_members) != class_names:
            raise ApiError(
                f"rsc.{namespace}.__api_members__ must exactly cover exported classes"
            )
        for class_name, members in class_members.items():
            _string_list(members, field=f"rsc.{namespace}.{class_name} members")
            owner = public[class_name]
            for name in members:
                try:
                    value = getattr(owner, name)
                except AttributeError as error:
                    raise ApiError(
                        "public member contract references absent method: "
                        f"{namespace}.{class_name}.{name}"
                    ) from error
                if not callable(value):
                    raise ApiError(
                        "public member contract is not callable: "
                        f"{namespace}.{class_name}.{name}"
                    )
                contract[f"{namespace}.{class_name}.{name}"] = value
    return dict(sorted(contract.items()))


def _deprecated_contract(rsc: Any) -> dict[str, str]:
    deprecated: dict[str, str] = {}
    for namespace, module in _facades(rsc):
        mapping = getattr(module, "__deprecated_exports__", {})
        if not isinstance(mapping, dict):
            raise ApiError(f"rsc.{namespace}.__deprecated_exports__ must be a mapping")
        public = set(_exports(module, label=f"rsc.{namespace}"))
        overlap = public & set(mapping)
        if overlap:
            raise ApiError(
                f"rsc.{namespace} exports deprecated names as public: "
                + ", ".join(sorted(overlap))
            )
        for name, message in mapping.items():
            if (
                not isinstance(name, str)
                or not _NAME_RE.fullmatch(name)
                or not isinstance(message, str)
                or not message.strip()
                or not callable(getattr(module, name, None))
            ):
                raise ApiError(
                    f"invalid rsc.{namespace}.__deprecated_exports__ entry: {name!r}"
                )
            deprecated[f"{namespace}.{name}"] = message.strip()
    return dict(sorted(deprecated.items()))


def _surface(rsc: Any, contract: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[int, list[tuple[str, Any]]] = {}
    methods: list[tuple[str, Any]] = []
    for symbol, value in contract.items():
        if symbol.count(".") == 1:
            grouped.setdefault(id(value), []).append((symbol, value))
        else:
            methods.append((symbol, value))

    records: list[dict[str, Any]] = []
    for candidates in grouped.values():
        candidates.sort(key=lambda item: item[0])
        symbol, value = candidates[0]
        aliases = [candidate for candidate, _ in candidates[1:]]
        records.append(
            {
                "symbol": symbol,
                "value": value,
                "generated": _generated_record(rsc, symbol, value, aliases=aliases),
            }
        )

    for symbol, value in methods:
        records.append(
            {
                "symbol": symbol,
                "value": value,
                "generated": _generated_record(rsc, symbol, value),
            }
        )
    return sorted(records, key=lambda item: item["symbol"])


def _normalize_symbol(requested: str) -> str:
    normalized = requested.strip()
    for prefix in ("rapids_singlecell.", "rapids-singlecell.", "rsc."):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix)
            break
    if not (_NAME_RE.fullmatch(normalized) or _SYMBOL_RE.fullmatch(normalized)):
        raise QueryError(f"invalid RSC symbol: {requested!r}")
    return normalized


def _deprecated_match(
    deprecated: dict[str, str], requested: str
) -> tuple[str, str] | None:
    try:
        normalized = _normalize_symbol(requested)
    except QueryError:
        return None
    if normalized in deprecated:
        return normalized, deprecated[normalized]
    if "." not in normalized:
        matches = [
            (symbol, message)
            for symbol, message in deprecated.items()
            if symbol.rsplit(".", maxsplit=1)[-1] == normalized
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def _contract_lookup(
    rsc: Any,
    contract: dict[str, Any],
    deprecated: dict[str, str],
    requested: str,
) -> dict[str, Any]:
    normalized = _normalize_symbol(requested)
    if "." not in normalized:
        matches = [
            symbol
            for symbol in contract
            if symbol.rsplit(".", maxsplit=1)[-1] == normalized
        ]
        if len(matches) == 1:
            normalized = matches[0]
        elif len(matches) > 1:
            choices = ", ".join(matches)
            raise QueryError(
                f"ambiguous symbol {requested!r}; choose one of: {choices}"
            )
    if normalized not in contract:
        if match := _deprecated_match(deprecated, normalized):
            symbol, message = match
            raise QueryError(
                f"rsc.{symbol} is deprecated and excluded from the public "
                f"contract: {message}"
            )
        raise QueryError(f"symbol is not in RSC's live public contract: {requested}")

    value = contract[normalized]
    aliases = (
        sorted(
            symbol
            for symbol, candidate in contract.items()
            if symbol != normalized and symbol.count(".") == 1 and candidate is value
        )
        if normalized.count(".") == 1
        else []
    )
    return {
        "symbol": normalized,
        "value": value,
        "generated": _generated_record(rsc, normalized, value, aliases=aliases),
    }


def _validate_hand_symbols(
    entries: dict[str, dict[str, Any]], contract: dict[str, Any]
) -> None:
    available = set(contract)
    for symbol, entry in entries.items():
        if symbol not in available:
            raise ApiError(f"hand layer references an absent live symbol: {symbol}")
        for related in entry.get("index", {}).get("related", ()):
            if related not in available:
                raise ApiError(
                    f"hand layer relation is absent from the live API: {symbol} -> {related}"
                )


def _hand_entry(
    entries: dict[str, dict[str, Any]], surface_entry: dict[str, Any]
) -> dict[str, Any]:
    candidates = [surface_entry["symbol"], *surface_entry["generated"]["aliases"]]
    return next((entries[item] for item in candidates if item in entries), {})


def _query_tokens(value: str) -> tuple[str, ...]:
    tokens = [
        token
        for token in _TOKEN_RE.findall(value.casefold())
        if token not in _STOPWORDS
    ]
    if not tokens:
        raise QueryError("search query must contain an informative word")
    return tuple(tokens)


def _search_score(
    entry: dict[str, Any], hand_entry: dict[str, Any], query: str
) -> int | None:
    query_tokens = _query_tokens(query)
    generated = entry["generated"]
    symbol = entry["symbol"].casefold().replace("_", " ")
    basename = symbol.rsplit(".", maxsplit=1)[-1]
    aliases = " ".join(generated["aliases"]).casefold().replace("_", " ")
    keywords = " ".join(hand_entry.get("index", {}).get("keywords", ())).casefold()
    summary = generated["summary"].casefold()
    signature = generated["signature"].casefold()
    corpus = set(
        _TOKEN_RE.findall(" ".join((symbol, aliases, keywords, summary, signature)))
    )
    matched_tokens = [token for token in query_tokens if token in corpus]
    required_matches = (
        1
        if len(query_tokens) == 1 or any(token == basename for token in query_tokens)
        else 2
    )
    if len(matched_tokens) < required_matches:
        return None

    normalized_query = " ".join(query_tokens)
    score = 0
    if normalized_query == basename:
        score += 100
    if normalized_query in symbol:
        score += 40
    if normalized_query in keywords:
        score += 60
    if normalized_query in summary:
        score += 20
    for token in matched_tokens:
        score += 12 if token in basename else 0
        score += 8 if token in keywords else 0
        score += 4 if token in summary else 0
        score += 2 if token in signature else 0
    score -= 2 * (len(query_tokens) - len(matched_tokens))
    return score


def search(
    surface: list[dict[str, Any]],
    entries: dict[str, dict[str, Any]],
    query: str,
    *,
    namespace: str | None = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Return deterministic matches built from the live surface and sparse index."""
    if not 1 <= limit <= _MAX_RESULTS:
        raise QueryError(f"limit must be between 1 and {_MAX_RESULTS}")
    namespaces = {entry["symbol"].split(".", maxsplit=1)[0] for entry in surface}
    if namespace is not None and namespace not in namespaces:
        raise QueryError(
            f"unknown live namespace {namespace!r}; choose one of: "
            + ", ".join(sorted(namespaces))
        )

    matches: list[tuple[int, dict[str, Any]]] = []
    prefix = f"{namespace}." if namespace else None
    for entry in surface:
        selected = entry
        if prefix:
            paths = [entry["symbol"], *entry["generated"]["aliases"]]
            matching = [path for path in paths if path.startswith(prefix)]
            if not matching:
                continue
            symbol = matching[0]
            if symbol != entry["symbol"]:
                selected = {
                    **entry,
                    "generated": dict(entry["generated"]),
                    "symbol": symbol,
                }
                selected["generated"]["aliases"] = [
                    path for path in paths if path != symbol
                ]
        if selected["generated"]["deprecated"]:
            continue
        score = _search_score(selected, _hand_entry(entries, selected), query)
        if score is not None and score >= _MIN_SCORE:
            matches.append((score, selected))
    matches.sort(key=lambda item: (-item[0], item[1]["symbol"]))
    return [entry for _, entry in matches[:limit]]


def _split_doc(doc: str) -> dict[str, str]:
    lines = doc.splitlines()
    headers: list[tuple[int, str]] = []
    for index in range(len(lines) - 1):
        underline = lines[index + 1].strip()
        if lines[index].strip() and len(underline) >= 3 and set(underline) == {"-"}:
            headers.append((index, lines[index].strip()))
    sections: dict[str, str] = {}
    for position, (index, name) in enumerate(headers):
        stop = headers[position + 1][0] if position + 1 < len(headers) else len(lines)
        body = lines[index + 2 : stop]
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        sections[name] = "\n".join(body)
    return sections


def _parameter_docs(section: str) -> dict[str, str]:
    lines = section.splitlines()
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if not indents:
        return {}
    header_indent = min(indents)
    headers = [
        index
        for index, line in enumerate(lines)
        if line.strip()
        and len(line) - len(line.lstrip()) == header_indent
        and not line.lstrip().startswith(("-", "*", ":"))
    ]
    descriptions: dict[str, str] = {}
    for position, index in enumerate(headers):
        stop = headers[position + 1] if position + 1 < len(headers) else len(lines)
        names = lines[index].strip().split(":", maxsplit=1)[0]
        description = " ".join(" ".join(lines[index + 1 : stop]).split())
        for name in names.split(","):
            descriptions[name.strip().removeprefix("*")] = description
    return descriptions


def _parameter_records(
    signature: inspect.Signature, sections: dict[str, str]
) -> list[dict[str, str]]:
    docs = _parameter_docs(sections.get("Parameters", ""))
    records = [
        {
            "annotation": _annotation(parameter.annotation),
            "default": _default(parameter),
            "description": docs.get(parameter.name.removeprefix("*"), "undocumented"),
            "kind": parameter.kind.name.casefold().replace("_", "-"),
            "name": parameter.name,
        }
        for parameter in signature.parameters.values()
    ]
    names = {record["name"].removeprefix("*") for record in records}
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        records.extend(
            {
                "annotation": "unspecified",
                "default": "not exposed by signature",
                "description": description,
                "kind": "forwarded-keyword",
                "name": name,
            }
            for name, description in docs.items()
            if name not in names
        )
    return records


def _merge_notes(
    hand_entry: dict[str, Any], built_against: dict[str, Any]
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    freshness = _freshness_diagnostics(built_against)
    for note in hand_entry.get("notes", ()):
        item = dict(note)
        if note["kind"] == "snapshot":
            mismatches = {
                package: {"expected": version, "observed": built_against.get(package)}
                for package, version in note["as_of"].items()
                if built_against.get(package) != version
            }
            item["status"] = "suspect" if mismatches or freshness else "current"
            if mismatches:
                item["mismatches"] = mismatches
            if freshness:
                item["freshness_diagnostics"] = [
                    diagnostic["message"] for diagnostic in freshness
                ]
        else:
            item["status"] = "cited"
        merged.append(item)
    return merged


def _merged_view(
    rsc: Any,
    surface_entry: dict[str, Any],
    hand_entry: dict[str, Any],
    built_against: dict[str, Any],
    *,
    parameter: str | None = None,
    section: str | None = None,
    full: bool = False,
    probed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    symbol = surface_entry["symbol"]
    value = surface_entry["value"]
    kind, signature, _ = _interface(rsc, symbol, value)
    doc = inspect.getdoc(value) or ""
    sections = _split_doc(doc)
    parameters = _parameter_records(signature, sections)
    generated = dict(surface_entry["generated"])
    generated["kind"] = kind
    generated["params"] = [item["name"] for item in parameters]
    generated["sections"] = sorted(sections)

    if parameter is not None:
        requested = parameter.removeprefix("*")
        match = next(
            (
                item
                for item in parameters
                if item["name"].removeprefix("*") == requested
            ),
            None,
        )
        if match is None:
            available = ", ".join(item["name"] for item in parameters)
            raise QueryError(
                f"{symbol} has no parameter {parameter!r}; available: {available}"
            )
        generated["parameter"] = match
    elif section is not None:
        match = next(
            (name for name in sections if name.casefold() == section.casefold()), None
        )
        if match is None:
            raise QueryError(
                f"{symbol} has no section {section!r}; available: "
                + ", ".join(sorted(sections))
            )
        generated["section"] = {"name": match, "body": sections[match]}
    elif full:
        generated["doc"] = doc

    index = hand_entry.get("index", {"keywords": [], "related": []})
    return {
        "built_against": built_against,
        "diagnostics": _freshness_diagnostics(built_against),
        "generated": generated,
        "index": index,
        "notes": _merge_notes(hand_entry, built_against),
        "probed": probed,
        "symbol": symbol,
    }


def _run_probe(symbol: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": (
            f"no safe tiny-input probe is registered for {symbol}; verify the "
            "behavior against the active package rather than inferring it"
        ),
    }


def _short_summary(entry: dict[str, Any]) -> str:
    value = entry["generated"]["summary"] or "No summary documented."
    value = _display_text(value)
    return textwrap.shorten(value, width=160, placeholder=" …")


def _display_text(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        if match.group("role") == "cite":
            return ""
        target = match.group("value").strip().lstrip("~")
        return target.split("<", maxsplit=1)[0].strip()

    rendered = " ".join(_SPHINX_ROLE_RE.sub(replace, value).split())
    return re.sub(r"\s+([,.;:])", r"\1", rendered)


def _print_search(
    built_against: dict[str, Any],
    query: str,
    entries: list[dict[str, Any]],
    *,
    miss_message: str | None = None,
) -> None:
    print(f"built against: rsc={built_against['rsc']}")
    _print_diagnostics(_freshness_diagnostics(built_against))
    print(f"query: {query}")
    if not entries:
        print(
            miss_message
            or "no exact matches; this is a lexical miss, not evidence that RSC "
            "lacks the API; retry with fewer or API-specific terms"
        )
        return
    for entry in entries:
        aliases = entry["generated"]["aliases"]
        suffix = (
            " (aliases: " + ", ".join(f"rsc.{alias}" for alias in aliases) + ")"
            if aliases
            else ""
        )
        print(f"rsc.{entry['symbol']}{suffix} — {_short_summary(entry)}")


def _print_diagnostics(diagnostics: list[dict[str, str]]) -> None:
    for diagnostic in diagnostics:
        print(f"{diagnostic['severity']}: {diagnostic['message']}")


def _print_context(view: dict[str, Any], *, include_index: bool) -> None:
    if include_index:
        index = view["index"]
        if index["keywords"]:
            print("keywords: " + ", ".join(index["keywords"]))
        if index["related"]:
            print("related: " + ", ".join(f"rsc.{item}" for item in index["related"]))
    for note in view["notes"]:
        print(f"{note['kind']} note [{note['status']}]: {note['claim']}")
        print(f"  source: {note['source']}")
        if as_of := note.get("as_of"):
            print("  as_of: " + json.dumps(as_of, sort_keys=True))
        if mismatches := note.get("mismatches"):
            print("  mismatches: " + json.dumps(mismatches, sort_keys=True))
    if view["probed"] is not None:
        print("probe: " + json.dumps(view["probed"], sort_keys=True))


def _print_view(view: dict[str, Any]) -> None:
    generated = view["generated"]
    built = view["built_against"]
    print(f"built against: rsc={built['rsc']}")
    if revision := built.get("rsc_source_revision"):
        print(f"source revision: {revision}")
    _print_diagnostics(view["diagnostics"])
    print(f"symbol: rsc.{view['symbol']}")

    parameter = generated.get("parameter")
    section = generated.get("section")
    if parameter is not None:
        for key in ("name", "kind", "annotation", "default", "description"):
            value = parameter[key]
            if key == "description":
                value = _display_text(value)
            print(f"{key}: {value}")
    elif section is not None:
        print(f"section: {section['name']}")
        print(section["body"])
    else:
        if instantiate := generated.get("instantiate"):
            if call := instantiate.get("call"):
                print(f"instantiate: {call}")
            else:
                print(
                    "instantiate: describe rsc."
                    f"{instantiate['owner']} first; required parameters: "
                    f"{instantiate['required_parameters']}"
                )
        call = generated.get("call", f"rsc.{view['symbol']}")
        print(f"signature: {call}{generated['signature']}")
        if "doc" in generated:
            print(generated["doc"])
        else:
            print(f"kind: {generated['kind']}")
            summary = generated["summary"] or "Undocumented."
            print(f"summary: {_display_text(summary)}")
            print("parameters: " + ", ".join(generated["params"]))
            if deprecated := generated.get("deprecated"):
                print(f"deprecated: {deprecated}")

    detail = parameter is not None or section is not None or "doc" in generated
    _print_context(view, include_index=not detail)
    if not detail:
        print("details: use --parameter NAME, --section NAME, --full, or --probe")


def _print_json(payload: dict[str, Any]) -> None:
    print(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="search the live RSC API")
    search_parser.add_argument("query", nargs="+")
    search_parser.add_argument("--namespace")
    search_parser.add_argument("--limit", type=int, default=8)
    search_parser.add_argument("--json", action="store_true")

    describe_parser = subparsers.add_parser(
        "describe", help="build one ephemeral merged API view"
    )
    describe_parser.add_argument("symbol")
    detail = describe_parser.add_mutually_exclusive_group()
    detail.add_argument("--parameter")
    detail.add_argument("--section")
    detail.add_argument("--full", action="store_true")
    describe_parser.add_argument("--probe", action="store_true")
    describe_parser.add_argument("--json", action="store_true")

    validate_parser = subparsers.add_parser(
        "validate", help="validate the hand layer against the live API"
    )
    validate_parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run live discovery and build per-symbol ephemeral merged views."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        entries = _load_hand_layer()
        rsc = _load_rsc()
        contract = _contract(rsc)
        deprecated = _deprecated_contract(rsc)
        _validate_hand_symbols(entries, contract)
        built_against = _built_against(rsc)

        if args.command == "validate":
            surface = _surface(rsc, contract)
            payload = {
                "annotated_symbols": len(entries),
                "built_against": built_against,
                "deprecated_symbols": len(deprecated),
                "diagnostics": _freshness_diagnostics(built_against),
                "live_symbols": len(surface),
                "status": "valid",
            }
            if args.json:
                _print_json(payload)
            else:
                print(
                    f"hand layer is valid for {len(surface)} live symbols; "
                    f"{len(entries)} have sparse annotations"
                )
                _print_diagnostics(_freshness_diagnostics(built_against))
            return 0

        if args.command == "search":
            surface = _surface(rsc, contract)
            query = " ".join(args.query)
            matches = search(
                surface,
                entries,
                query,
                namespace=args.namespace,
                limit=args.limit,
            )
            deprecated_match = (
                _deprecated_match(deprecated, query) if not matches else None
            )
            miss_message = None
            if deprecated_match is not None:
                symbol, message = deprecated_match
                miss_message = (
                    f"rsc.{symbol} is deprecated and excluded from the public "
                    f"contract: {message}"
                )
            if args.json:
                payload: dict[str, Any] = {
                    "built_against": built_against,
                    "diagnostics": _freshness_diagnostics(built_against),
                    "query": query,
                    "results": [
                        {
                            "aliases": [
                                f"rsc.{alias}"
                                for alias in entry["generated"]["aliases"]
                            ],
                            "deprecated": entry["generated"]["deprecated"],
                            "summary": _short_summary(entry),
                            "symbol": f"rsc.{entry['symbol']}",
                        }
                        for entry in matches
                    ],
                }
                if not matches:
                    payload["warning"] = miss_message or (
                        "lexical miss; this is not evidence that RSC lacks the API"
                    )
                _print_json(payload)
            else:
                _print_search(
                    built_against,
                    query,
                    matches,
                    miss_message=miss_message,
                )
            return 0

        surface_entry = _contract_lookup(rsc, contract, deprecated, args.symbol)
        hand_entry = _hand_entry(entries, surface_entry)
        probed = _run_probe(surface_entry["symbol"]) if args.probe else None
        view = _merged_view(
            rsc,
            surface_entry,
            hand_entry,
            built_against,
            parameter=args.parameter,
            section=args.section,
            full=args.full,
            probed=probed,
        )
        if args.json:
            _print_json(view)
        else:
            _print_view(view)
        return 0
    except QueryError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except ApiError as error:
        print(f"error: {error}", file=sys.stderr)
        return 3


__all__ = ["main", "search"]


if __name__ == "__main__":
    raise SystemExit(main())
