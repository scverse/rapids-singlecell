"""Collect combined PyPI download numbers for all rapids-singlecell distributions.

Since the CUDA 12 / CUDA 13 split, downloads are spread over three PyPI
projects. Download counters like pepy only ever show one of them, which
undercounts the project by more than half. This script sums them up.

Numbers come from the public ClickHouse PyPI dataset (the one behind
clickpy.clickhouse.com), which needs no API key and matches the raw
PyPI download counts that pepy reports.

Outputs (into ``--output-dir``):
    downloads.json     full report, per project and combined
    badge-total.json   shields.io endpoint badge, all-time combined
    badge-month.json   shields.io endpoint badge, last 30 days combined
    history.csv        one row per project per day, appended over time
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

CLICKHOUSE_URL = (
    "https://sql-clickhouse.clickhouse.com/?user=demo&default_format=JSONEachRow"
)

PACKAGES = [
    "rapids-singlecell",
    "rapids-singlecell-cu12",
    "rapids-singlecell-cu13",
]

RECENT_DAYS = 30
RECENT_KEY = f"last_{RECENT_DAYS}_days"

# ClickHouse is the only source, so be patient before giving up: 5 attempts
# with 1s/2s/4s/8s backoff.
MAX_ATTEMPTS = 5


def warn(message: str) -> None:
    """Emit a message that GitHub Actions surfaces as a run annotation."""
    print(f"::warning::{message}")


def query(sql: str, *, attempts: int = MAX_ATTEMPTS, timeout: int = 60) -> list[dict]:
    """Run a read-only query against the public ClickHouse endpoint."""
    request = urllib.request.Request(
        CLICKHOUSE_URL,
        data=sql.encode(),
        headers={"User-Agent": "rapids-singlecell-download-stats"},
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode()
            rows = [json.loads(line) for line in body.splitlines() if line.strip()]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
        else:
            if rows and "exception" in rows[0]:
                # A query error will not fix itself; fail without retrying.
                raise RuntimeError(rows[0]["exception"])
            return rows
        if attempt < attempts - 1:
            time.sleep(2**attempt)
    raise RuntimeError(
        f"ClickHouse query failed after {attempts} attempts: {last_error}"
    )


def fetch_totals(projects: str) -> dict[str, int]:
    rows = query(
        f"SELECT project, sum(count) AS total FROM pypi.pypi_downloads "
        f"WHERE project IN ({projects}) GROUP BY project"
    )
    return {row["project"]: int(row["total"]) for row in rows}


def fetch_recent(projects: str) -> tuple[dict[str, int], str]:
    rows = query(
        f"SELECT project, sum(count) AS recent, max(date) AS latest "
        f"FROM pypi.pypi_downloads_per_day "
        f"WHERE project IN ({projects}) AND date > today() - {RECENT_DAYS} "
        f"GROUP BY project"
    )
    recent = {row["project"]: int(row["recent"]) for row in rows}
    # The dataset lags PyPI by roughly a day; report what it actually covers.
    return recent, max(row["latest"] for row in rows)


def previous_value(previous: dict | None, package: str, key: str) -> int:
    if previous is None:
        return 0
    return int(previous.get("packages", {}).get(package, {}).get(key, 0))


def collect(previous: dict | None = None) -> dict:
    """Fetch both figures, falling back to the last published one per figure.

    The two queries are independent, so a failure in one must not discard the
    other. Only if both fail do we give up, leaving the published numbers
    untouched and the workflow run red.
    """
    projects = ", ".join(f"'{name}'" for name in PACKAGES)
    stale: list[str] = []

    try:
        totals = fetch_totals(projects)
        totals_error = None
    except RuntimeError as error:
        totals, totals_error = {}, error

    try:
        recent, data_date = fetch_recent(projects)
        recent_error = None
    except RuntimeError as error:
        recent, data_date, recent_error = {}, None, error

    if totals_error and recent_error:
        raise RuntimeError(
            f"all queries failed; totals: {totals_error}; recent: {recent_error}"
        )

    if totals_error:
        if previous is None:
            raise RuntimeError(
                f"no previous data to fall back on; totals: {totals_error}"
            )
        stale.append("total")
        warn(f"keeping previously published all-time totals: {totals_error}")
    if recent_error:
        if previous is None:
            raise RuntimeError(
                f"no previous data to fall back on; recent: {recent_error}"
            )
        stale.append(RECENT_KEY)
        data_date = previous["data_date"]
        warn(f"keeping previously published {RECENT_DAYS}-day totals: {recent_error}")

    packages = {
        name: {
            "total": previous_value(previous, name, "total")
            if totals_error
            else totals.get(name, 0),
            RECENT_KEY: previous_value(previous, name, RECENT_KEY)
            if recent_error
            else recent.get(name, 0),
        }
        for name in PACKAGES
    }
    report = {
        "data_date": data_date,
        "source": "https://clickpy.clickhouse.com",
        "packages": packages,
        "combined": {
            "total": sum(entry["total"] for entry in packages.values()),
            RECENT_KEY: sum(entry[RECENT_KEY] for entry in packages.values()),
        },
    }
    if stale:
        report["stale"] = stale
    return report


def humanize(count: int) -> str:
    """Format a download count the way pepy/shields do: 942, 12k, 1.2M."""
    if count < 1_000:
        return str(count)
    if count < 10_000:
        return f"{count / 1_000:.1f}k"
    if count < 1_000_000:
        return f"{round(count / 1_000)}k"
    return f"{count / 1_000_000:.2f}M"


def write_badge(path: Path, label: str, count: int) -> None:
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "label": label,
                "message": humanize(count),
                "color": "blue",
            },
            indent=2,
        )
        + "\n"
    )


def load_previous(path: Path) -> dict | None:
    """Read the last published report, so a failed query can fall back to it."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        warn(f"ignoring unreadable {path}")
        return None


def append_history(path: Path, report: dict) -> None:
    """Append today's snapshot, keeping the file idempotent per day."""
    fields = ["date", "project", "total", RECENT_KEY]
    rows = []
    if path.exists():
        with path.open(newline="") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row["date"] != report["data_date"]
            ]

    for name, entry in report["packages"].items():
        rows.append(
            {
                "date": report["data_date"],
                "project": name,
                "total": entry["total"],
                RECENT_KEY: entry[RECENT_KEY],
            }
        )
    rows.sort(key=lambda row: (row["date"], row["project"]))

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("stats"),
        help="directory the report, badges and history are written to",
    )
    args = parser.parse_args()

    report_path = args.output_dir / "downloads.json"
    report = collect(load_previous(report_path))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_badge(
        args.output_dir / "badge-total.json", "downloads", report["combined"]["total"]
    )
    write_badge(
        args.output_dir / "badge-month.json",
        "downloads/month",
        report["combined"][RECENT_KEY],
    )
    append_history(args.output_dir / "history.csv", report)

    for name, entry in report["packages"].items():
        print(f"{name:28} {entry['total']:>9,} total  {entry[RECENT_KEY]:>7,} /30d")
    combined = report["combined"]
    print(
        f"{'combined':28} {combined['total']:>9,} total  "
        f"{combined[RECENT_KEY]:>7,} /30d  (as of {report['data_date']})"
    )
    if report.get("stale"):
        print(f"note: carried over from the previous run: {', '.join(report['stale'])}")


if __name__ == "__main__":
    main()
