#!/usr/bin/env python3
"""Require every PR to ship a CHANGELOG entry for the version it releases.

Run locally with `make changelog-check`; CI runs it from
.github/workflows/changelog-check.yml.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
UV_LOCK = REPO_ROOT / "uv.lock"

PACKAGE = "datarobot-fastrag"
SKIP_LABEL = "skip-changelog"
INSERTION_FLAG = "<!-- version list -->"

IN_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"

# Matches the established entry style, e.g. "## v0.2.3 (2026-09-02)"
HEADING_RE = re.compile(
    r"^## v(?P<version>\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-]+)?)"
    r" \((?P<date>\d{4}-\d{2}-\d{2})\)[ \t]*$",
    re.MULTILINE,
)


class Entry(NamedTuple):
    version: str
    date: str
    line: int  # 1-indexed, for GitHub file annotations
    body: str


class Failure(NamedTuple):
    message: str
    hint: str
    line: int | None = None


def git(*args: str) -> str | None:
    """stdout of a git command, or None if it failed (e.g. an unfetched ref)."""
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout if result.returncode == 0 else None


def warn(message: str) -> None:
    print(f"::warning::{message}" if IN_ACTIONS else f"⚠️  {message}")


def line_number(text: str, position: int) -> int:
    return text.count("\n", 0, position) + 1


def read_pyproject_version() -> str:
    with PYPROJECT.open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def read_lock_version() -> str | None:
    """Version recorded for this package in uv.lock, or None if absent."""
    with UV_LOCK.open("rb") as handle:
        lock = tomllib.load(handle)
    return next((p["version"] for p in lock["package"] if p["name"] == PACKAGE), None)


def parse_changelog(text: str) -> list[Entry]:
    """Every `## vX.Y.Z (date)` section, in file order (newest first)."""
    headings = list(HEADING_RE.finditer(text))
    ends = [h.start() for h in headings[1:]] + [len(text)]
    return [
        Entry(
            version=h["version"],
            date=h["date"],
            line=line_number(text, h.start()),
            body=text[h.end() : end].strip(),
        )
        for h, end in zip(headings, ends)
    ]


def check_headings_parse(text: str) -> Failure | None:
    """Catch a section that looks like an entry but won't be recognised as one."""
    for number, line in enumerate(text.splitlines(), 1):
        if line.startswith("## ") and not HEADING_RE.match(line):
            return Failure(
                f"CHANGELOG.md:{number} is not a recognised entry heading: {line!r}",
                'Use the established format, e.g. "## v0.2.3 (2026-09-02)".',
                number,
            )
    return None


def check_flag_precedes_newest(text: str, newest: Entry) -> Failure | None:
    if INSERTION_FLAG not in text:
        return Failure(
            f"CHANGELOG.md no longer contains the {INSERTION_FLAG!r} marker",
            "python-semantic-release inserts generated sections directly below it\n"
            "(changelog.insertion_flag), so it has to stay above the newest entry.",
        )
    flag_line = line_number(text, text.index(INSERTION_FLAG))
    if flag_line > newest.line:
        return Failure(
            f"CHANGELOG.md:{newest.line} puts v{newest.version} above the "
            f"{INSERTION_FLAG!r} marker on line {flag_line}",
            "Newest entry goes directly below the marker, so entries stay newest-first.",
            newest.line,
        )
    return None


def check_newest_matches_pyproject(newest: Entry, version: str) -> Failure | None:
    if newest.version == version:
        return None
    return Failure(
        f"CHANGELOG.md's newest entry is v{newest.version}, but pyproject.toml ships {version}",
        f'Add a "## v{version} (<YYYY-MM-DD>)" section at the top of CHANGELOG.md,\n'
        f"or correct the version in pyproject.toml.",
        newest.line,
    )


def check_entry_has_content(newest: Entry) -> Failure | None:
    if newest.body:
        return None
    return Failure(
        f"CHANGELOG.md's v{newest.version} section is empty",
        "Describe the change in at least one bullet",
        newest.line,
    )


def check_lock_in_sync(version: str) -> Failure | None:
    locked = read_lock_version()
    if locked == version:
        return None
    found = f"pins {PACKAGE} {locked}" if locked else f"has no {PACKAGE!r} entry"
    return Failure(
        f"uv.lock {found}, but pyproject.toml ships {version}",
        "Run `uv lock` and commit the result.",
    )


def check_version_is_new(version: str, base_ref: str) -> Failure | None:
    """Require the entry to be absent from the base branch."""
    # On the base branch itself the released version is *expected* to be present,
    # so there is no bump to demand.
    head = git("rev-parse", "HEAD")
    if head is not None and head == git("rev-parse", base_ref):
        print(f'HEAD is {base_ref} — skipping the "is this version new?" check')
        return None

    base_text = git("show", f"{base_ref}:CHANGELOG.md")
    if base_text is None:
        warn(f'Could not read CHANGELOG.md at {base_ref} — skipped the "is this new?" check.')
        return None

    if version not in {entry.version for entry in parse_changelog(base_text)}:
        return None
    return Failure(
        f"v{version} already has a CHANGELOG.md entry on {base_ref}",
        f"This PR reuses an already-released version. Bump the version in\n"
        f"pyproject.toml, run `uv lock`, and add a new section at the top of\n"
        f"CHANGELOG.md.\n"
        f"If the change genuinely needs no release (docs, CI, tests only),\n"
        f"add the '{SKIP_LABEL}' label to the PR.",
    )


def report(failures: list[Failure]) -> int:
    for failure in failures:
        if IN_ACTIONS:
            location = f",line={failure.line}" if failure.line else ""
            print(f"::error file=CHANGELOG.md{location}::{failure.message}")
        print(f"❌ {failure.message}")
        for hint_line in failure.hint.splitlines():
            print(f"   {hint_line}")
        print()
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Branch the PR merges into (default: origin/main).",
    )
    args = parser.parse_args()

    version = read_pyproject_version()
    text = CHANGELOG.read_text()
    print(f"Version in pyproject.toml: {version}")

    # Structural checks first: without a parseable newest entry the rest is noise.
    if malformed := check_headings_parse(text):
        return report([malformed])

    entries = parse_changelog(text)
    if not entries:
        return report(
            [
                Failure(
                    "CHANGELOG.md has no version entries",
                    f'Add a "## v{version} (<YYYY-MM-DD>)" section below the\n'
                    f"{INSERTION_FLAG!r} marker.",
                )
            ]
        )

    newest = entries[0]
    print(f"Newest CHANGELOG.md entry: v{newest.version} ({newest.date})")

    failures = [
        failure
        for failure in (
            check_flag_precedes_newest(text, newest),
            check_newest_matches_pyproject(newest, version),
            check_entry_has_content(newest),
            check_lock_in_sync(version),
            check_version_is_new(version, args.base_ref),
        )
        if failure
    ]
    if failures:
        return report(failures)

    print(f"✅ CHANGELOG.md documents v{version}, and pyproject.toml and uv.lock agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
