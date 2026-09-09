#!/usr/bin/env python3
"""Bump the version in the three places a release reads it from.

Releasing is driven by the version in pyproject.toml (see .github/workflows/release.yml),
and `scripts/check_changelog.py` requires uv.lock and CHANGELOG.md to agree with it. This
does all three, so a bump can't half-land.

Run with `make bump` (patch), `make bump PART=minor`, or add notes:
    make bump NOTES="Fixed the thing;Added the other thing"

Notes are semicolon-separated so the whole set survives one shell word from make.
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys

# scripts/ is sys.path[0] when run as a script, so the release contract stays
# defined in one place instead of being restated here.
from check_changelog import CHANGELOG, INSERTION_FLAG, PYPROJECT, read_pyproject_version

PLACEHOLDER = "TODO: describe this release"
PARTS = ("major", "minor", "patch")
VERSION_RE = re.compile(r'^version = "(?P<version>\d+\.\d+\.\d+)"$', re.MULTILINE)


def next_version(current: str, part: str) -> str:
    major, minor, patch = (int(piece) for piece in current.split("."))
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def write_pyproject_version(version: str) -> None:
    text = PYPROJECT.read_text()
    # Anchored to the [project] table's own key: the file also carries a
    # [tool.semantic_release] block, and a loose match could rewrite the wrong line.
    project_table = text.index("[project]")
    match = VERSION_RE.search(text, project_table)
    if match is None:
        sys.exit("❌ No `version = \"X.Y.Z\"` line found under [project] in pyproject.toml")
    PYPROJECT.write_text(text[: match.start()] + f'version = "{version}"' + text[match.end() :])


def insert_changelog_entry(version: str, notes: list[str]) -> None:
    text = CHANGELOG.read_text()
    if INSERTION_FLAG not in text:
        sys.exit(f"❌ CHANGELOG.md no longer contains the {INSERTION_FLAG!r} marker")
    today = datetime.date.today().isoformat()
    bullets = "\n".join(f"- {note}" for note in notes or [PLACEHOLDER])
    entry = f"\n\n## v{version} ({today})\n\n{bullets}"
    CHANGELOG.write_text(text.replace(INSERTION_FLAG, INSERTION_FLAG + entry, 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", nargs="?", default="patch", choices=PARTS)
    parser.add_argument(
        "--notes",
        default="",
        metavar="TEXT;TEXT",
        help=f"Semicolon-separated CHANGELOG bullets. Without them, a {PLACEHOLDER!r} bullet is left.",
    )
    args = parser.parse_args()
    notes = [note.strip() for note in args.notes.split(";") if note.strip()]

    current = read_pyproject_version()
    version = next_version(current, args.part)
    print(f"Bumping {current} -> {version} ({args.part})", flush=True)

    write_pyproject_version(version)
    insert_changelog_entry(version, notes)
    subprocess.run(["uv", "lock"], check=True)

    # The release gate itself, so a bad bump fails here and not in CI.
    check = subprocess.run([sys.executable, "scripts/check_changelog.py"], check=False)
    if check.returncode != 0:
        return check.returncode

    if not notes:
        print(f"\n⚠️  CHANGELOG.md v{version} still says {PLACEHOLDER!r} — replace it before merging.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
