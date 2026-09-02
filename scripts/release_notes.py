#!/usr/bin/env python3
"""Print a version's CHANGELOG.md section, for use as GitHub release notes.

.github/workflows/release.yml feeds this to `gh release create --notes-file` when a
version bump lands on main, so the release notes are the CHANGELOG entry verbatim
rather than a hand copy-paste.

Preview the notes the current pyproject.toml version would release with:

    uv run python scripts/release_notes.py
"""

from __future__ import annotations

import argparse
import sys

from check_changelog import CHANGELOG
from check_changelog import parse_changelog
from check_changelog import read_pyproject_version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version",
        nargs="?",
        help="Version to extract, without the leading 'v' (default: pyproject.toml's).",
    )
    args = parser.parse_args()

    version = args.version or read_pyproject_version()
    entries = {entry.version: entry for entry in parse_changelog(CHANGELOG.read_text())}

    entry = entries.get(version)
    if entry is None:
        print(f"CHANGELOG.md has no entry for v{version}", file=sys.stderr)
        print("Add one, or correct the version in pyproject.toml.", file=sys.stderr)
        return 1

    print(entry.body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
