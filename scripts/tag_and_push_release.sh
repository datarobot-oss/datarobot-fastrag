#!/usr/bin/env bash
# Manual fallback for tagging a release. The normal path is automatic: merging a
# version bump to main makes .github/workflows/release.yml cut the `v<version>` tag
# and the GitHub release. Use this only when releasing outside that flow, and note
# it creates the tag alone — no GitHub release, so nothing publishes to PyPI.
set -ex

version=$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")

# The `v` prefix is the repo's tag convention (v0.2.1, v0.2.2) and what release.yml
# looks for when deciding whether a version has already shipped.
echo "Tagging release with version v${version}"
git tag -a -f "v$version" -m "Tagging datarobot-fastrag release v$version"
git push origin "v$version"
