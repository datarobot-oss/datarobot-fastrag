#!/usr/bin/env bash
set -ex

version=$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")

echo "Tagging release with version ${version}"
git tag -a -f "$version" -m "Tagging datarobot-fastrag release $version"
git push origin "$version"
