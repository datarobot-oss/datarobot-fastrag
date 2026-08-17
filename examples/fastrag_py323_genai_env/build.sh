#!/usr/bin/env bash
#
# Build the fastrag GenAI execution-environment image locally and export it as a
# tarball you can upload to DataRobot as a prebuilt custom environment image.
#
# Steps:
#   1. Copy the datarobot-fastrag wheel from the repo's dist/ into this build context
#      (the Dockerfile COPYs it by a version-agnostic glob -- datarobot_fastrag-*.whl --
#      so the wheel just needs to sit next to the Dockerfile, whatever its version).
#   2. docker build  -- forced to linux/amd64: the DR runtime and the ubi9 base image
#      are x86-64, so an arm64 (Apple Silicon) build would not run on the platform.
#   3. docker save | gzip  -- a portable image tarball, written to the git-ignored dist/.
#
# Usage:
#   ./build.sh                          # build with defaults
#   IMAGE_TAG=my-env:1 ./build.sh       # override the image tag
#   PLATFORM=linux/amd64 ./build.sh     # override the target platform
#   WHEEL_FILE=dist/....whl ./build.sh  # use a specific wheel instead of auto-discovery
#
set -euo pipefail

# --- locate paths ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIST_DIR="${REPO_ROOT}/dist"

# --- config (override via env) -----------------------------------------------
IMAGE_TAG="${IMAGE_TAG:-datarobot-fastrag-genai-env:local}"
PLATFORM="${PLATFORM:-linux/amd64}"

# --- 1. locate and copy the wheel into the build context ---------------------
# The Dockerfile COPYs the wheel by a version-agnostic glob (datarobot_fastrag-*.whl),
# so we discover the built wheel in dist/ rather than parsing a pinned version out of
# the Dockerfile. Set WHEEL_FILE to bypass auto-discovery and use a specific wheel.
if [[ -n "${WHEEL_FILE:-}" ]]; then
  SRC_WHEEL="${WHEEL_FILE}"
  if [[ ! -f "${SRC_WHEEL}" ]]; then
    echo "ERROR: WHEEL_FILE does not exist: ${SRC_WHEEL}" >&2
    exit 1
  fi
else
  # Version-agnostic match against dist/. nullglob so a no-match yields an empty array.
  shopt -s nullglob
  WHEELS=("${DIST_DIR}"/datarobot_fastrag-*-py3-none-any.whl)
  shopt -u nullglob
  if [[ ${#WHEELS[@]} -eq 0 ]]; then
    echo "ERROR: no datarobot_fastrag wheel found in ${DIST_DIR}" >&2
    echo "       Build it first from the repo root:  uv build   (or: make build)" >&2
    exit 1
  fi
  if [[ ${#WHEELS[@]} -gt 1 ]]; then
    # Ambiguous: fall back to the most recently built wheel and warn.
    SRC_WHEEL="$(ls -t "${DIST_DIR}"/datarobot_fastrag-*-py3-none-any.whl | head -1)"
    echo ">> WARNING: multiple wheels in ${DIST_DIR}; using the newest (${SRC_WHEEL##*/})." >&2
    printf '     found: %s\n' "${WHEELS[@]##*/}" >&2
    echo "     Set WHEEL_FILE=... to pick a specific one." >&2
  else
    SRC_WHEEL="${WHEELS[0]}"
  fi
fi

WHEEL_NAME="$(basename "${SRC_WHEEL}")"
echo ">> Copying ${WHEEL_NAME} into the build context"
cp "${SRC_WHEEL}" "${SCRIPT_DIR}/${WHEEL_NAME}"
# Remove the copied wheel on exit so we don't leave a build artifact in the source tree.
trap 'rm -f "${SCRIPT_DIR}/${WHEEL_NAME}"' EXIT

# --- 2. build the image ------------------------------------------------------
echo ">> Building ${IMAGE_TAG} for ${PLATFORM}"
docker build --platform "${PLATFORM}" -t "${IMAGE_TAG}" "${SCRIPT_DIR}"

# --- 3. export the image as a tarball ----------------------------------------
TARBALL="${DIST_DIR}/$(echo "${IMAGE_TAG}" | tr '/:' '__').tar.gz"
echo ">> Saving image to ${TARBALL}"
docker save "${IMAGE_TAG}" | gzip > "${TARBALL}"

echo ""
echo "Done."
echo "  Image:   ${IMAGE_TAG}"
echo "  Tarball: ${TARBALL}"
echo "  Upload the tarball to DataRobot as a prebuilt custom environment image"
echo "  (Registry -> Environments -> new version -> upload Docker image)."
