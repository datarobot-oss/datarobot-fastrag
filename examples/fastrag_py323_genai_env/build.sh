#!/usr/bin/env bash
#
# Build the fastrag GenAI execution-environment image locally and export it as a
# tarball you can upload to DataRobot as a prebuilt custom environment image.
#
# Steps:
#   1. Copy the datarobot-fastrag wheel from the repo's dist/ into this build context
#      (the Dockerfile COPYs it by name, so it must sit next to the Dockerfile).
#   2. docker build  -- forced to linux/amd64: the DR runtime and the ubi9 base image
#      are x86-64, so an arm64 (Apple Silicon) build would not run on the platform.
#   3. docker save | gzip  -- a portable image tarball, written to the git-ignored dist/.
#
# Usage:
#   ./build.sh                          # build with defaults
#   IMAGE_TAG=my-env:1 ./build.sh       # override the image tag
#   PLATFORM=linux/amd64 ./build.sh     # override the target platform
#
set -euo pipefail

# --- locate paths ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIST_DIR="${REPO_ROOT}/dist"

# --- config (override via env) -----------------------------------------------
IMAGE_TAG="${IMAGE_TAG:-datarobot-fastrag-genai-env:local}"
PLATFORM="${PLATFORM:-linux/amd64}"

# --- 1. copy the wheel into the build context --------------------------------
# The Dockerfile is the source of truth for the exact wheel filename it expects.
WHEEL_NAME="$(grep -oE 'datarobot_fastrag-[0-9][0-9.]*-py3-none-any\.whl' "${SCRIPT_DIR}/Dockerfile" | head -1)"
if [[ -z "${WHEEL_NAME}" ]]; then
  echo "ERROR: could not determine the wheel filename from the Dockerfile." >&2
  exit 1
fi

SRC_WHEEL="${DIST_DIR}/${WHEEL_NAME}"
if [[ ! -f "${SRC_WHEEL}" ]]; then
  echo "ERROR: wheel not found: ${SRC_WHEEL}" >&2
  echo "       Build it first from the repo root:  make build   (or: uv build)" >&2
  exit 1
fi

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
