#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"
REPO_NAME="CMKD_OpenPCDet"
BRANCH_NAME="$(git -C "$ROOT_DIR" branch --show-current)"
COMMIT_SHA="$(git -C "$ROOT_DIR" rev-parse --short HEAD)"
ARCHIVE_BASENAME="${REPO_NAME}-${BRANCH_NAME}-${COMMIT_SHA}"

mkdir -p "$DIST_DIR"

TAR_PATH="${DIST_DIR}/${ARCHIVE_BASENAME}.tar.gz"
BUNDLE_PATH="${DIST_DIR}/${ARCHIVE_BASENAME}.bundle"

# source archive without .git so users can extract and run directly

tar \
  --exclude="${ROOT_DIR}/.git" \
  --exclude="${ROOT_DIR}/dist" \
  -czf "$TAR_PATH" \
  -C "$ROOT_DIR/.." \
  "$(basename "$ROOT_DIR")"

# git bundle preserves history for users who want the branch exactly as committed

git -C "$ROOT_DIR" bundle create "$BUNDLE_PATH" --all

cat <<MSG
Created:
  $TAR_PATH
  $BUNDLE_PATH

Target machine quick start:
  mkdir -p ~/packages
  cp "$TAR_PATH" ~/packages/
  cd ~
  tar -xzf ~/packages/$(basename "$TAR_PATH")
  cd ${REPO_NAME}
MSG
