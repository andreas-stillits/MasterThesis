#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG -----------------------------------------------------
# Local data folder
LOCAL_DIR="/home/andreasstillits/coding/MasterThesis/.treasury"

# Remote data folder (same for all machines)
REMOTE_DIR="andreasstillits@10.209.65.239:/home/andreasstillits/coding/MasterThesis/.treasury"
# ---------------------------------------------------------------

usage() {
  cat >&2 <<EOF
Usage: $0 [push|pull] [--dry-run]

  push    : sync LOCAL -> REMOTE (upload)
  pull    : sync REMOTE -> LOCAL (download)
  --dry-run: show what would happen without changing anything
EOF
  exit 1
}

# Parse args
DRY_RUN=""
[[ $# -ge 1 ]] || usage

DIRECTION="$1"
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN="--dry-run"
      shift
      ;;
    *)
      usage
      ;;
  esac
done

case "$DIRECTION" in
  push)
    echo ">>> Pushing LOCAL ($LOCAL_DIR) -> REMOTE ($REMOTE_DIR)"
    rsync -av $DRY_RUN --delete "$LOCAL_DIR/" "$REMOTE_DIR/"
    ;;
  pull)
    echo ">>> Pulling REMOTE ($REMOTE_DIR) -> LOCAL ($LOCAL_DIR)"
    rsync -av $DRY_RUN --delete "$REMOTE_DIR/" "$LOCAL_DIR/"
    ;;
  *)
    usage
    ;;
esac
