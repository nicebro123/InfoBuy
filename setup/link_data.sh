#!/usr/bin/env bash
# Optional compatibility bridge.
#
# The preferred InfoBuy layout keeps datasets outside the code repo and uses
# $INFOBUY_GENERATED_DATA directly. Run this only if an older command still
# expects repo data/ paths.
#
# Link repo data/ to $INFOBUY_STORE/datasets, so repo data/infobuy maps to
# $INFOBUY_STORE/datasets/infobuy. Usage:
#   source setup/env.sh && bash setup/link_data.sh
set -euo pipefail

: "${INFOBUY_GENERATED_DATA:?source setup/env.sh first}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DATA="$REPO_ROOT/data"
STORE_DATA_ROOT="$(dirname "$INFOBUY_GENERATED_DATA")"

mkdir -p "$INFOBUY_GENERATED_DATA"

if [ -L "$REPO_DATA" ]; then
  current_target="$(readlink "$REPO_DATA")"
  if [ "$current_target" = "$STORE_DATA_ROOT" ]; then
    echo "data/ is already linked to $STORE_DATA_ROOT"
    exit 0
  fi
  echo "Refusing to replace existing data symlink: $REPO_DATA -> $current_target" >&2
  exit 2
fi

if [ -d "$REPO_DATA" ]; then
  echo "Copying existing repo data/ into $STORE_DATA_ROOT before linking..."
  mkdir -p "$STORE_DATA_ROOT"
  rsync -a "$REPO_DATA/" "$STORE_DATA_ROOT/"
  backup="$REPO_ROOT/data.local_backup.$(date +%Y%m%d_%H%M%S)"
  mv "$REPO_DATA" "$backup"
  echo "Moved original data/ to $backup"
elif [ -e "$REPO_DATA" ]; then
  echo "Refusing to replace non-directory path: $REPO_DATA" >&2
  exit 2
else
  mkdir -p "$STORE_DATA_ROOT"
fi

ln -s "$STORE_DATA_ROOT" "$REPO_DATA"
echo "Linked $REPO_DATA -> $STORE_DATA_ROOT"
