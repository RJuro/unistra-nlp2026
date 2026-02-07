#!/usr/bin/env bash
set -euo pipefail

# Downscale Gemini-generated PNGs to slide-friendly size and copy to generated/.
# Usage:
#   scripts/optimize_gemini_figures.sh [src_dir] [dst_dir]
# Defaults:
#   src_dir=lecture/figures/gemini
#   dst_dir=lecture/figures/generated

SRC_DIR="${1:-lecture/figures/gemini}"
DST_DIR="${2:-lecture/figures/generated}"
MAX_EDGE="${MAX_EDGE:-1920}"

if ! command -v sips >/dev/null 2>&1; then
  echo "Error: sips is required but not found." >&2
  exit 1
fi

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Error: source directory not found: $SRC_DIR" >&2
  exit 1
fi

mkdir -p "$DST_DIR"

for src in "$SRC_DIR"/*.png; do
  [[ -e "$src" ]] || continue

  base="$(basename "$src")"
  out="$base"

  # Canonical output names expected by slides.
  case "$base" in
    topic_model.png) out="topic_model_viz.png" ;;
  esac

  dst="$DST_DIR/$out"
  cp "$src" "$dst"
  sips -Z "$MAX_EDGE" "$dst" >/dev/null

  # Keep an additional alias for topic model for convenience.
  if [[ "$base" == "topic_model.png" ]]; then
    cp "$dst" "$DST_DIR/topic_model.png"
  fi

  printf "optimized: %s -> %s\n" "$src" "$dst"
done

echo "Done. Optimized figures in: $DST_DIR"
