#!/usr/bin/env bash
set -euo pipefail

# Export every slide of a PDF deck to PNG files.
# Usage:
#   scripts/export_slides_png.sh [pdf_path] [out_dir]
# Defaults:
#   pdf_path=lecture/slides_round1.pdf
#   out_dir=lecture/qa/slides_round1_png

PDF_PATH="${1:-lecture/slides_round1.pdf}"
OUT_DIR="${2:-lecture/qa/slides_round1_png}"
DPI="${DPI:-170}"

if ! command -v pdftoppm >/dev/null 2>&1; then
  echo "Error: pdftoppm is required but not found." >&2
  exit 1
fi

if [[ ! -f "$PDF_PATH" ]]; then
  echo "Error: PDF not found: $PDF_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# Clean old exports
rm -f "$OUT_DIR"/slide-*.png

prefix="$OUT_DIR/slide"
pdftoppm -png -r "$DPI" "$PDF_PATH" "$prefix" >/dev/null

# Normalize names to slide-001.png format.
for file in "$OUT_DIR"/slide-*.png; do
  [[ -e "$file" ]] || continue
  n="${file##*-}"
  n="${n%.png}"
  n_base10=$((10#$n))
  printf -v pad "%03d" "$n_base10"
  mv "$file" "$OUT_DIR/slide-$pad.png"
done

# Create a simple HTML index for manual visual QA.
{
  echo "<!doctype html><html><head><meta charset=\"utf-8\"><title>Slide QA</title></head><body>"
  echo "<h1>Slide QA: $(basename "$PDF_PATH")</h1>"
  for img in "$OUT_DIR"/slide-*.png; do
    base="$(basename "$img")"
    echo "<div style=\"margin:16px 0\"><h3>$base</h3><img src=\"$base\" style=\"width:100%;max-width:1200px;border:1px solid #ccc\"></div>"
  done
  echo "</body></html>"
} >"$OUT_DIR/index.html"

count="$(find "$OUT_DIR" -maxdepth 1 -name 'slide-*.png' | wc -l | tr -d ' ')"
echo "Exported $count slides to: $OUT_DIR"
echo "Review index: $OUT_DIR/index.html"
