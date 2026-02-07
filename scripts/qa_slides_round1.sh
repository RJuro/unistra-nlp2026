#!/usr/bin/env bash
set -euo pipefail

# Full QA run for lecture/slides_round1.tex:
# 1) compile PDF
# 2) export per-slide PNGs
# 3) extract layout warnings

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LECTURE_DIR="$ROOT_DIR/lecture"
QA_DIR="$LECTURE_DIR/qa"
WARNINGS_FILE="$QA_DIR/slides_round1_layout_warnings.txt"

mkdir -p "$QA_DIR"

(
  cd "$LECTURE_DIR"
  pdflatex -interaction=nonstopmode -halt-on-error slides_round1.tex >/dev/null
  bibtex slides_round1 >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error slides_round1.tex >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error slides_round1.tex >/dev/null
)

"$ROOT_DIR/scripts/export_slides_png.sh" \
  "$LECTURE_DIR/slides_round1.pdf" \
  "$QA_DIR/slides_round1_png"

rg -n "Overfull|Underfull|Warning" "$LECTURE_DIR/slides_round1.log" >"$WARNINGS_FILE" || true

echo "Wrote warnings report: $WARNINGS_FILE"
