#!/bin/bash
# Rebuild the manuscript PDF from records. Run from anywhere.
#
# Order matters: macros.tex is GENERATED from results/records, so regenerate it
# first or the PDF can silently carry stale numbers. tools/audit.py then verifies
# every macro still derives from its record before we typeset anything.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="${PY:-/n/holylabs/hekstra_lab/Everyone/ianpoon/envs/fs-cpu/bin/python}"
export TEXMFVAR="${TEXMFVAR:-/tmp/texmf-var-$USER}"; mkdir -p "$TEXMFVAR"

echo "== regenerating macros from records"
"$PY" "$ROOT/tools/paper_numbers.py"

echo "== regenerating figures and tables from records"
"$PY" "$ROOT/tools/make_figures.py"
"$PY" "$ROOT/tools/make_tables.py"

echo "== verifying provenance (fails if any macro stopped deriving)"
"$PY" "$ROOT/tools/paper_numbers.py" --check

cd "$HERE"
echo "== typesetting"
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null 2>&1 || echo "   (bibtex reported issues; see main.blg)"
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null

if grep -qE "Citation .* undefined|Reference .* undefined" main.log; then
  echo "   WARNING: unresolved citations/references remain"; fi
echo "== done: $HERE/main.pdf ($(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/{print $2}') pages)"
