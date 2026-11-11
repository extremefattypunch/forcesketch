#!/bin/bash
# Build the Supporting Information PDF on the ACS suppinfo class.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
export TEXMFVAR="${TEXMFVAR:-/tmp/texmf-var-$USER}"; mkdir -p "$TEXMFVAR"
pdflatex -interaction=nonstopmode si.tex >/dev/null
bibtex si >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode si.tex >/dev/null
pdflatex -interaction=nonstopmode si.tex >/dev/null
grep -qE "Citation .* undefined|Reference .* undefined" si.log && echo "   WARNING: unresolved references in SI" || true
echo "== done: $HERE/si.pdf ($(pdfinfo si.pdf | awk '/^Pages/{print $2}') pages)"
