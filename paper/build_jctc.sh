#!/bin/bash
# Build the JCTC submission PDF on the ACS class. Run from anywhere.
#
# Does NOT regenerate macros/figures/tables -- run build.sh first if records have
# changed. macros.tex and body.tex are shared with the ICLR build, so a
# correction to the science reaches both manuscripts together.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export TEXMFVAR="${TEXMFVAR:-/tmp/texmf-var-$USER}"; mkdir -p "$TEXMFVAR"

echo "== typesetting main_jctc.tex"
pdflatex -interaction=nonstopmode main_jctc.tex >/dev/null
bibtex main_jctc >/dev/null 2>&1 || echo "   (bibtex reported issues; see main_jctc.blg)"
pdflatex -interaction=nonstopmode main_jctc.tex >/dev/null
pdflatex -interaction=nonstopmode main_jctc.tex >/dev/null

if grep -qE "Citation .* undefined|Reference .* undefined" main_jctc.log; then
  echo "   WARNING: unresolved citations/references remain"; fi

# Same artifact gate as the ICLR build: a placeholder locator must never reach a
# journal. Counted rather than tested with grep -q, because under pipefail a
# matching grep -q kills pdftotext with SIGPIPE and inverts the test.
pdftotext main_jctc.pdf - > .phcheck.txt
PH=$(grep -c 'PLACEHOLDER' .phcheck.txt || true)
NAME=$(grep -c 'Ian Poon' .phcheck.txt || true)
rm -f .phcheck.txt
[ "$PH" -eq 0 ] || { echo "   ERROR: artifact locator is still a PLACEHOLDER in the rendered PDF."; \
                     echo "          Create the deposit, then fill paper/artifact.tex."; exit 1; }
[ "$NAME" -ge 1 ] || { echo "   ERROR: author name missing -- this build is NOT anonymous and should name the author"; exit 1; }

echo "== done: $HERE/main_jctc.pdf ($(pdfinfo main_jctc.pdf | awk '/^Pages/{print $2}') pages)"
