#!/bin/bash
# Build the ICLR submission PDF. Run from anywhere.
#
# Separate from build.sh because the ICLR class must be found under its own name:
# TEXINPUTS has to include template_iclr/, or `\usepackage{iclr2027_conference}`
# fails. Loading it as `template_iclr/iclr2027_conference` does compile, but it
# suppresses the class's \pagestyle{fancy} -- so the failure is a silently missing
# running header rather than an error, which is worse.
#
# This does NOT regenerate macros/figures/tables. Run build.sh first if records
# have changed: macros.tex is shared by both manuscripts, and body.tex is shared
# verbatim, so a correction to the science reaches ICLR and JCTC together.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export TEXINPUTS="./template_iclr:${TEXINPUTS:-}"
export BSTINPUTS="./template_iclr:${BSTINPUTS:-}"
export TEXMFVAR="${TEXMFVAR:-/tmp/texmf-var-$USER}"; mkdir -p "$TEXMFVAR"

echo "== typesetting main_iclr.tex"
pdflatex -interaction=nonstopmode main_iclr.tex >/dev/null
bibtex main_iclr >/dev/null 2>&1 || echo "   (bibtex reported issues; see main_iclr.blg)"
pdflatex -interaction=nonstopmode main_iclr.tex >/dev/null
pdflatex -interaction=nonstopmode main_iclr.tex >/dev/null

if grep -qE "Citation .* undefined|Reference .* undefined" main_iclr.log; then
  echo "   WARNING: unresolved citations/references remain"; fi

# Double-blind gate. ICLR desk-rejects on de-anonymisation and there is no
# rebuttal for it, so this is checked mechanically against the RENDERED pdf --
# the author block is set by the class, which the source never spells out.
# Counted rather than tested with `grep -q`: under `set -o pipefail`, grep -q
# exits on the first match and pdftotext then dies of SIGPIPE, so the pipeline
# reports failure precisely when the pattern IS found. That inverted both gates.
pdftotext main_iclr.pdf - > .anoncheck.txt
LEAK=$(grep -ciE 'Ian Poon|Harvard|ian_poon' .anoncheck.txt || true)
ANON=$(grep -ci 'Anonymous authors' .anoncheck.txt || true)
rm -f .anoncheck.txt
[ "$LEAK" -eq 0 ] || { echo "   ERROR: $LEAK identifying string(s) in the rendered PDF"; exit 1; }
[ "$ANON" -ge 1 ] || { echo "   ERROR: anonymous author block missing"; exit 1; }

BODY=0
for p in $(seq 1 20); do
  pdftotext -f "$p" -l "$p" main_iclr.pdf - > .pagecheck.txt 2>/dev/null || true
  # ICLR sets section headings in small caps, which pdftotext extracts with a
  # space after the leading capital -- "R EFERENCES", not "REFERENCES". Matching
  # the obvious spelling silently reported a 0-page body.
  N=$(grep -ciE '^[[:space:]]*R[[:space:]]*EFERENCES[[:space:]]*$' .pagecheck.txt || true)
  if [ "$N" -ge 1 ]; then BODY=$((p-1)); break; fi
done
rm -f .pagecheck.txt
echo "== done: $HERE/main_iclr.pdf ($(pdfinfo main_iclr.pdf | awk '/^Pages/{print $2}') pages, body ${BODY}/9)"
[ "$BODY" -gt 9 ] && { echo "   ERROR: body exceeds the 9-page ICLR limit"; exit 1; } || true
