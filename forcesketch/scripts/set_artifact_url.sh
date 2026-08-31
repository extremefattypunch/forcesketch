#!/usr/bin/env bash
# Swap the artifact URL in the manuscript. Usage:
#   scripts/set_artifact_url.sh https://anonymous.4open.science/r/forcesketch-artifact-XXXX
#
# 09_freeze.py refuses to freeze while the URL still says PLACEHOLDER, so this is
# the last step before submission. The URL's host must be on the allowlist in
# configs/anonymity.yaml, or the anonymity audit will reject it.
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: $0 <artifact-url>" >&2; exit 2; }
URL="$1"
case "$URL" in
  https://anonymous.4open.science/*|https://doi.org/*|https://zenodo.org/*) ;;
  *) echo "REFUSING: '$URL' is not an anonymised host. A github.com/<user> link" >&2
     echo "identifies the author and the call for papers lists that as grounds" >&2
     echo "for desk rejection." >&2; exit 1 ;;
esac
python3 - "$URL" <<'PY'
import pathlib, re, sys
p = pathlib.Path(__file__).resolve().parent.parent / "paper" / "main.tex"
p = pathlib.Path("paper/main.tex")
s = p.read_text()
new = re.sub(r"(\\newcommand\{\\fsCodeUrl\}\{)[^}]*(\})", lambda m: m.group(1)+sys.argv[1]+m.group(2), s)
assert new != s, "fsCodeUrl not found in paper/main.tex"
p.write_text(new)
print(f"set \\fsCodeUrl -> {sys.argv[1]}")
PY
cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null
echo "rebuilt paper/main.pdf -- now run: python scripts/09_freeze.py"
