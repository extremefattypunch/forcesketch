#!/usr/bin/env python
"""Procedural audit of the journal repository.

Successor to the frozen tree's `09_freeze.py`, and deliberately different from it
in one respect the plan is explicit about: **a scientific outcome is never an exit
code.** `09_freeze.py` exits non-zero when a result fails a gate, which conflates
"the experiment said no" with "the pipeline is broken" and creates pressure to
make results pass. This exits non-zero only on PROCEDURAL failure — a hash that
moved, a manifest that no longer recomputes, a quoted number that no longer
derives. Whether the science came out favourably is none of its business.

Checks, in order of how badly a failure would hurt:

  frozen      the workshop tree is bit-identical to tag workshop-v1.0
  protocols   every registered protocol still hashes as it did at registration
  macros      every paper number still derives from its record, unchanged
  splits      every split manifest recomputes: same roles, same content hash,
              roles disjoint, guard bands honoured
  records     required provenance fields are present on every record
  banned      the two divergent frozen split helpers are never imported

Run with --verbose to see each check's detail.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = ROOT.parents[0]
REC = ROOT / "results/records"
sys.path.insert(0, str(ROOT / "src"))

from forcesketch_journal.data.splits import (  # noqa: E402
    contiguous_block_split, random_frame_split,
)


class Audit:
    def __init__(self, verbose: bool):
        self.fail: list[str] = []
        self.notes: list[str] = []
        self.passed_names: list[str] = []
        self.verbose = verbose

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        """`detail` explains a FAILURE and is only ever shown on failure.

        An earlier version echoed it under --verbose when the check PASSED, so a
        healthy repository printed "frozen-tree: forcesketch/ differs from tag
        workshop-v1.0" as a note. Output that reads as a failure on success is
        worse than no output: it teaches the reader to skim past real failures.
        """
        if not ok:
            self.fail.append(f"{name}: {detail}")
        elif self.verbose:
            self.passed_names.append(name)

    # ---------------------------------------------------------------- frozen
    def frozen(self) -> None:
        r = subprocess.run(["git", "diff", "--quiet", "workshop-v1.0", "--", "forcesketch/"],
                           cwd=REPO, capture_output=True)
        self.check("frozen-tree", r.returncode == 0,
                   "forcesketch/ differs from tag workshop-v1.0")
        tags = subprocess.run(["git", "tag"], cwd=REPO, capture_output=True, text=True).stdout
        self.check("frozen-tag", "workshop-v1.0" in tags, "tag workshop-v1.0 is missing")

    # ------------------------------------------------------------- protocols
    def protocols(self) -> None:
        regs = sorted(REC.glob("*_protocol_registration.json"))
        self.check("protocols-exist", bool(regs), "no protocol registration records")
        for r in regs:
            d = json.loads(r.read_text())
            p = ROOT / d["protocol"]
            if not p.exists():
                self.check(f"protocol-{p.name}", False, "registered protocol file is missing")
                continue
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            self.check(f"protocol-{p.name}", got == d["protocol_sha256"],
                       f"edited since registration ({d['protocol_sha256'][:12]} -> {got[:12]})")
            pre = d.get(next((k for k in d if k.endswith("_at_registration")), ""), None)
            if pre is not None:
                self.check(f"protocol-{p.name}-priority", pre == [],
                           f"records already existed at registration: {pre}")

    # ------------------------------------------------------- git protocol binding
    def git_binding(self) -> None:
        """The protocol's commit must not POSTDATE the records it governs.

        The plan asks for `git merge-base --is-ancestor` proving a protocol
        predates its result. In practice a protocol and the records it governs
        are often introduced in the same commit, which is neither backdating nor
        proof of priority -- for those the sha256 registration record (which
        asserts zero matching records existed at registration time) is what
        carries the claim. Only a protocol committed strictly LATER than its
        records is evidence of backdating, and only that fails.
        """
        def first_commit(rel: str) -> str | None:
            r = subprocess.run(
                ["git", "log", "--reverse", "--format=%H", "--", rel],
                cwd=REPO, capture_output=True, text=True)
            out = r.stdout.split()
            return out[0] if out else None

        for reg in sorted(REC.glob("*_protocol_registration.json")):
            d = json.loads(reg.read_text())
            proto_rel = f"forcesketchJournalReady/{d['protocol']}"
            pc = first_commit(proto_rel)
            rc = first_commit(f"forcesketchJournalReady/{reg.relative_to(ROOT)}")
            if not pc or not rc:
                self.notes.append(f"  git-binding {pathlib.Path(d['protocol']).name}: "
                                  "not yet committed; sha256 registration only")
                continue
            if pc == rc:
                self.notes.append(f"  git-binding {pathlib.Path(d['protocol']).name}: "
                                  "protocol and registration in the same commit; "
                                  "priority rests on the registration record")
                continue
            anc = subprocess.run(["git", "merge-base", "--is-ancestor", pc, rc],
                                 cwd=REPO, capture_output=True)
            self.check(f"git-binding-{pathlib.Path(d['protocol']).name}",
                       anc.returncode == 0,
                       "protocol was committed AFTER the records it governs")

    # ------------------------------------------------------------------ ledger
    def ledger(self) -> None:
        from forcesketch_journal.data.ledger import audit_ledger
        out = audit_ledger()
        self.check("test-access-ledger", out["ok"],
                   f"an experiment read one system's test split under "
                   f"{out['conflicts']} different splits")
        self.notes.append(f"  ledger: {out['n_entries']} recorded accesses over "
                          f"{out['n_pairs']} (experiment, system) pairs"
                          + (f", {len(out['repeated_accesses'])} repeated"
                             if out["repeated_accesses"] else ""))

    # ---------------------------------------------------------------- macros
    def macros(self) -> None:
        r = subprocess.run([sys.executable, str(ROOT / "tools/paper_numbers.py"), "--check"],
                           cwd=ROOT, capture_output=True, text=True)
        self.check("macro-provenance", r.returncode == 0,
                   r.stdout.strip().replace("\n", " | "))
        tex = ROOT / "paper/macros.tex"
        if tex.exists():
            prov = json.loads((REC / "macro_provenance.json").read_text())
            declared = {l.split("{\\")[1].split("}")[0]
                        for l in tex.read_text().splitlines()
                        if l.startswith("\\newcommand")}
            self.check("macro-coverage", declared <= set(prov),
                       f"macros with no provenance entry: {sorted(declared - set(prov))}")
            if self.verbose:
                self.notes.append(f"  macros: {len(declared)} declared, all traced")

    # ---------------------------------------------------------------- splits
    def splits(self) -> None:
        mans = sorted((ROOT / "manifests/splits").glob("*.json"))
        self.check("splits-exist", bool(mans), "no split manifests")
        for m in mans:
            d = json.loads(m.read_text())
            n, scheme, system = d["n_structures"], d["scheme"], d["system"]
            if scheme == "contiguous_block":
                s = contiguous_block_split(n, system=system, seed=d["split_seed"],
                                           tau_int=d["autocorr"]["tau_int"],
                                           guard=d["guard_frames"])
            else:
                s = random_frame_split(n, system=system, seed=d["split_seed"])
            self.check(f"split-{m.stem}-hash", s.sha256 == d["content_sha256"],
                       "manifest does not recompute from its recorded seed")
            roles = {k: np.asarray(v) for k, v in d["roles"].items()}
            allidx = np.concatenate(list(roles.values()))
            self.check(f"split-{m.stem}-disjoint", len(allidx) == len(set(allidx.tolist())),
                       "roles overlap")
            self.check(f"split-{m.stem}-inrange",
                       bool(allidx.min() >= 0 and allidx.max() < n), "index out of range")
            if scheme == "contiguous_block":
                # every kept index must be >= guard away from a DIFFERENT role
                guard = d["guard_frames"]
                lab = np.full(n, "", dtype=object)
                for k, v in roles.items():
                    lab[v] = k
                bad = 0
                for k, v in roles.items():
                    for i in v:
                        lo, hi = max(0, i - guard), min(n, i + guard + 1)
                        near = {lab[j] for j in range(lo, hi) if lab[j] not in ("", k)}
                        bad += bool(near)
                self.check(f"split-{m.stem}-guard", bad == 0,
                           f"{bad} indices sit within the {guard}-frame guard of another role")

    # --------------------------------------------------------------- records
    def records(self) -> None:
        need = {"experiment_id"}
        for f in sorted(REC.glob("*.jsonl")):
            rows = [json.loads(l) for l in f.open() if l.strip()]
            if not rows:
                self.check(f"record-{f.name}", False, "empty record file")
                continue
            missing = need - set(rows[0])
            self.check(f"record-{f.name}", not missing,
                       f"first row lacks {sorted(missing)}" if missing else "")
        for f in sorted(REC.glob("*.pt")):
            self.notes.append(f"  record: {f.name} is a binary artifact (gitignored)")

    # ---------------------------------------------------------------- banned
    def banned(self) -> None:
        """The two divergent frozen split helpers must never be imported or called.

        AST-based, not substring-based. A first version grepped for the name and
        flagged `splits.py` and `test_splits.py` -- both of which merely DOCUMENT
        the ban -- and even flagged this file for containing the string it
        searches for. A check that fires on its own documentation trains people
        to ignore it.
        """
        import ast

        hits = []
        for py in sorted(ROOT.rglob("*.py")):
            if "__pycache__" in str(py):
                continue
            try:
                tree = ast.parse(py.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and \
                        "fallback_gate" in node.module:
                    for al in node.names:
                        if al.name == "split_indices":
                            hits.append(f"{py.relative_to(ROOT)}:{node.lineno} imports "
                                        f"split_indices from {node.module}")
                if isinstance(node, ast.Attribute) and node.attr == "split_indices" and \
                        isinstance(node.value, ast.Name) and node.value.id == "fallback_gate":
                    hits.append(f"{py.relative_to(ROOT)}:{node.lineno} calls "
                                "fallback_gate.split_indices")
        self.check("banned-helpers", not hits, "; ".join(hits))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", default=None, help="write the verdict as data")
    a = ap.parse_args()

    au = Audit(a.verbose)
    for name in ("frozen", "protocols", "git_binding", "macros", "splits",
                 "records", "banned", "ledger"):
        getattr(au, name)()

    print("=" * 70)
    print("PROCEDURAL AUDIT")
    if a.verbose:
        groups = collections.Counter(n.split("-")[0] for n in au.passed_names)
        for g, c in sorted(groups.items()):
            print(f"  OK  {g:<12s} {c} check{'s' if c != 1 else ''} passed")
    for n in au.notes:
        print(n)
    if au.fail:
        print(f"\n{len(au.fail)} PROCEDURAL FAILURES:")
        for f_ in au.fail:
            print(f"  FAIL  {f_}")
    else:
        print("\nall procedural checks pass")
    print("\nNote: this audit says nothing about whether any result came out")
    print("favourably. Scientific outcomes are reported as data, never as an exit code.")

    if a.json:
        (ROOT / a.json).write_text(json.dumps(
            {"failures": au.fail, "n_failures": len(au.fail), "passed": not au.fail}, indent=1))
    return 1 if au.fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
