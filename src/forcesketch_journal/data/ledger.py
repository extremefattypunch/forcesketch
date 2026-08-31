"""Append-only ledger for every access to a test split.

The plan asks for this because the project is executed by a long-running agent,
and the specific failure it guards against is the one nobody notices: looking at
the test set, adjusting something, and looking again. No single access is wrong;
the *sequence* is. A ledger makes the sequence visible.

Test indices are obtainable only through `test_indices()`, which writes an
append-only entry recording who asked, for which system, under which split, and
why. `tools/audit.py` then reports the access pattern and fails when an
`experiment_id` has touched the same system's test set under **two different
splits** — that is the signature of a split being changed underneath a result.

**Honest limit, stated because a ledger implies more than it can deliver.** This
was added late. It cannot retroactively certify the accesses that produced the
existing results; it can only bind future ones. What protects the earlier work is
different and weaker: the split manifests are content-hashed and recompute from
their recorded seeds (`tools/audit.py`), the pre-registration protocols are
sha256-bound, and the analyses that fit anything do so on the design role. A
reader should treat the ledger as a forward commitment, not as evidence about
what already happened.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
LEDGER = ROOT / "manifests/test_access_ledger.jsonl"


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


@contextlib.contextmanager
def test_indices(system: str, *, experiment_id: str, purpose: str,
                 scheme: str = "contiguous_block",
                 ledger: pathlib.Path | None = None):
    """Yield the test-role indices for `system`, recording the access.

    `purpose` is required and free-text: an access nobody can explain later is an
    access that should not have happened.
    """
    if not experiment_id or not purpose:
        raise ValueError("experiment_id and purpose are both required to read a test split")
    man = ROOT / f"manifests/splits/{system}__{scheme}.json"
    if not man.exists():
        raise FileNotFoundError(f"no split manifest for {system} under {scheme}")
    d = json.loads(man.read_text())
    idx = np.asarray(d["roles"]["test"])

    path = ledger or LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps({
            "utc": _now(), "experiment_id": experiment_id, "system": system,
            "scheme": scheme, "purpose": purpose,
            "split_sha256": d["content_sha256"], "n_test": len(idx),
        }, sort_keys=True) + "\n")
    yield idx


def read_ledger(path: pathlib.Path | None = None) -> list[dict]:
    p = path or LEDGER
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open() if l.strip()]


def audit_ledger(path: pathlib.Path | None = None) -> dict:
    """Access pattern, and the one condition that is a hard failure.

    Repeated access is reported, not failed: re-running an analysis unchanged is
    legitimate and common. What fails is the same experiment reading the same
    system's test set under two DIFFERENT split hashes, because then whichever
    number was published cannot be attributed to a split.
    """
    rows = read_ledger(path)
    by: dict[tuple[str, str], set[str]] = {}
    counts: dict[tuple[str, str], int] = {}
    for r in rows:
        k = (r["experiment_id"], r["system"])
        by.setdefault(k, set()).add(r["split_sha256"])
        counts[k] = counts.get(k, 0) + 1
    conflicts = [{"experiment_id": e, "system": s, "n_distinct_splits": len(v)}
                 for (e, s), v in by.items() if len(v) > 1]
    repeats = [{"experiment_id": e, "system": s, "accesses": c}
               for (e, s), c in sorted(counts.items()) if c > 1]
    return {"n_entries": len(rows), "n_pairs": len(by),
            "conflicts": conflicts, "repeated_accesses": repeats,
            "ok": not conflicts}
