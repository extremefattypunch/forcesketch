"""The single split implementation, and its manifest format.

The frozen tree has two divergent ones -- `screening.fallback_gate.split_indices`
uses `int(round(0.2*n))` while `scripts/07_gate_baselines.py` inlines
`int(0.2*S)`, so for most S they disagree by a structure and produce different
index sets. Neither writes a manifest, so nothing downstream can state which was
used. Both stay frozen and neither is ever called from here.

Three roles, used strictly:

    design  -- learn Q_r0, choose K/r0/score/method, and fit tau
    cal     -- estimate the conformal multiplier c_alpha, and nothing else
    test    -- touched once, at the end

Two schemes are supported on purpose. `contiguous_block` is the defensible one
for trajectory data: cut the time-ordered series into blocks, assign whole blocks
to roles, and drop a guard band at every role boundary so no two structures in
different roles are within one correlation time of each other. `random_frame`
reproduces the workshop's `torch.randperm` behaviour. Running both and reporting
the difference turns "your splits leak" from an objection into a measurement.

`content_sha256` deliberately covers only the structure ids and the role
assignment -- not timestamps, paths, or tau estimates -- so that a regenerated
manifest hashes identically iff the assignment is identical. That is the property
the audit needs to be able to check.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass

import numpy as np

ROLES = ("design", "cal", "test")


@dataclass(frozen=True)
class Split:
    system: str
    scheme: str
    design: np.ndarray
    cal: np.ndarray
    test: np.ndarray
    dropped: np.ndarray
    guard: int
    tau_int: float
    n: int
    sha256: str

    def indices(self, role: str) -> np.ndarray:
        return {"design": self.design, "cal": self.cal, "test": self.test}[role]

    def assert_valid(self) -> None:
        d, c, t, x = (set(map(int, a)) for a in (self.design, self.cal, self.test, self.dropped))
        assert not (d & c) and not (d & t) and not (c & t), "roles overlap"
        assert d | c | t | x == set(range(self.n)), "roles + dropped do not partition"
        assert not (x & (d | c | t)), "dropped index also assigned a role"


def _hash(system: str, scheme: str, roles: dict[str, np.ndarray]) -> str:
    payload = json.dumps(
        {"system": system, "scheme": scheme,
         **{r: sorted(int(i) for i in roles[r]) for r in ROLES}},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def contiguous_block_split(
    n: int, *, system: str, seed: int, tau_int: float, guard: int,
    n_blocks: int = 10, counts: tuple[int, int, int] = (2, 2, 6),
) -> Split:
    """Cut the time-ordered index range into `n_blocks`, assign whole blocks.

    Blocks are permuted before assignment so every role samples the whole
    trajectory -- otherwise a slow drift in the trajectory would be perfectly
    confounded with the role, which is a worse problem than the leakage the
    blocking is meant to fix.

    Guard frames are dropped on both sides of every boundary where the role
    actually changes. Dropping at same-role boundaries would discard data for
    nothing.
    """
    assert sum(counts) == n_blocks
    edges = np.linspace(0, n, n_blocks + 1).round().astype(int)
    labels = np.array(sum(([r] * c for r, c in zip(ROLES, counts)), []))
    # Mix the system name into the seed. Previously the permutation depended only
    # on `seed`, so EVERY system received the identical block assignment -- and the
    # canonical seed happened to place the two calibration blocks adjacently (a
    # ~20% event), making the calibration split one contiguous run spanning ~19% of
    # each trajectory. That both contradicted the anti-confound rationale below and
    # made cross-system agreement correlated rather than independent evidence.
    sysseed = int.from_bytes(hashlib.blake2b(system.encode(), digest_size=8).digest(), "big")
    labels = labels[np.random.default_rng((seed ^ sysseed) % (2**63)).permutation(n_blocks)]

    role_of = np.empty(n, dtype=object)
    for b in range(n_blocks):
        role_of[edges[b]:edges[b + 1]] = labels[b]

    drop = np.zeros(n, dtype=bool)
    for b in range(1, n_blocks):
        if labels[b] != labels[b - 1]:
            e = edges[b]
            drop[max(0, e - guard):min(n, e + guard)] = True

    idx = np.arange(n)
    roles = {r: idx[(role_of == r) & ~drop] for r in ROLES}
    return Split(system=system, scheme="contiguous_block", **roles,
                 dropped=idx[drop], guard=guard, tau_int=float(tau_int), n=n,
                 sha256=_hash(system, "contiguous_block", roles))


def random_frame_split(
    n: int, *, system: str, seed: int, fracs: tuple[float, float, float] = (0.2, 0.2, 0.6),
) -> Split:
    """The workshop scheme: a frozen permutation, no temporal structure.

    Retained as a named, hashed, auditable scheme so the blocked-vs-random
    comparison is a controlled experiment rather than a code change.
    """
    perm = np.random.default_rng(seed).permutation(n)
    n1 = int(round(fracs[0] * n))
    n2 = n1 + int(round(fracs[1] * n))
    roles = {"design": np.sort(perm[:n1]), "cal": np.sort(perm[n1:n2]), "test": np.sort(perm[n2:])}
    return Split(system=system, scheme="random_frame", **roles,
                 dropped=np.array([], dtype=int), guard=0, tau_int=float("nan"), n=n,
                 sha256=_hash(system, "random_frame", roles))


def alpha_feasible(n_cal: int, alpha: float) -> tuple[bool, int]:
    """Split conformal needs the ceil((n+1)(1-alpha))-th order statistic to exist.

    Returns (feasible, k). Infeasible means `fallback_gate.calibrate` returns
    +inf and the gate skips nothing -- which must be recorded as `underpowered`
    rather than reported as a 0% skip rate.
    """
    k = int(np.ceil((n_cal + 1) * (1.0 - alpha)))
    return k <= n_cal, k


def coverage_law(n_cal: int, alpha: float) -> dict:
    """Realised coverage of a split-conformal gate is Beta(k, n+1-k) over calibration draws.

    Reporting only the nominal 1-alpha hides that a small calibration set makes
    the *achieved* coverage a random variable with real spread.
    """
    from scipy.stats import beta as beta_dist
    ok, k = alpha_feasible(n_cal, alpha)
    if not ok:
        return {"feasible": False, "k": k, "n_cal": n_cal, "alpha": alpha}
    a, b = k, n_cal + 1 - k
    return {"feasible": True, "k": k, "n_cal": n_cal, "alpha": alpha,
            "mean": float(a / (a + b)), "sd": float(beta_dist(a, b).std()),
            "q05": float(beta_dist(a, b).ppf(0.05)), "q95": float(beta_dist(a, b).ppf(0.95))}


def write_manifest(split: Split, path: pathlib.Path, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 1, "system": split.system, "scheme": split.scheme,
        "n_structures": split.n, "guard_frames": split.guard, "tau_int": split.tau_int,
        "n_design": len(split.design), "n_cal": len(split.cal), "n_test": len(split.test),
        "n_dropped": len(split.dropped),
        "roles": {r: [int(i) for i in split.indices(r)] for r in ROLES},
        "dropped": [int(i) for i in split.dropped],
        "content_sha256": split.sha256, **extra,
    }, sort_keys=True, indent=1))
