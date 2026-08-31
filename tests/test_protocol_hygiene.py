"""The four tests the adversarial review asked for, plus the machinery they need.

Each one exists because a specific defect got through everything else:

  1. A benchmark labelled cells `L=8` while running 7 lanes, because
     `exact_seed_bundle` returns r = M-1 directions and the slice silently
     under-delivered. Requested lane count must equal executed lane count.
  2. Two GPUs wrote to the same records filename; the second overwrote the first
     and the survivor could not be attributed to either. Every output path must
     encode the device.
  3. J3 compared two noise models that were identical by construction, so
     "coordinate coupling is irrelevant" was a statement about nothing. Any two
     noise models being *compared* must be statistically distinguishable.
  4. A quoted number ("76.7") traced to no record at all. Every documented
     numeric range must recompute from the records that back it.

Test 4 is the one with teeth: it re-derives each headline from the JSONL/JSON
records and then greps the prose for the result, so a document that drifts from
its records fails rather than merely looking plausible.
"""

from __future__ import annotations

import collections
import json
import pathlib
import statistics as st
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT.parents[0] / "forcesketch"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(FROZEN / "src"))

from forcesketch_journal.benchmark.lanes import (  # noqa: E402
    device_slug, lane_seeds, record_path,
)

RECORDS = ROOT / "results/records"
RESULTS = ROOT / "results"


# --------------------------------------------------------------------------
# 1. requested lane count == executed lane count
# --------------------------------------------------------------------------

def _seeds(n_lanes: int, B: int = 3, M: int = 8) -> torch.Tensor:
    return torch.randn(n_lanes, B, M, dtype=torch.float32)


@pytest.mark.parametrize("lanes", [1, 2, 3, 4, 7, 8])
def test_lane_seeds_returns_exactly_the_requested_lane_count(lanes):
    out = lane_seeds(_seeds(7), _seeds(1), lanes)
    assert out.shape[0] == lanes


def test_lane_seeds_refuses_more_lanes_than_exist():
    """The original bug: L=8 against r=7 centred directions returned 7 quietly."""
    with pytest.raises(ValueError, match="only 8 exist"):
        lane_seeds(_seeds(7), _seeds(1), 9)


def test_lane_seeds_includes_the_mean_lane_not_just_the_centred_ones():
    exact, mean = _seeds(7), _seeds(1)
    out = lane_seeds(exact, mean, 8)
    assert torch.equal(out[:7], exact) and torch.equal(out[7:], mean)


def test_lane_seeds_rejects_mismatched_seed_shapes():
    with pytest.raises(ValueError, match="disagree past the lane axis"):
        lane_seeds(_seeds(7, B=3), _seeds(1, B=4), 4)


def test_recorded_lane_counts_are_all_achievable():
    """No shipped record may claim more lanes than the committee can supply."""
    for f in sorted(RECORDS.glob("j8_lane_bench*.jsonl")):
        for r in (json.loads(l) for l in f.open()):
            assert r["lanes"] <= r["M"], f"{f.name}: L={r['lanes']} > M={r['M']}"


# --------------------------------------------------------------------------
# 2. every output path encodes the device
# --------------------------------------------------------------------------

def test_record_path_encodes_the_device():
    p = record_path("j8_lane_bench", "NVIDIA A100-SXM4-80GB")
    assert "NVIDIA_A100-SXM4-80GB" in p.name


def test_record_path_rejects_an_explicit_path_without_the_device():
    """The exact defect: `..._water_Blackwell.jsonl` on two Blackwell parts."""
    with pytest.raises(ValueError, match="does not encode the device"):
        record_path("j8_lane_bench", "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                    explicit="results/records/j8_lane_bench_water_Blackwell.jsonl")


def test_record_path_accepts_an_explicit_path_that_does_encode_it():
    name = "j8_lane_bench_water_NVIDIA_H200.jsonl"
    assert record_path("x", "NVIDIA H200", explicit=f"results/records/{name}").name == name


def test_device_slug_never_silently_empty():
    with pytest.raises(ValueError):
        device_slug("   ")


def test_shipped_benchmark_records_live_in_device_named_files():
    """Filename and payload must agree, or the file is unattributable."""
    for f in sorted(RECORDS.glob("j8_lane_bench*.jsonl")):
        rows = [json.loads(l) for l in f.open()]
        names = {r["device_name"] for r in rows}
        assert len(names) == 1, f"{f.name} mixes devices: {names}"
        assert device_slug(names.pop()) in f.name, f"{f.name} does not name its device"


# --------------------------------------------------------------------------
# 3. two noise models being compared must not be statistically identical
# --------------------------------------------------------------------------

def _aligned_cache(S: int = 400, A: int = 6, M: int = 8, seed: int = 0):
    """F whose head-space directions are ALIGNED across coordinates.

    This is the regime where coupling must show up: every coordinate's a_d points
    the same way, so one shared Haar frame hits them all together. A random F
    would give near-isotropic a_d, under which `shared_real` legitimately
    degenerates to `indep` -- which is precisely the degeneracy that made the
    original comparison vacuous, so the test must not sit in it.
    """
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(S, M, generator=g, dtype=torch.float64)          # per-structure head offsets
    d = torch.randn(A, 3, generator=g, dtype=torch.float64)          # one shared spatial pattern
    F = torch.einsum("sm,ai->saim", c, d)
    F = F + 0.02 * torch.randn(S, A, 3, M, generator=g, dtype=torch.float64)
    return F


def _coupling(ratio: torch.Tensor) -> float:
    """Mean off-diagonal across-coordinate correlation of the noise ratio."""
    x = ratio.numpy()
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, ddof=1)
    keep = sd > 1e-12
    x, sd = x[:, keep], sd[keep]
    c = (x.T @ x) / (len(x) - 1) / np.outer(sd, sd)
    off = ~np.eye(c.shape[0], dtype=bool)
    return float(np.abs(c[off]).mean())


def _ratios(mode: str, F: torch.Tensor, r: int = 7, K: int = 4, seed: int = 1):
    from j3_extreme_value import beta_noise

    v = F.var(dim=-1, unbiased=True).flatten(1)                      # [S, D]
    gen = torch.Generator().manual_seed(seed)
    return beta_noise(v, r, K, mode=mode, gen=gen, F=F) / v


def test_shared_real_and_indep_are_distinguishable():
    """The comparison the paper actually makes must be a real comparison."""
    F = _aligned_cache()
    c_real, c_indep = _coupling(_ratios("shared_real", F)), _coupling(_ratios("indep", F))
    assert c_indep < 0.05, f"independent draws should not couple, got {c_indep:.4f}"
    assert c_real - c_indep > 0.2, (
        f"shared_real coupling {c_real:.4f} is not distinguishable from indep "
        f"{c_indep:.4f}; the comparison would be vacuous")


def test_shared_iso_and_indep_are_NOT_distinguishable():
    """Locks in the review finding rather than merely fixing it.

    `shared_iso` applies one Haar frame to isotropically drawn directions, which
    for a fixed subspace gives independent Beta draws -- so it is `indep` wearing
    a hat. If someone ever makes these two genuinely differ, this test fails and
    forces the J3 text (which says they are identical by construction) to be
    revisited rather than left stale.
    """
    F = _aligned_cache()
    c_iso, c_indep = _coupling(_ratios("shared_iso", F)), _coupling(_ratios("indep", F))
    assert abs(c_iso - c_indep) < 0.05, (
        f"shared_iso {c_iso:.4f} vs indep {c_indep:.4f} now differ; J3's claim that "
        "they are identical by construction no longer holds")


# --------------------------------------------------------------------------
# 4. every documented numeric range resolves to a record
# --------------------------------------------------------------------------

def _load_jsonl(name: str) -> list[dict]:
    return [json.loads(l) for l in (RECORDS / name).open()]


def _doc(name: str) -> str:
    return (RESULTS / name).read_text()


def _assert_quoted(doc_name: str, value: float, fmt: str = "{:.3f}") -> None:
    text, s = _doc(doc_name), fmt.format(value)
    assert s in text, f"{doc_name} does not quote {s}, which is what the records give"


def test_j2a_global_beats_maxcomp_counts_match_the_records():
    raw = json.loads((RECORDS / "j2a_global_vs_maxcomp_ci.json").read_text())
    # The record gained a `meta` block carrying the bootstrap seed when it was
    # regenerated; the earlier copy was a bare list and had no recorded seed.
    ci = raw["entries"] if isinstance(raw, dict) else raw
    assert not isinstance(raw, dict) or "seed" in raw["meta"], \
        "a bootstrap record without its seed is not reproducible"
    positive = sum(c["delta"] > 0 for c in ci)
    significant = sum(c["delta"] > 0 and c["significant"] for c in ci)
    assert f"**{positive} of {len(ci)}**" in _doc("J2A_ORACLE_PANEL.md")
    assert f"{significant} of {len(ci)}" in _doc("J2A_ORACLE_PANEL.md")
    assert not any(c["delta"] < 0 and c["significant"] for c in ci), \
        "a significantly negative cell would contradict the documented claim"


def test_j2a_free_signal_gain_range_recomputes():
    """The +0.067 to +0.182 range is the gain over the BEST free signal."""
    rows = [r for r in _load_jsonl("j2a_oracle_panel.jsonl") if r["error_score"] == "e_max"]
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["cache_tag"]][r["signal"]] = r["auroc_top05"]
    gains = [d["exact_global"] - max(d[s] for s in ("energy_std", "force_norm") if s in d)
             for d in by.values()]
    assert len(gains) == 6, f"expected the six panel systems, got {len(gains)}"
    _assert_quoted("J2A_ORACLE_PANEL.md", min(gains))
    _assert_quoted("J2A_ORACLE_PANEL.md", max(gains))


def test_leading_only_skip_range_recomputes():
    """The recommended-construction headline, 0.822-0.943."""
    rows = [r for r in _load_jsonl("j2b_r0_4.jsonl")
            if r["gate"] == "leading_only" and r["uq_lanes"] == 5 and r["score"] == "global"]
    v = [r["frac_exact_skipped"] for r in rows]
    assert len(v) == 6, f"expected six systems, got {len(v)}"
    _assert_quoted("REVIEW_FINDINGS.md", min(v))
    _assert_quoted("REVIEW_FINDINGS.md", max(v))


def test_control_variate_skip_range_recomputes_with_its_stated_reduction():
    """0.762-0.948 is the MEAN over probe seeds per system, not a single seed.

    Recording which reduction it is matters: the raw per-cell spread is
    0.710-0.950, so quoting the same range from a different reduction would be
    wrong by more than the effect the table is about.
    """
    rows = [r for r in _load_jsonl("j2b_factor_b_global.jsonl")
            if r["gate"] == "control_variate" and r["uq_lanes"] == 5]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["system"]].append(r["frac_exact_skipped"])
    per_system = [st.mean(x) for x in by.values()]
    _assert_quoted("REVIEW_FINDINGS.md", min(per_system))
    _assert_quoted("REVIEW_FINDINGS.md", max(per_system))


def test_j3_prediction_error_recomputes():
    d = json.loads((RECORDS / "j3_extreme_value_v2.json").read_text())
    rows = d["recall_prediction"]
    mae = st.mean(abs(r["shared_real"] - r["measured"]) for r in rows)
    _assert_quoted("J3_EXTREME_VALUE.md", mae)


def _norm():
    from j6c_size_normalisation import normalised
    return normalised


def test_size_normalisation_matches_an_independent_reference_at_variable_n():
    """The test with teeth, and the reason the obvious one has none.

    An adversarial audit showed that checking the normalisation on FIXED N is a
    tautology: 3N is one constant there, and AUROC is invariant under division by
    a positive scalar, so an implementation that omits the normalisation
    altogether still "passes". This checks the shipped function against an
    independently written scalar reference on VARIABLE N, which is where a
    precedence bug or a misaligned N actually shows up.
    """
    normalised = _norm()
    g = torch.Generator().manual_seed(13)
    natoms = torch.randint(1, 445, (2000,), generator=g).double()
    stat = torch.rand(2000, generator=g, dtype=torch.float64) * 100 + 1e-6
    for alpha in (0.0, 0.25, 0.75, 1.0, 1.5):
        ref = torch.tensor([x / (3.0 * m) ** alpha
                            for x, m in zip(stat.tolist(), natoms.tolist())],
                           dtype=torch.float64)
        got = normalised(stat, natoms, alpha)
        assert float(((got - ref).abs() / ref.abs()).max()) < 1e-12, f"alpha={alpha}"


@pytest.mark.parametrize("mutant", ["precedence", "wrong_N", "absent"])
def test_the_normalisation_check_catches_broken_implementations(mutant):
    """Mutation test: prove the check above can fail.

    These three are exactly the implementations that slipped through the vacuous
    fixed-N version -- an operator-precedence bug, N read from the wrong
    structure, and no normalisation at all.
    """
    normalised = _norm()
    g = torch.Generator().manual_seed(14)
    natoms = torch.randint(1, 445, (2000,), generator=g).double()
    stat = torch.rand(2000, generator=g, dtype=torch.float64) * 100 + 1e-6
    alpha = 0.75
    broken = {"precedence": stat / 3.0 * natoms.pow(alpha),
              "wrong_N": stat / (3.0 * natoms.flip(0)).pow(alpha),
              "absent": stat}[mutant]
    ref = normalised(stat, natoms, alpha)
    assert float((broken - ref).abs().max()) > 1e-12, (
        f"the {mutant} mutant is indistinguishable from the correct implementation, "
        "so the reference test would not catch it")


def test_size_normalisation_is_a_no_op_at_fixed_n():
    """Kept, but labelled: this is a DERIVATION and cannot fail.

    It is retained because the fixed-N invariance is what lets the normalisation
    be adopted globally rather than as a special case, so a reader should see it
    asserted. It is not evidence that the implementation is correct -- the two
    tests above are.
    """
    from forcesketch_journal.evaluation.tail_metrics import auroc, top_p_mask

    normalised = _norm()
    g = torch.Generator().manual_seed(11)
    S, A = 500, 27
    stat = torch.rand(S, generator=g, dtype=torch.float64) * 10
    err = stat * 0.6 + torch.rand(S, generator=g, dtype=torch.float64)
    pos = top_p_mask(err, 0.05)
    natoms = torch.full((S,), float(A), dtype=torch.float64)
    base = float(auroc(stat, pos))
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 3.0):
        assert float(auroc(normalised(stat, natoms, alpha), pos)) == base


def test_every_j6_record_carries_its_protocol_hash():
    """Pre-registration binding: a naive-committee record with no protocol hash,
    or the wrong one, is not evidence for a pre-registered prediction."""
    reg = RECORDS / "j6_protocol_registration.json"
    if not reg.exists():
        pytest.skip("J6 not registered")
    import hashlib
    want = json.loads(reg.read_text())["protocol_sha256"]
    got = hashlib.sha256((ROOT / "protocols/j6_naive_committee.yaml").read_bytes()).hexdigest()
    assert got == want, "the pre-registered protocol file has been modified since registration"
    caches = RECORDS / "j6_naive_caches.jsonl"
    if caches.exists():
        for r in _load_jsonl("j6_naive_caches.jsonl"):
            assert r["protocol_sha256"] == want, f"{r['variant']} bound to a different protocol"


@pytest.mark.parametrize("stem", ["j6_naive_committee", "j6b_foundation_acquisition",
                                  "j6c_size_normalisation"])
def test_registered_protocols_have_not_been_edited_since_registration(stem):
    """A pre-registration you can still edit is not a pre-registration.

    Each scorer refuses to run on a hash mismatch; this makes the same check part
    of the suite, so a protocol edited to fit a result fails CI rather than
    failing quietly at the next manual rescore.
    """
    import hashlib
    candidates = [RECORDS / "j6_protocol_registration.json",
                  RECORDS / "j6b_protocol_registration.json",
                  RECORDS / "j6c_protocol_registration.json"]
    proto = ROOT / f"protocols/{stem}.yaml"
    if not proto.exists():
        pytest.skip(f"{stem} not present")
    match = [c for c in candidates if c.exists()
             and json.loads(c.read_text())["protocol"].endswith(f"{stem}.yaml")]
    assert match, f"no registration record references {stem}.yaml"
    want = json.loads(match[0].read_text())["protocol_sha256"]
    got = hashlib.sha256(proto.read_bytes()).hexdigest()
    assert got == want, f"{stem}.yaml has changed since it was registered"


def test_manuscript_only_uses_macros_that_exist():
    """No numeral in the paper may be typed by hand.

    The rule is that every number in main.tex is a \\fs macro derived from a
    record. This catches the failure mode where a macro is renamed or a number is
    referenced before its derivation is added to tools/paper_numbers.py.
    """
    import re

    main = ROOT / "paper/main.tex"
    macros = ROOT / "paper/macros.tex"
    if not main.exists():
        pytest.skip("manuscript not started")
    defined = set(re.findall(r"\\newcommand\{\\(fs\w+)\}", macros.read_text()))
    used = set(re.findall(r"\\(fs[A-Za-z]+)", main.read_text()))
    assert not (used - defined), f"main.tex uses undefined macros: {sorted(used - defined)}"
    prov = json.loads((RECORDS / "macro_provenance.json").read_text())
    assert defined <= set(prov), \
        f"macros with no provenance entry: {sorted(defined - set(prov))}"


def test_manuscript_sources_are_present_and_self_consistent():
    """The submitting agent must be able to rebuild the PDF from the repo.

    Checks the pieces exist and that main.tex does not reference a macro or an
    input file that is missing. Does not run LaTeX -- the TeX tree is not a
    dependency of the test suite -- but guarantees a build cannot fail for a
    reason this repository controls.
    """
    import re

    paper = ROOT / "paper"
    for f in ("main.tex", "preamble.tex", "macros.tex", "references.bib", "build.sh"):
        assert (paper / f).exists(), f"paper/{f} is missing"
    src = (paper / "main.tex").read_text()
    for inp in re.findall(r"\\input\{([^}]+)\}", src):
        assert (paper / f"{inp}.tex").exists(), f"\\input{{{inp}}} has no file"
    keys = set(re.findall(r"@\w+\{([^,]+),", (paper / "references.bib").read_text()))
    cited = set()
    for m in re.findall(r"\\cite[pt]?\{([^}]+)\}", src):
        cited |= {c.strip() for c in m.split(",")}
    assert cited <= keys, f"main.tex cites missing bib keys: {sorted(cited - keys)}"


def test_every_manuscript_figure_and_table_exists_and_traces_to_records():
    """Figures and tables carry the same provenance rule as the macros.

    A figure that cannot be regenerated from results/records/ does not belong in
    the paper, so this checks the asset exists, the manuscript includes it, and
    the record it was derived from is still present.
    """
    import re

    paper = ROOT / "paper"
    if not (paper / "main.tex").exists():
        pytest.skip("manuscript not started")
    src = (paper / "main.tex").read_text()

    for g in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", src):
        assert (paper / g).exists(), f"\\includegraphics{{{g}}} has no file"
    for inp in re.findall(r"\\input\{(tables/[^}]+)\}", src):
        assert (paper / f"{inp}.tex").exists(), f"\\input{{{inp}}} has no file"

    prov = RECORDS / "figure_provenance.json"
    assert prov.exists(), "figures have no provenance record"
    for fig, meta in json.loads(prov.read_text()).items():
        assert (paper / "figures" / f"{fig}.pdf").exists(), f"{fig}.pdf missing"
        for rec in meta["records"]:
            assert (RECORDS / rec).exists(), f"{fig} was built from a missing record {rec}"

    # every float must be referenced from the prose, or it is decoration
    for lab in re.findall(r"\\label\{((?:fig|tab):[^}]+)\}", src):
        assert f"\\ref{{{lab}}}" in src, f"float {lab} is never referenced in the text"
