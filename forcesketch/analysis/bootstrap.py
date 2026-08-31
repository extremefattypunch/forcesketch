"""Paired bootstrap over structures (spec SS47).

Every approximate score is compared with the exact score on the IDENTICAL
structure, and the bootstrap resamples structures -- not seeds -- so the interval
answers "how much would this number move on a different draw of 3BPA
configurations?"

Two sources of variability are reported separately, because they mean different
things and the spec asks for both:

  * `ci_lo` / `ci_hi`  -- 95% percentile CI from resampling STRUCTURES, computed
    on the seed-averaged statistic. This is the interval to quote.
  * `seed_std`         -- spread across the ten frozen probe seeds at fixed data
    (spec SS36's "seed-to-seed variance").

Implementation note on ties. A bootstrap resample contains duplicated structures,
so tied scores occur where the underlying distribution is continuous. Exact and
approximate scores are gathered with the SAME indices, so duplicates are paired
and the effect on rank correlation is symmetric; we use ordinal ranks and note
this rather than silently averaging ranks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class Interval:
    point: float
    ci_lo: float
    ci_hi: float
    seed_std: float
    n_boot: int

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.ci_lo:.3f}, {self.ci_hi:.3f}]"

    def as_dict(self, prefix: str) -> dict:
        return {prefix: self.point, f"{prefix}_ci_lo": self.ci_lo,
                f"{prefix}_ci_hi": self.ci_hi, f"{prefix}_seed_std": self.seed_std,
                f"{prefix}_n_boot": self.n_boot}


def _ranks(x: torch.Tensor) -> torch.Tensor:
    """Ordinal ranks along the last axis."""
    return x.argsort(dim=-1).argsort(dim=-1).to(x.dtype)


def _spearman_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise Spearman for [..., n] tensors -> [...]."""
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean(dim=-1, keepdim=True)
    rb = rb - rb.mean(dim=-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp_min(1e-30)


def _topp_recall_rows(exact: torch.Tensor, approx: torch.Tensor, p: float) -> torch.Tensor:
    """Row-wise top-p% recall for [..., n] tensors -> [...]."""
    n = exact.shape[-1]
    k = max(1, int(round(p * n)))
    e_idx = exact.topk(k, dim=-1).indices
    a_idx = approx.topk(k, dim=-1).indices
    e_mask = torch.zeros_like(exact, dtype=torch.bool).scatter_(-1, e_idx, True)
    a_mask = torch.zeros_like(approx, dtype=torch.bool).scatter_(-1, a_idx, True)
    return (e_mask & a_mask).sum(-1).to(exact.dtype) / k


def _topp_jaccard_rows(exact: torch.Tensor, approx: torch.Tensor, p: float) -> torch.Tensor:
    n = exact.shape[-1]
    k = max(1, int(round(p * n)))
    e_idx = exact.topk(k, dim=-1).indices
    a_idx = approx.topk(k, dim=-1).indices
    e_mask = torch.zeros_like(exact, dtype=torch.bool).scatter_(-1, e_idx, True)
    a_mask = torch.zeros_like(approx, dtype=torch.bool).scatter_(-1, a_idx, True)
    inter = (e_mask & a_mask).sum(-1).to(exact.dtype)
    union = (e_mask | a_mask).sum(-1).to(exact.dtype)
    return inter / union.clamp_min(1)


METRICS = {
    "spearman": lambda e, a: _spearman_rows(e, a),
    "top5_recall": lambda e, a: _topp_recall_rows(e, a, 0.05),
    "top1_recall": lambda e, a: _topp_recall_rows(e, a, 0.01),
    "top10_recall": lambda e, a: _topp_recall_rows(e, a, 0.10),
    "top5_jaccard": lambda e, a: _topp_jaccard_rows(e, a, 0.05),
}


def paired_bootstrap(
    s_exact: torch.Tensor,
    s_approx: torch.Tensor,
    *,
    metrics: tuple[str, ...] = ("spearman", "top5_recall", "top5_jaccard"),
    n_boot: int = 1000,
    seed: int = 20260902,
    alpha: float = 0.05,
    device: str | None = None,
    chunk: int = 250,
) -> dict[str, Interval]:
    """`s_exact [S]`, `s_approx [n_seeds, S]` -> {metric: Interval}.

    For each bootstrap resample of structures we evaluate the metric per probe
    seed and then average over seeds, so the interval is on the number actually
    reported (the seed-averaged statistic), not on a single seed's value.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    e = s_exact.to(dev, torch.float32)
    a = s_approx.to(dev, torch.float32)
    n_seeds, S = a.shape

    g = torch.Generator(device="cpu").manual_seed(seed)
    point = {m: float(METRICS[m](e.expand(n_seeds, S), a).mean()) for m in metrics}
    seed_std = {m: float(METRICS[m](e.expand(n_seeds, S), a).std()) for m in metrics}

    draws: dict[str, list[torch.Tensor]] = {m: [] for m in metrics}
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = torch.randint(0, S, (b, S), generator=g).to(dev)     # [b, S]
        e_bs = e[idx]                                              # [b, S]
        a_bs = a[:, idx]                                           # [n_seeds, b, S]
        e_bs = e_bs.unsqueeze(0).expand(n_seeds, b, S)
        for m in metrics:
            draws[m].append(METRICS[m](e_bs, a_bs).mean(dim=0).cpu())  # avg over seeds
        done += b

    out = {}
    for m in metrics:
        d = torch.cat(draws[m]).numpy()
        lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[m] = Interval(point[m], float(lo), float(hi), seed_std[m], n_boot)
    return out


def paired_bootstrap_difference(
    s_exact: torch.Tensor,
    s_a: torch.Tensor,
    s_b: torch.Tensor,
    *,
    metric: str = "top5_recall",
    n_boot: int = 1000,
    seed: int = 20260902,
    alpha: float = 0.05,
    device: str | None = None,
) -> Interval:
    """CI on metric(A) - metric(B) under the SAME structure resamples.

    This is the statistic that decides spec SS49 -- "does ForceSketch beat head
    subsampling?" -- because a difference of two separately-computed CIs is not a
    test. If this interval excludes zero, the ordering is real.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    e = s_exact.to(dev, torch.float32)
    A, B = s_a.to(dev, torch.float32), s_b.to(dev, torch.float32)
    S = e.shape[0]
    fn = METRICS[metric]

    point = float(fn(e.expand(*A.shape), A).mean() - fn(e.expand(*B.shape), B).mean())
    g = torch.Generator(device="cpu").manual_seed(seed)
    diffs = []
    done = 0
    while done < n_boot:
        b = min(250, n_boot - done)
        idx = torch.randint(0, S, (b, S), generator=g).to(dev)
        eb = e[idx]
        da = fn(eb.unsqueeze(0).expand(A.shape[0], b, S), A[:, idx]).mean(dim=0)
        db = fn(eb.unsqueeze(0).expand(B.shape[0], b, S), B[:, idx]).mean(dim=0)
        diffs.append((da - db).cpu())
        done += b
    d = torch.cat(diffs).numpy()
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(point, float(lo), float(hi), 0.0, n_boot)
