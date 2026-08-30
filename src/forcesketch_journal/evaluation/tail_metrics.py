"""Rank- and tail-oriented metrics for scoring an uncertainty signal against a
reference error.

Everything here is rank-based and vectorised over a leading resample axis, so the
same functions serve point estimates and bootstrap replicates without a second
implementation. That constraint is why AUROC is written as a Mann-Whitney rank
statistic rather than by sweeping a threshold: the rank form is O(n log n), exact
in the presence of ties, and differentiates cleanly over a [B, n] batch.

Positives are always defined as the top-p fraction of the ERROR score, never of
the uncertainty signal. Confusing the two is the single easiest way to report a
number that looks like physical validation but is actually self-consistency.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _rank(x: Tensor) -> Tensor:
    """Average ranks along the last axis, ties shared (1-indexed).

    Ties matter here: force-error scores are continuous so exact ties are rare,
    but a degenerate signal (a constant, or the random null under a fixed seed)
    must score exactly 0.5 AUROC rather than 1.0, and only tie-averaged ranks
    give that.

    Fully vectorised over the leading (resample) axes: ranks within a tie-run are
    consecutive integers, so the run mean is just (first + last) / 2, and the run
    endpoints come from a scatter_reduce over a cumsum-derived group id. That
    keeps a 10,000 x 2,139 bootstrap in seconds rather than a Python loop.
    """
    n = x.shape[-1]
    order = x.argsort(dim=-1)
    xs = x.gather(-1, order)

    pos = torch.arange(1, n + 1, dtype=x.dtype, device=x.device).expand_as(xs)
    is_new = torch.ones_like(xs, dtype=torch.bool)
    is_new[..., 1:] = xs[..., 1:] != xs[..., :-1]
    gid = (is_new.cumsum(-1) - 1).long()

    shape = x.shape[:-1] + (n,)
    lo = torch.full(shape, float("inf"), dtype=x.dtype, device=x.device)
    hi = torch.full(shape, float("-inf"), dtype=x.dtype, device=x.device)
    lo.scatter_reduce_(-1, gid, pos, reduce="amin", include_self=True)
    hi.scatter_reduce_(-1, gid, pos, reduce="amax", include_self=True)

    mean_rank_sorted = (lo.gather(-1, gid) + hi.gather(-1, gid)) / 2
    ranks = torch.empty_like(x)
    ranks.scatter_(-1, order, mean_rank_sorted)
    return ranks


def top_p_mask(error: Tensor, p: float) -> Tensor:
    """Boolean mask of the top-`p` fraction by `error`, ties resolved inclusively.

    Inclusive ties match `analysis.metrics.top_p_set`'s default in the frozen
    tree, so positive-set definitions stay identical across the workshop and
    journal code paths.
    """
    n = error.shape[-1]
    k = max(1, int(round(p * n)))
    thresh = error.sort(dim=-1, descending=True).values[..., k - 1:k]
    return error >= thresh


def auroc(signal: Tensor, positive: Tensor) -> Tensor:
    """Mann-Whitney AUROC of `signal` for detecting `positive`.

    AUROC = (mean rank of positives - (n_pos+1)/2) / n_neg, which is the exact
    probability that a randomly drawn positive outranks a randomly drawn
    negative, with ties counted as half.
    """
    r = _rank(signal)
    n_pos = positive.sum(dim=-1).to(signal.dtype)
    n_neg = positive.shape[-1] - n_pos
    rank_sum = (r * positive).sum(dim=-1)
    out = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg).clamp(min=1)
    # AUROC is undefined when either class is empty. Clamping the denominator used
    # to return 0.0 there -- a plausible-looking value that silently propagates.
    return torch.where((n_pos > 0) & (n_neg > 0), out, torch.full_like(out, float("nan")))


def auprc(signal: Tensor, positive: Tensor) -> Tensor:
    """Average precision, with ties resolved by their expected value.

    Scanning `argsort(descending=True)` breaks ties by INDEX ORDER, so a fully
    tied (worthless) signal scored anywhere from 0.033 to 0.209 against a true
    value of 0.05, purely according to where the positives happened to sit in the
    array. Ranking by tie-averaged rank instead makes every tied group contribute
    its expected precision, so a constant signal scores the prevalence as it must.
    """
    key = -_rank(signal)                       # tie-averaged, descending
    order = key.argsort(dim=-1, stable=True)
    pos = positive.gather(-1, order).to(signal.dtype)
    ks = key.gather(-1, order)

    n = pos.shape[-1]
    k = torch.arange(1, n + 1, dtype=signal.dtype, device=signal.device)
    tp = pos.cumsum(dim=-1)
    # within a tie-group, replace tp/k by the group's end-of-run values so every
    # member of the group gets the same (expected) precision
    is_end = torch.ones_like(ks, dtype=torch.bool)
    is_end[..., :-1] = ks[..., 1:] != ks[..., :-1]
    gid = is_end.cumsum(-1) - is_end.to(torch.long)      # group index, 0-based
    gid = gid.long().clamp(min=0)
    ng = int(gid.max().item()) + 1
    shape = pos.shape[:-1] + (n,)
    tp_end = torch.zeros(shape, dtype=signal.dtype, device=signal.device)
    k_end = torch.zeros(shape, dtype=signal.dtype, device=signal.device)
    tp_end.scatter_reduce_(-1, gid, tp, reduce="amax", include_self=False)
    k_end.scatter_reduce_(-1, gid, k.expand_as(tp), reduce="amax", include_self=False)
    precision = tp_end.gather(-1, gid) / k_end.gather(-1, gid).clamp(min=1)
    return (precision * pos).sum(dim=-1) / pos.sum(dim=-1).clamp(min=1)


def recall_at(signal: Tensor, positive: Tensor, p: float) -> Tensor:
    """Fraction of positives captured by the top-`p` fraction of `signal`."""
    sel = top_p_mask(signal, p)
    return (sel & positive).sum(dim=-1).to(signal.dtype) / positive.sum(dim=-1).clamp(min=1)


def enrichment_at(signal: Tensor, positive: Tensor, p: float) -> Tensor:
    """Precision in the top-`p` of `signal`, divided by base prevalence.

    1.0 means the signal is worthless; 1/p is the perfect-detector ceiling.
    """
    sel = top_p_mask(signal, p)
    precision = (sel & positive).sum(dim=-1).to(signal.dtype) / sel.sum(dim=-1).clamp(min=1)
    prevalence = positive.sum(dim=-1).to(signal.dtype) / positive.shape[-1]
    return precision / prevalence.clamp(min=1e-12)


def risk_coverage(signal: Tensor, error: Tensor, reduce: str = "mean") -> dict:
    """Selective risk as a function of coverage, accepting lowest-signal first.

    Returns AURC together with the oracle (accept by true error) and random
    (= global mean error) references, and the normalised

        aurc_excess = (AURC - AURC_oracle) / (AURC_random - AURC_oracle)

    which lands in [0, 1] with 0 = perfect and 1 = worthless. The normalisation is
    what makes systems with different error scales comparable at all -- raw AURC
    on water (192 atoms) and ethanol (9 atoms) are not commensurable numbers.

    `reduce="max"` gives the safety-relevant variant: the worst error admitted at
    each coverage, rather than the average.
    """
    n = signal.shape[-1]

    def _curve(key: Tensor) -> Tensor:
        """Selective-risk curve, accepting lowest-`key` first.

        Ties in `key` leave the accept order genuinely undetermined, and taking
        whatever `argsort` happens to produce made the result depend on input
        index order: a constant signal scored 0.84 or 1.09 on the same data
        depending only on how it was laid out. For the mean reduction the
        expectation over random tie-breaks has a closed form -- within a tied
        group every member contributes the group's mean error -- so we substitute
        that and the curve becomes exactly the expected one, order-independent.
        """
        r = _rank(key)
        order = r.argsort(dim=-1, stable=True)
        e = error.gather(-1, order)
        if reduce == "mean":
            ks = r.gather(-1, order)
            n_ = e.shape[-1]
            is_new = torch.ones_like(ks, dtype=torch.bool)
            is_new[..., 1:] = ks[..., 1:] != ks[..., :-1]
            gid = (is_new.cumsum(-1) - 1).long()
            shape = e.shape[:-1] + (n_,)
            gsum = torch.zeros(shape, dtype=e.dtype, device=e.device)
            gcnt = torch.zeros(shape, dtype=e.dtype, device=e.device)
            gsum.scatter_add_(-1, gid, e)
            gcnt.scatter_add_(-1, gid, torch.ones_like(e))
            e = (gsum / gcnt.clamp(min=1)).gather(-1, gid)
        if reduce == "mean":
            k = torch.arange(1, n + 1, dtype=error.dtype, device=error.device)
            return e.cumsum(dim=-1) / k
        return e.cummax(dim=-1).values

    curve = _curve(signal)
    oracle = _curve(error)
    aurc, aurc_oracle = curve.mean(dim=-1), oracle.mean(dim=-1)
    if reduce == "mean":
        aurc_random = error.mean(dim=-1)
    else:
        # A random accept-order's expected running max, averaged over coverage --
        # NOT the global max, which is the anti-oracle and makes aurc_excess look
        # better than it is. E[max of a random prefix of length k] is obtained by
        # averaging the running max of a shuffled copy.
        perm = torch.argsort(torch.rand_like(error), dim=-1)
        aurc_random = error.gather(-1, perm).cummax(dim=-1).values.mean(dim=-1)
    denom = aurc_random - aurc_oracle
    return {
        "aurc": aurc,
        "aurc_oracle": aurc_oracle,
        "aurc_random": aurc_random,
        # NaN, not a 1e27 artefact, when the normalisation is genuinely undefined
        # (every error identical, so oracle == random and there is nothing to scale by)
        "aurc_excess": torch.where(denom.abs() > 1e-30, (aurc - aurc_oracle) / denom,
                                   torch.full_like(aurc, float("nan"))),
        "curve": curve,
    }
