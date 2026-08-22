"""Effective rank, the C1 decorrelation curve, and figure data.

**C1** is the diagnostic the paper turns on: the rank correlation between
per-dimension *consistency error* and per-dimension *read-influence* should
decay as the retrieval horizon ``H`` grows. If it does not decay, the claim
that uniform L2 is misaligned with what memory is read for loses its
mechanism, and that is a result worth reporting rather than hiding
(SPEC.md section 9, M5).

Two subtleties, both of which silently flatten the curve if ignored:

* ``grad_influence`` drops anchors whose horizon contains no scored position, so
  a short ``H`` and a long ``H`` would otherwise be evaluated at *different*
  sets of positions. ``c1_curve`` picks one anchor set valid at the smallest
  horizon in the sweep and reuses it throughout.
* C1 needs **full-sequence targets**, even when the model was trained with
  answer-only CE. Under an answer-only loss there is exactly one scored
  position, so every horizon that reaches it scores the identical set and
  ``grad_influence`` returns bit-identical values for H=8 and H=128 -- the
  curve comes out perfectly flat for a reason that has nothing to do with the
  hypothesis. ``c1_curve`` therefore builds full-sequence targets by default.
  This is legitimate: "which memory dimensions affect predictions at
  ``j+1..j+H``" is well posed regardless of which positions the model was
  *trained* on.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor

from .influence import (
    default_anchors,
    grad_influence,
    saliency_influence,
    scored_positions,
    spearman,
)
from .model.maglev import MaglevModel

__all__ = [
    "consistency_error_by_dim",
    "effective_rank",
    "c1_curve",
    "memory_spectrum",
    "full_sequence_targets",
]

IGNORE_INDEX = -100


def full_sequence_targets(tokens: Tensor) -> Tensor:
    """Next-token targets at every position -- what the C1 horizon axis needs."""
    targets = torch.full_like(tokens, IGNORE_INDEX)
    targets[:, :-1] = tokens[:, 1:]
    return targets


def consistency_error_by_dim(m: Tensor, m_prime: Tensor) -> Tensor:
    """Per-dimension consistency error ``E_b |m[b,t,i] - m'[b,t,i]|``.

    Returns (T, d) -- kept per position, because C1 correlates error against
    influence *at the same position*, and averaging over t first would blur
    exactly the structure the diagnostic is looking for.
    """
    return (m - m_prime).abs().mean(dim=0)


def effective_rank(memory: Tensor, *, center: bool = True) -> float:
    """Entropy-based effective rank of a (N, d) memory matrix.

    ``exp(H(p))`` for ``p`` the normalised singular value spectrum. Maglev
    explains its lambda anomaly as *capacity*; effective rank measures directly
    how much of the memory space is in use, which is what separates a capacity
    story from an allocation one.

    Centred by default. Uncentred, a shared mean offset across positions shows
    up as one dominant singular direction and drags the effective rank toward 1,
    which describes the bias vector rather than the capacity actually in use.
    """
    flat = memory.float().reshape(-1, memory.shape[-1])
    if center:
        flat = flat - flat.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(flat)
    p = s / s.sum().clamp_min(1e-12)
    entropy = -(p * p.clamp_min(1e-12).log()).sum()
    return float(entropy.exp())


def memory_spectrum(memory: Tensor) -> Dict[str, float]:
    """Effective rank plus the participation ratio, for the capacity argument."""
    flat = memory.float().reshape(-1, memory.shape[-1])
    centred = flat - flat.mean(dim=0, keepdim=True)
    s2 = torch.linalg.svdvals(centred).pow(2)
    return {
        "effective_rank": effective_rank(flat, center=True),
        "effective_rank_uncentred": effective_rank(flat, center=False),
        "participation_ratio": float(s2.sum().pow(2) / s2.pow(2).sum().clamp_min(1e-12)),
        "n_dims": flat.shape[-1],
    }


@torch.no_grad()
def _memory_pair(model: MaglevModel, tokens: Tensor):
    out = model.forward_train(tokens)
    return out.m, out.m_prime


def c1_curve(
    model: MaglevModel,
    tokens: Tensor,
    targets: Optional[Tensor] = None,
    *,
    horizons: Sequence[int],
    anchors: Optional[Sequence[int]] = None,
    n_anchors: int = 6,
    detach_local: bool = True,
    include_saliency: bool = False,
) -> Dict[str, object]:
    """Rank correlation of consistency error against read-influence, vs ``H``.

    Returns a dict with one entry per horizon, each carrying the mean
    per-position Spearman rho (the primary number, since it is what the RWC loss
    actually weights) and the pooled rho over dimension-aggregated vectors.
    """
    horizons = sorted(horizons)
    if targets is None:
        targets = full_sequence_targets(tokens)
    n_scored = len(scored_positions(targets))
    if n_scored < 2 * len(horizons):
        raise ValueError(
            f"only {n_scored} scored positions: widening H would not add anything "
            "to score, so the curve would be flat for reasons unrelated to C1. "
            "Pass full-sequence targets (analysis.full_sequence_targets)."
        )
    if anchors is None:
        # One anchor set, valid at the *smallest* horizon, reused for all of
        # them -- otherwise different H are scored at different positions.
        anchors = default_anchors(targets, horizons[0], n=n_anchors)
    anchors = list(anchors)

    scored = set(scored_positions(targets).tolist())
    unusable = [j for j in anchors if not any(j < s <= j + horizons[0] for s in scored)]
    if unusable:
        raise ValueError(
            f"anchors {unusable} have no scored position within the smallest "
            f"horizon {horizons[0]}; the curve would not be comparable across H"
        )

    m, m_prime = _memory_pair(model, tokens)
    error = consistency_error_by_dim(m, m_prime)  # (T, d)

    per_horizon: List[Dict[str, float]] = []
    for h in horizons:
        w = grad_influence(
            model, tokens, targets, horizon_H=h, anchors=anchors,
            detach_local=detach_local,
        )  # (n_anchors, d)
        rhos = [spearman(error[j], w[i]) for i, j in enumerate(anchors)]
        finite = [r for r in rhos if r == r]  # drop NaN from constant inputs
        per_horizon.append({
            "horizon": h,
            "rho_mean": sum(finite) / len(finite) if finite else float("nan"),
            "rho_min": min(finite) if finite else float("nan"),
            "rho_max": max(finite) if finite else float("nan"),
            "rho_pooled": spearman(error[anchors].mean(dim=0), w.mean(dim=0)),
            "n_anchors": len(finite),
        })

    out: Dict[str, object] = {
        "anchors": anchors,
        "horizons": list(horizons),
        "per_horizon": per_horizon,
        "slope": _slope([p["horizon"] for p in per_horizon],
                        [p["rho_mean"] for p in per_horizon]),
        "memory_spectrum": memory_spectrum(m),
    }
    if include_saliency:
        b = saliency_influence(model, tokens).mean(dim=0)  # (T, d)
        out["rho_saliency"] = {
            "rho_mean": sum(spearman(error[j], b[j]) for j in anchors) / len(anchors)
        }
    return out


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Least-squares slope of rho against horizon. Negative = C1 holds."""
    pairs = [(x, y) for x, y in zip(xs, ys) if y == y]
    if len(pairs) < 2:
        return float("nan")
    n = len(pairs)
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    denom = sum((x - mx) ** 2 for x, _ in pairs)
    if denom == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in pairs) / denom
