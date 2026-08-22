"""Closed-loop gradient probe.

All that survives of the read-influence line, which was refuted: re-weighting
memory dimensions cannot help when the loss switches the memory channel off.

``grad_influence`` is kept because it produced a result the paper relies on --
the gradient from a query DOES reach the planting site through the recurrence,
essentially flat in distance (ratio 0.217 at D=32 against a rho^D prediction of
3.9e-5). That rules out vanishing gradient as the reason the memory chain never
learns to carry a binding.

It must be run as a truncated recurrent rollout. In the parallel graph the
memory chain does not exist, and the gradient there merely recovers the stacked
local reach L*(W-1) -- exactly zero beyond it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .losses import IGNORE_INDEX
from .model.maglev import MaglevModel

__all__ = ["grad_influence", "default_anchors", "scored_positions", "normalise"]


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def normalise(w: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalise per position so the last dim sums to 1 (SPEC.md section 4).

    ``clamp_min`` rather than ``sum + eps``: gradient magnitudes here are often
    ~1e-7, comparable to eps itself, and the additive form then quietly shrinks
    the weights by several percent in a way that depends on gradient scale --
    an error that would ride along in every RWC weight without ever failing.
    Clamping is exact whenever the sum is non-zero, and leaves a genuinely
    all-zero position (memory nothing reads) at zero.
    """
    return w / w.sum(dim=-1, keepdim=True).clamp_min(eps)


def _to_memory_index(w_mem_input: Tensor) -> Tensor:
    """Re-index from mem-input position to memory position.

    ``mem_in[t] = m'_{t-1}``, so the influence of ``m'_j`` sits at ``t = j+1``.
    ``m'_{T-1}`` is never read by anyone, so it gets exactly zero.
    """
    out = torch.zeros_like(w_mem_input)
    out[..., :-1, :] = w_mem_input[..., 1:, :]
    return out




# --------------------------------------------------------------------------
# Estimator A -- gradient (reference, exact, slow)
# --------------------------------------------------------------------------


def grad_influence(
    model: MaglevModel,
    tokens: Tensor,
    targets: Tensor,
    *,
    horizon_H: int,
    anchors: Optional[Sequence[int]] = None,
    detach_local: bool = True,
    eps: float = 1e-8,
    normalise_out: bool = True,
) -> Tensor:
    """``w_j[i] = E_batch | d L_CE(j+1 .. j+H) / d m_j[i] |``.

    Returns ``(n_anchors, d)``, one row per anchor position ``j``.

    ``detach_local`` cuts the gates' dependence on the residual stream. Gradient
    w.r.t. ``m_j`` is already confined to the recurrent channel by construction
    -- memory reaches the network only through ``W_k_rec``/``W_v_rec`` -- but the
    gates ``G_loc``/``G_rec`` read the residual ``a_t``, which at layers above
    the first does depend on memory. Detaching them is how "gradient must flow
    only through the recurrent channel" is honoured.
    """
    device = tokens.device
    b, t = tokens.shape
    d = model.cfg.d_model
    if anchors is None:
        anchors = default_anchors(targets, horizon_H)
    anchors = _keep_scored_anchors(list(anchors), targets, horizon_H)

    was_training = model.training
    model.eval()
    rows: List[Tensor] = []
    try:
        for j in anchors:
            end = min(t, j + 1 + horizon_H)
            if end <= j + 1:
                rows.append(torch.zeros(d, device=device))
                continue

            # Warm the cache and the memory chain up to position j, with no
            # graph: those steps are the *context*, not what we differentiate.
            with torch.no_grad():
                caches = model.make_caches(
                    b, device=device, dtype=model.embed.weight.dtype
                )
                m_prev = torch.zeros(
                    b, 1, d, device=device, dtype=model.embed.weight.dtype
                )
                for step in range(j + 1):
                    _, m_step, _ = model.forward_decoder(
                        tokens[:, step : step + 1],
                        m_prev,
                        positions=torch.arange(step, step + 1, device=device),
                        caches=caches,
                    )
                    m_prev = m_step

            # m_j is the leaf we differentiate against.
            m_leaf = m_prev.detach().clone().requires_grad_(True)
            mem = m_leaf
            logits_steps: List[Tensor] = []
            for step in range(j + 1, end):
                logits_t, m_t, _ = model.forward_decoder(
                    tokens[:, step : step + 1],
                    mem,
                    positions=torch.arange(step, step + 1, device=device),
                    caches=caches,
                    detach_gate_input=detach_local,
                )
                logits_steps.append(logits_t)
                mem = m_t  # the chain, retained -- this is what needs recurrence

            logits = torch.cat(logits_steps, dim=1)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets[:, j + 1 : end].reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            (grad,) = torch.autograd.grad(loss, m_leaf)
            rows.append(grad.abs().squeeze(1).mean(dim=0))
    finally:
        model.train(was_training)

    w = torch.stack(rows)
    return normalise(w, eps) if normalise_out else w


def scored_positions(targets: Tensor) -> Tensor:
    """Positions that contribute to CE, i.e. whose target is not ignored."""
    return (targets != IGNORE_INDEX).any(dim=0).nonzero().flatten()


def default_anchors(targets: Tensor, horizon_H: int, n: int = 8) -> List[int]:
    """Anchors ``j`` whose horizon ``j+1 .. j+H`` actually contains a target.

    Choosing anchors by position alone looks reasonable and is wrong: under an
    answer-only loss almost every window is entirely ignored, ``L_CE`` over it
    is undefined, and the estimator would quietly return zeros -- which reads
    exactly like "this memory is never used". Anchors are picked from the
    scoring positions instead.
    """
    scored = scored_positions(targets)
    if scored.numel() == 0:
        raise ValueError("no scored positions: every target is ignored")
    lo = max(0, int(scored.min()) - horizon_H)
    hi = max(lo, int(scored.max()) - 1)
    if hi <= lo:
        return [lo]
    stride = max(1, (hi - lo) // n)
    return list(range(lo, hi + 1, stride))[:n]


def _keep_scored_anchors(
    anchors: Sequence[int], targets: Tensor, horizon_H: int
) -> List[int]:
    """Drop anchors whose horizon contains no target, and fail if none remain.

    An anchor with nothing to score makes ``L_CE`` undefined; averaging that in
    would return NaN, and silently zeroing it would read as "this memory is
    never used". Neither is acceptable in a number that gates the paper.
    """
    scored = set(scored_positions(targets).tolist())
    kept = [j for j in anchors if any(j < s <= j + horizon_H for s in scored)]
    if not kept:
        raise ValueError(
            f"none of the {len(anchors)} anchors has a scored position within "
            f"horizon_H={horizon_H}; the influence estimate would be identically "
            f"zero. Scored positions: {sorted(scored)[:8]}..."
        )
    return kept


# --------------------------------------------------------------------------
# Estimator B -- saliency (cheap, used in training)
# --------------------------------------------------------------------------






