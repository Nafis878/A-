"""CE, uniform consistency, RWC, and the masked variants (SPEC.md section 5).

    uniform (Maglev, Eq. 5)  L = CE + lam * ||m_t - m'_t||_2 / sqrt(d)
    RWC     (ours)           L = CE + lam * ||w_t * (m_t - m'_t)||_2 / sqrt(d)
    masked  (C2 ablations)   L = CE + lam * ||mask_t * (m_t - m'_t)||_2 / sqrt(d)

**Weight scaling is the one thing here that decides whether the lambda sweep
means anything.** The estimators normalise ``w`` to sum to 1 over ``d`` (SPEC
section 4), so a raw ``w`` has entries around ``1/d`` and the RWC penalty would
be ~``d`` times weaker than the uniform one at the same lambda -- the C3
comparison would then be measuring penalty strength, not allocation. So ``w`` is
rescaled to *mean* 1 before it multiplies the difference. That makes uniform L2
exactly the special case of RWC with flat weights, and lambda directly
comparable between the two arms, which is the whole point of the sweep.

Masks are left as raw 0/1, so ``mask_topk`` at k=100 reduces exactly to uniform.
Within one k the three mask kinds select the same number of dimensions and are
therefore scale-matched, which is what C2 compares. Across different k the
penalty scale does vary as ``sqrt(k)``; that is inherent to "mask the loss to k%
of dimensions" and is noted rather than corrected.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "IGNORE_INDEX",
    "cross_entropy",
    "consistency_loss",
    "influence_mask",
    "propagation_penalty",
    "MASK_KINDS",
]

IGNORE_INDEX = -100
MASK_KINDS = ("mask_topk", "mask_bottomk", "mask_randomk")


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean CE over positions whose target is not ``IGNORE_INDEX``.

    Averaged over *scoring positions* rather than over all positions, so the
    number reported is comparable between full-sequence and answer-only runs.
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def influence_mask(
    weights: Tensor,
    kind: str,
    k_pct: float,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """A 0/1 mask over ``d`` selecting k% of dimensions by read-influence.

    ``mask_randomk`` draws a *fixed* subset rather than resampling every step.
    The control for "the top k% of dimensions" is "some k% of dimensions", not
    "a different k% each step" -- resampling would be a dropout-like
    intervention and would test a different hypothesis than C2 asks about.
    """
    if kind not in MASK_KINDS:
        raise ValueError(f"unknown mask kind {kind!r}; expected one of {MASK_KINDS}")
    if not 0.0 < k_pct <= 100.0:
        raise ValueError(f"k_pct must lie in (0, 100], got {k_pct}")

    d = weights.numel()
    k = max(1, int(round(d * k_pct / 100.0)))
    mask = torch.zeros(d, device=weights.device, dtype=weights.dtype)

    if kind == "mask_topk":
        idx = torch.topk(weights, k).indices
    elif kind == "mask_bottomk":
        idx = torch.topk(weights, k, largest=False).indices
    else:
        idx = torch.randperm(d, generator=generator, device=weights.device)[:k]
    mask[idx] = 1.0
    return mask


def consistency_loss(
    m: Tensor,
    m_prime: Tensor,
    *,
    kind: str,
    weights: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    detach_target: bool = False,
) -> Tensor:
    """The consistency penalty, *before* multiplication by lambda.

    ``m``, ``m_prime`` are (B, T, d). Returns a scalar: the per-position L2 norm
    of the (weighted) difference, divided by ``sqrt(d)``, averaged over batch
    and positions.

    ``weights`` (for ``rwc``) and ``mask`` (for the masked variants) are (d,)
    and must already be detached -- they are loss *weights*, not parameters, and
    gradient must not flow into the influence estimate.
    """
    if m.shape != m_prime.shape:
        raise ValueError(f"shape mismatch: m {tuple(m.shape)} vs {tuple(m_prime.shape)}")
    if kind == "none":
        return m.new_zeros(())

    # With detach_target the prefiller is a fixed teacher for this term and the
    # decoder chases it; without it, the two are pulled toward each other and a
    # large lambda can be satisfied by degrading the teacher instead.
    diff = m - (m_prime.detach() if detach_target else m_prime)
    d = diff.shape[-1]

    if kind == "uniform":
        scaled = diff
    elif kind == "rwc":
        if weights is None:
            raise ValueError("consistency kind 'rwc' requires weights")
        if weights.requires_grad:
            raise ValueError("influence weights must be detached before use as a loss weight")
        # Rescale sum-to-1 weights to mean 1, so flat weights reproduce uniform
        # L2 exactly and lambda is comparable between the two arms.
        scaled = diff * (weights * d).to(diff.dtype)
    elif kind in MASK_KINDS:
        if mask is None:
            raise ValueError(f"consistency kind {kind!r} requires a mask")
        if mask.requires_grad:
            raise ValueError("mask must be detached before use as a loss weight")
        scaled = diff * mask.to(diff.dtype)
    else:
        raise ValueError(f"unknown consistency kind {kind!r}")

    # Norm per position, then mean -- not a norm over the flattened tensor,
    # which would let one position's error dominate the whole batch.
    per_position = scaled.float().pow(2).sum(dim=-1).sqrt()
    return per_position.mean() / (d**0.5)


def propagation_penalty(
    step_fn,
    mem_in: Tensor,
    positions: Tensor,
    *,
    target_gain: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Keep the memory recurrence from being switched off.

    Measured failure this exists to prevent: the consistency loss drives
    ``rho(dm_t/dm_{t-1})`` from 0.728 at lambda=0 to ~0.01 at lambda=0.03 --
    below its value at random initialisation (~0.10). Memory is still *read*
    (zeroing it moves logits by 6-13) but nothing is *written forward*, so at
    deployment ``m_t`` depends only on local context and nothing survives past
    the window.

    The penalty is on the log-gain, so overshoot and undershoot cost the same
    in relative terms and the term cannot be gamed by inflating the memory
    scale. It targets a gain rather than merely a floor because an expansive
    recurrence (gain > 1) compounds error instead: from
    ``delta_t = e_t + J_t delta_{t-1}``, a gain below 1 bounds the deployed
    deviation by ``||e|| / (1 - gain)`` while a gain above 1 leaves it unbounded.

    ``step_fn`` must run attention under the ``math`` backend: the fused SDPA
    kernel has no forward-mode AD rule, so a JVP through it raises.

    Cost is one forward-mode JVP. The tangent is supported on a single position
    per sequence and the output is read at that same position, which makes this
    the *exact* diagonal block ``dm_t/dm_{t-1}`` rather than a windowed mixture:
    perturbing only ``mem_in[t]`` and reading only ``m[t]`` admits no other path.
    """
    tangent = torch.zeros_like(mem_in)
    rows = torch.arange(mem_in.shape[0], device=mem_in.device)
    v = torch.randn(
        mem_in.shape[0], mem_in.shape[-1],
        generator=generator, device=mem_in.device, dtype=mem_in.dtype,
    )
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    tangent[rows, positions] = v

    _, jv = torch.func.jvp(step_fn, (mem_in,), (tangent,))
    gain = jv[rows, positions].norm(dim=-1)  # ||J v|| with ||v|| = 1
    log_gain = gain.clamp_min(1e-8).log()
    return (log_gain - math.log(target_gain)).pow(2).mean()
