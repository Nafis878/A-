"""CE, the memory-consistency term, and the propagation regulariser.

    consistency (Maglev, Eq. 5)  L = CE + lam * ||m_t - m'_t||_2 / sqrt(d)

Read-weighted and masked variants were removed: measured, refuted, and
explained. Re-weighting WHICH memory dimensions to match cannot help when the
loss has switched the memory channel off, which is what it does past a lambda
threshold (prefiller spread falls ~1800x below random init).
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
    "propagation_penalty",
]

IGNORE_INDEX = -100


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




def consistency_loss(
    m: Tensor,
    m_prime: Tensor,
    *,
    kind: str,
    detach_target: bool = False,
) -> Tensor:
    """The consistency penalty, *before* multiplication by lambda.

    ``m``, ``m_prime`` are (B, T, d). Returns a scalar: the per-position L2 norm
    of the (weighted) difference, divided by ``sqrt(d)``, averaged over batch
    and positions.

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

    if kind != "uniform":
        raise ValueError(f"unknown consistency kind {kind!r}")
    scaled = diff

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
