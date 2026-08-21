"""CE, uniform consistency, RWC, and the masked variants (SPEC.md section 5).

M3 implements the cross-entropy term, which the trainer needs. The consistency
family lands in M5 alongside the C1 diagnostic.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["IGNORE_INDEX", "cross_entropy"]

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


def consistency_loss(*args, **kwargs):
    """Uniform / RWC / masked consistency. Milestone M5."""
    raise NotImplementedError("milestone M5: see SPEC.md section 9")
