"""Evaluation: probe accuracy per distance bucket, and BPB for the LM runs.

M3 implements probe accuracy, which the trainer reports and the run manifest
records. BPB and the C1 correlation land in M5.

Probe accuracy is *always* broken out per distance bucket and never averaged
into a single number (SPEC.md section 6): the slope against distance is the
finding, and a mean would destroy it.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

import torch
from torch import Tensor, nn

from .data.synthetic import ProbeBatch, SyntheticGenerator

__all__ = ["probe_accuracy", "ProbeResult", "bits_per_byte"]

ProbeResult = Dict[str, Dict[str, float]]


@torch.no_grad()
def probe_accuracy(
    forward: Callable[[Tensor], Tensor],
    batches: Iterable[ProbeBatch],
    *,
    device: Optional[torch.device] = None,
) -> ProbeResult:
    """Accuracy at the answer position, bucketed by retrieval distance.

    ``forward`` maps tokens -> logits; passing it in rather than a model keeps
    this usable for the Maglev decoder, the baselines, and the recurrent
    inference path without branching here.

    Returns ``{"<distance>": {"acc": .., "n": .., "nll": ..}}``.
    """
    hits: Dict[int, int] = {}
    total: Dict[int, int] = {}
    nll_sum: Dict[int, float] = {}

    for batch in batches:
        tokens = batch.tokens.to(device) if device is not None else batch.tokens
        logits = forward(tokens)
        rows = torch.arange(tokens.shape[0], device=logits.device)
        idx = batch.answer_idx.to(logits.device)
        answer_logits = logits[rows, idx].float()
        gold = batch.targets[rows.cpu(), batch.answer_idx].to(logits.device)

        correct = answer_logits.argmax(dim=-1) == gold
        nll = torch.nn.functional.cross_entropy(answer_logits, gold, reduction="none")

        for i, d in enumerate(batch.distance.tolist()):
            total[d] = total.get(d, 0) + 1
            hits[d] = hits.get(d, 0) + int(correct[i])
            nll_sum[d] = nll_sum.get(d, 0.0) + float(nll[i])

    return {
        str(d): {
            "acc": hits[d] / total[d],
            "nll": nll_sum[d] / total[d],
            "n": total[d],
        }
        for d in sorted(total)
    }


def chance_accuracy(gen: SyntheticGenerator) -> float:
    """Accuracy from guessing uniformly among value tokens -- the floor to beat."""
    return 1.0 / gen.vocab.n_values


def bits_per_byte(*args, **kwargs):
    """BPB for the LM runs, comparable to Maglev's table. Milestone M5."""
    raise NotImplementedError("milestone M5: see SPEC.md section 9")
