"""Evaluation: probe accuracy per distance bucket, and BPB for the LM runs.

Probe accuracy is measured on the model as *deployed* (Eq. 10) -- prefiller
discarded, decoder recurrent on its own memory. Measuring the parallel path
instead saturates near 1.0 at every retrieval distance, because the prefiller's
full attention hands the decoder the answer, and the distance axis then carries
no signal at all. The C1 correlation lives in analysis.py.

Probe accuracy is *always* broken out per distance bucket and never averaged
into a single number (SPEC.md section 6): the slope against distance is the
finding, and a mean would destroy it.
"""

from __future__ import annotations

import math
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


def bits_per_byte(mean_ce_nats: float, n_tokens: int, n_bytes: int) -> float:
    """Bits per byte, so numbers are comparable to Maglev's table.

    ``bpb = (CE_nats / ln 2) * (tokens / bytes)``. Reported instead of raw loss
    because loss depends on the tokenizer -- a 16k vocab and a 50k vocab give
    different CE for identical modelling quality, and this project uses a 16k
    one (SPEC.md section 6). The byte count comes from the shard metadata
    written by scripts/prepare_lm_data.py.
    """
    if n_bytes <= 0 or n_tokens <= 0:
        raise ValueError(f"need positive token and byte counts, got {n_tokens}, {n_bytes}")
    return (mean_ce_nats / math.log(2.0)) * (n_tokens / n_bytes)
