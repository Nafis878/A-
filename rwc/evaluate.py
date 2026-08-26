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

__all__ = ["probe_accuracy", "ProbeResult", "bits_per_byte", "lm_memory_benefit"]

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


# --------------------------------------------------------------------------
# Language modelling: what does the memory channel actually buy?
# --------------------------------------------------------------------------
#
# Raw BPB is the wrong headline for this question. Character prediction is
# dominated by local statistics -- the last few characters carry most of the
# predictable signal -- so two models whose memory channels differ enormously
# can post nearly the same aggregate BPB. Reporting it alone would hide the
# effect rather than measure it.
#
# The quantity that isolates the channel is the DIFFERENCE between the deployed
# model and the same weights with memory forced to zero. Under
# ``mem_in_override`` the decoder still has its sliding window, so the ablated
# model is exactly a window-W, L-layer local model, reach L*(W-1). Whatever
# separates them is what memory contributed and nothing else.
#
# Resolving that by position gives the LM analogue of the synthetic distance
# axis. A model whose memory is dead can improve only until its context fills
# the local reach and must then flatten; a live channel keeps improving.


@torch.no_grad()
def lm_memory_benefit(
    model: nn.Module,
    loader,
    *,
    n_batches: int = 16,
    batch_size: int = 8,
    buckets: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Held-out BPB with memory live vs zeroed, overall and by position.

    ``loader`` is a ``CharShardLoader`` on the held-out split. Returns bits per
    byte (one text8 character is one byte, so this is also bits per character),
    the ablated baseline, their difference, and the same split by position.
    """
    buckets = buckets or [0, 8, 16, 32, 64, 128, 256]
    ce_live: List[Tensor] = []
    ce_dead: List[Tensor] = []

    was_training = model.training
    model.eval()
    try:
        for i in range(n_batches):
            batch = loader.batch(i, batch_size)
            tokens = batch.tokens.to(device) if device is not None else batch.tokens
            targets = batch.targets.to(tokens.device)

            live = model.forward_recurrent(tokens).logits
            zeros = torch.zeros(
                tokens.shape[0], tokens.shape[1], model.cfg.d_model,
                device=tokens.device, dtype=live.dtype,
            )
            dead = model.forward_recurrent(tokens, mem_in_override=zeros).logits

            for out, sink in ((live, ce_live), (dead, ce_dead)):
                nll = torch.nn.functional.cross_entropy(
                    out.reshape(-1, out.shape[-1]).float(),
                    targets.reshape(-1),
                    reduction="none",
                ).view(targets.shape)
                sink.append(nll.mean(dim=0))  # mean over batch, keep position
    finally:
        model.train(was_training)

    live_pos = torch.stack(ce_live).mean(dim=0)
    dead_pos = torch.stack(ce_dead).mean(dim=0)
    n = live_pos.numel()

    def bpb(ce_nats: float) -> float:
        # One text8 character is one byte, so tokens == bytes and the ratio is
        # 1. Go through the shared helper anyway, so this and the synthetic path
        # cannot drift apart if the definition ever changes.
        return bits_per_byte(ce_nats, n_tokens=1, n_bytes=1)

    by_position = []
    for lo, hi in zip(buckets, buckets[1:]):
        if lo >= n:
            break
        hi = min(hi, n)
        by_position.append({
            "lo": lo, "hi": hi,
            "live": bpb(float(live_pos[lo:hi].mean())),
            "dead": bpb(float(dead_pos[lo:hi].mean())),
            "benefit": bpb(float((dead_pos[lo:hi] - live_pos[lo:hi]).mean())),
        })

    return {
        "bpb_live": bpb(float(live_pos.mean())),
        "bpb_dead": bpb(float(dead_pos.mean())),
        "benefit": bpb(float((dead_pos - live_pos).mean())),
        "by_position": by_position,
        "n_positions": n,
        "n_sequences": n_batches * batch_size,
    }
