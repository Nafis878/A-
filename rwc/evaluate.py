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

__all__ = ["probe_accuracy", "ProbeResult", "bits_per_byte",
           "lm_memory_benefit", "lm_induction_bpb", "induction_positions"]

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


# --------------------------------------------------------------------------
# Long-range exact retrieval inside natural text
# --------------------------------------------------------------------------
#
# Aggregate BPB cannot test the claim, and that is measurable rather than
# arguable. On held-out text8 only 3.87% of positions are ones where the next
# character is fixed by a substring that last occurred beyond the decoder's
# reach. The other 96% are predictable from the last few characters, so a defect
# confined to long-range exact copies is averaged into invisibility.
#
# But "the context repeated" is not yet the right filter either. Most repeats are
# common phrases -- " the was", " of the " -- whose continuation local statistics
# already supply, so they test nothing. Measured: an UNTRAINED model scores
# BETTER on raw induction positions than on ordinary ones. Filtering to
# substrings that never occur in a large sample of the training corpus leaves
# 12.6% of them, 0.49% of all positions, median distance 72 characters. Those are
# the ones where the continuation can only come from this context.
#
# So three groups are reported, never one:
#
#   induction_novel   repeat, and the substring is absent from training
#                     -> genuine long-range exact retrieval
#   induction_common  repeat, but the substring is common in training
#                     -> local statistics suffice; a control
#   other             everything else
#
# An arm that wins on `induction_novel` while losing on `other` has made a
# trade-off, and reporting only the flattering half would be the same error as
# quoting the confounded `benefit` metric above.


def build_reference(text: str, min_match: int = 8, n_chars: int = 3_000_000) -> set:
    """Substrings of length ``min_match`` occurring in a training sample."""
    sample = text[:n_chars]
    return {sample[i:i + min_match] for i in range(len(sample) - min_match)}


def induction_positions(
    seq_bytes: bytes,
    *,
    min_match: int,
    reach: int,
    alphabet: Optional[str] = None,
    reference: Optional[set] = None,
):
    """[(position, distance, is_novel)] where the context repeats from beyond reach.

    ``is_novel`` is True when the repeated substring is absent from ``reference``
    -- i.e. the continuation cannot be supplied by training statistics and must
    come from this context. Without a reference every hit is reported as novel.
    """
    out = []
    for t in range(min_match, len(seq_bytes)):
        pat = seq_bytes[t - min_match:t]
        prev = seq_bytes.rfind(pat, 0, t - min_match)
        if prev < 0:
            continue
        distance = t - min_match - prev
        if distance <= reach:
            continue
        novel = True
        if reference is not None and alphabet is not None:
            novel = "".join(alphabet[c] for c in pat) not in reference
        out.append((t, distance, novel))
    return out


@torch.no_grad()
def lm_induction_bpb(
    model: nn.Module,
    loader,
    *,
    reference: Optional[set] = None,
    n_batches: int = 200,
    batch_size: int = 8,
    min_match: int = 8,
    buckets: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Held-out BPB split into novel-induction, common-induction and other."""
    buckets = buckets or [7, 16, 32, 64, 128, 256]
    reach = model.cfg.n_layers * (model.cfg.window - 1)
    alphabet = getattr(loader, "alphabet", None)

    novel: List[float] = []
    novel_d: List[int] = []
    common: List[float] = []
    other: List[float] = []

    was_training = model.training
    model.eval()
    try:
        for i in range(n_batches):
            batch = loader.batch(i, batch_size)
            tokens = batch.tokens.to(device) if device is not None else batch.tokens
            targets = batch.targets.to(tokens.device)
            logits = model.forward_recurrent(tokens).logits
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1), reduction="none",
            ).view(targets.shape)

            for b in range(tokens.shape[0]):
                row = tokens[b].to(torch.uint8).cpu().numpy().tobytes()
                hits = induction_positions(
                    row, min_match=min_match, reach=reach,
                    alphabet=alphabet, reference=reference,
                )
                marked = set()
                for t, d, is_novel in hits:
                    marked.add(t)
                    if is_novel:
                        novel.append(float(nll[b, t]))
                        novel_d.append(d)
                    else:
                        common.append(float(nll[b, t]))
                other += [float(nll[b, t]) for t in range(tokens.shape[1])
                          if t not in marked]
    finally:
        model.train(was_training)

    def bpb(vals) -> float:
        if not vals:
            return float("nan")
        return bits_per_byte(sum(vals) / len(vals), n_tokens=1, n_bytes=1)

    by_distance = []
    for lo, hi in zip(buckets, buckets[1:]):
        vals = [v for v, d in zip(novel, novel_d) if lo <= d < hi]
        by_distance.append({"lo": lo, "hi": hi, "n": len(vals), "bpb": bpb(vals)})

    total = len(novel) + len(common) + len(other)
    return {
        "induction_novel": bpb(novel),
        "induction_common": bpb(common),
        "other": bpb(other),
        "all": bpb(novel + common + other),
        "n_novel": len(novel), "n_common": len(common), "n_other": len(other),
        "novel_fraction": len(novel) / total if total else 0.0,
        "by_distance": by_distance,
        "min_match": min_match, "reach": reach,
        "n_sequences": n_batches * batch_size,
    }
