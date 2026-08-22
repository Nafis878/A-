"""Recurrence diagnostics.

Deployed probe accuracy is a poor readout at this scale: it needs the whole task
to work, costs a sequential rollout, and its seed spread swamps every effect.
These quantities are deterministic given a checkpoint, cost one forward pass
(or d backward passes for the exact Jacobian), and localise WHERE the memory
pathway fails:

    recurrence_gain        rho(dm_t/dm_{t-1})  -- is memory WRITTEN forward?
    memory_ablation_effect outputs with memory zeroed -- is it READ at all?
    gate_statistics        Eq. 7's g_rec vs g_loc -- is the channel open?
    memory_information     spread of m' across positions -- does it carry anything?
    prefiller_locality     does m'_t depend on tokens beyond the window?

The failure they were built to catch: memory can be read while rho ~ 0, i.e.
consumed at every position but never carried across steps. Accuracy cannot tell
that apart from "memory is unused"; these can.

The C1 decorrelation curve and the C3 slope statistics lived here and are gone:
both served the read-influence hypothesis, which our own measurements refuted
(C1's slope came out positive on all three uniform runs, and C3 showed no
advantage in six lambda x sharing cells).
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .model.maglev import MaglevModel

__all__ = [
    "wilson_interval",
    "recurrence_gain",
    "memory_ablation_effect",
    "gate_statistics",
    "RecurrenceDiagnostics",
    "diagnose_recurrence",
    "prefiller_locality",
    "memory_information",
]

















# --------------------------------------------------------------------------
# Statistics for C3
# --------------------------------------------------------------------------
#
# C3 is a claim about a *slope*: RWC's advantage over uniform L2 should be near
# zero at short range and grow with retrieval distance. A flat advantage would
# mean RWC is a generic regulariser, which the SPEC says would weaken the paper.
# So the quantity to test is the slope of (RWC - uniform) against distance, and
# the unit of independence is the SEED, not the distance bucket -- the six
# buckets within one run share a model and are emphatically not independent
# samples of anything.


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used rather than the normal approximation because probe accuracies here sit
    near 0.06 with n of a few hundred, where the normal interval runs below zero
    and badly understates the uncertainty.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))








# --------------------------------------------------------------------------
# Recurrence diagnostics
# --------------------------------------------------------------------------
#
# Deployed probe accuracy is a noisy readout: it needs the whole task to work,
# it costs a sequential rollout, and at small scale its seed-to-seed spread
# swamps every effect. These three quantities are deterministic given a
# checkpoint, cost one forward pass (or d backward passes for the exact
# Jacobian), and localise *where* the memory pathway fails:
#
#   recurrence_gain        rho(dm_t / dm_{t-1}) -- is memory WRITTEN forward?
#   memory_ablation_effect how much do outputs move if memory is zeroed --
#                          is memory READ at all?
#   gate_statistics        Eq. 7's g_rec vs g_loc -- is the channel open?
#
# The failure mode they were built to catch: memory can be read while
# rho(J) ~ 0, i.e. consumed at every position but never carried across steps.
# Accuracy cannot distinguish that from "memory is unused"; these can.


class RecurrenceDiagnostics(NamedTuple):
    rho: float            # spectral radius of dm_t/dm_{t-1}
    sigma_max: float      # largest singular value of the same Jacobian
    ablation_delta: float # max |logit| change when memory is zeroed
    g_rec: float          # mean recurrent gate (Eq. 7)
    g_loc: float          # mean local gate

    @property
    def propagates(self) -> bool:
        """Is anything written forward through the memory chain?"""
        return self.rho > 0.05

    @property
    def is_read(self) -> bool:
        """Is memory consumed at all, whether or not it is propagated?"""
        return self.ablation_delta > 1e-2

    def verdict(self) -> str:
        if not self.is_read:
            return "CHANNEL CLOSED: memory neither read nor propagated"
        if not self.propagates:
            return "WRITE SEVERED: memory is read but not carried across steps"
        return "chain live"


def recurrence_gain(
    model: MaglevModel,
    tokens: Tensor,
    mem_in: Tensor,
    *,
    positions: Sequence[int] = (),
) -> Tuple[float, float]:
    """Spectral radius and largest singular value of ``dm_t / dm_{t-1}``.

    The exact Jacobian, formed column by column -- ``d`` is small enough at the
    scales this is used at, and an estimate would blur exactly the distinction
    (rho ~ 0.01 vs rho ~ 0.7) the diagnostic exists to make.

    ``rho >= 1`` means errors compound through the recurrence and one-step
    consistency cannot bound the deployed deviation. ``rho ~ 0`` means the
    opposite failure and the one actually observed: nothing is carried at all.
    """
    t_len = tokens.shape[1]
    if not positions:
        positions = [t_len // 4, t_len // 2, 3 * t_len // 4]

    # One sequence at a time: batching would build a block-diagonal Jacobian
    # whose off-diagonal blocks are known to be zero, at B times the cost.
    tokens = tokens[:1]
    mem_in = mem_in[:1]

    rhos, sigmas = [], []
    for t in positions:
        base = mem_in.clone()

        def step(v: Tensor) -> Tensor:
            mi = base.clone()
            mi[:, t] = v.unsqueeze(0)
            _, m, _ = model.forward_decoder(tokens, mi)
            return m[0, t]

        jac = torch.autograd.functional.jacobian(step, base[0, t].clone())
        rhos.append(float(torch.linalg.eigvals(jac).abs().max()))
        sigmas.append(float(torch.linalg.svdvals(jac).max()))
    return sum(rhos) / len(rhos), sum(sigmas) / len(sigmas)


@torch.no_grad()
def memory_ablation_effect(model: MaglevModel, tokens: Tensor, mem_in: Tensor) -> float:
    """Largest logit change when the memory input is replaced by zeros."""
    live, _, _ = model.forward_decoder(tokens, mem_in)
    dead, _, _ = model.forward_decoder(tokens, torch.zeros_like(mem_in))
    return float((live - dead).abs().max())


@torch.no_grad()
def gate_statistics(model: MaglevModel, tokens: Tensor, mem_in: Tensor) -> Tuple[float, float]:
    """Mean of Eq. 7's gates, ``(g_rec, g_loc)``, averaged over layers."""
    out = model.forward_decoder(tokens, mem_in, need_mass=True)
    g_rec = torch.stack([a.g_rec.mean() for a in out[2]]).mean()
    h = model.embed(tokens)
    g_loc = torch.stack([
        (2 * torch.sigmoid(model.injections[i].G_loc(
            model.decoder_blocks[i].attn_norm(h)))).mean()
        for i in range(model.cfg.n_layers)
    ]).mean()
    return float(g_rec), float(g_loc)


def diagnose_recurrence(model: MaglevModel, tokens: Tensor) -> RecurrenceDiagnostics:
    """All three readouts from one teacher-forced pass."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            mem_in = MaglevModel.shift_memory(model.forward_prefiller(tokens))
        rho, sigma = recurrence_gain(model, tokens, mem_in)
        ablation = memory_ablation_effect(model, tokens, mem_in)
        g_rec, g_loc = gate_statistics(model, tokens, mem_in)
    finally:
        model.train(was_training)
    return RecurrenceDiagnostics(rho, sigma, ablation, g_rec, g_loc)


@torch.no_grad()
def prefiller_locality(
    model: MaglevModel,
    tokens: Tensor,
    *,
    n_probes: int = 16,
    seed: int = 0,
) -> Dict[str, float]:
    """How much of the prefiller's memory depends on tokens outside the window.

    This measures the *premise* of the degenerate optimum. The consistency term
    can reach zero with the memory channel switched off precisely when the
    prefiller's trajectory is computable from local context alone: a
    sliding-window decoder can then reproduce ``m'_t`` from the last W tokens
    with ``g_rec = 0``, satisfying the objective while carrying nothing across
    steps. If instead ``m'_t`` genuinely depends on distant tokens, that escape
    is unavailable and the consistency term must be paid for with real memory.

    Perturbs a token at distance > W before position t and reports the change in
    ``m'_t``, normalised by the change from perturbing a token *inside* the
    window. A ratio near 0 means the prefiller has gone local.
    """
    g = torch.Generator().manual_seed(seed)
    window = model.cfg.window
    t_len = tokens.shape[1]
    was_training = model.training
    model.eval()
    try:
        base = model.forward_prefiller(tokens)
        far_deltas, near_deltas = [], []
        for _ in range(n_probes):
            t = int(torch.randint(2 * window + 2, t_len, (1,), generator=g))
            far = int(torch.randint(0, t - window, (1,), generator=g))
            near = int(torch.randint(max(0, t - window + 1), t + 1, (1,), generator=g))
            for pos, bucket in ((far, far_deltas), (near, near_deltas)):
                pert = tokens.clone()
                pert[:, pos] = (pert[:, pos] + 1) % model.cfg.vocab_size
                m2 = model.forward_prefiller(pert)
                bucket.append(float((m2[:, t] - base[:, t]).norm(dim=-1).mean()))
    finally:
        model.train(was_training)

    far = sum(far_deltas) / len(far_deltas)
    near = sum(near_deltas) / len(near_deltas)
    return {
        "far_delta": far,
        "near_delta": near,
        # ~0 => the prefiller ignores everything outside the window, so the
        # degenerate zero-consistency solution is available to the decoder.
        "far_over_near": far / max(near, 1e-12),
    }


@torch.no_grad()
def memory_information(model: MaglevModel, tokens: Tensor) -> Dict[str, float]:
    """How much information the prefiller's memory trajectory actually carries.

    The consistency term ``lam * ||m_t - m'_t||`` attains its global minimum,
    zero, at ANY constant trajectory ``m_t = m'_t = c``. That solution carries
    no information and nothing in the objective forbids it; only cross-entropy
    opposes it, and only to the extent that the task needs memory at all.

    ``spread`` is the per-position standard deviation of ``m'`` relative to its
    own norm, so a large constant offset does not disguise a collapsed
    trajectory. Measured against a random-init control it falls by ~1800x at
    lambda >= 0.3 (0.071 -> 0.00004), which is the degenerate optimum being
    found rather than the model failing to start.
    """
    was_training = model.training
    model.eval()
    try:
        m_prime = model.forward_prefiller(tokens)
    finally:
        model.train(was_training)
    scale = m_prime.norm(dim=-1).mean().clamp_min(1e-12)
    centred = (m_prime[0] - m_prime[0].mean(dim=0)).double()
    return {
        "spread": float(m_prime.std(dim=1).mean() / scale),
        "norm": float(scale),
        "rank": float(torch.linalg.matrix_rank(centred, rtol=1e-3)),
    }
