"""Read-influence estimators (SPEC.md section 4).

Read-influence ``w_j`` measures how much each dimension of memory ``m_j``
affects predictions at positions ``j+1 .. j+H`` *through the recurrent K/V
channel* -- not general position importance.

Two estimators:

* **A, gradient** -- the reference. Exact, slow, used to validate B and to
  produce the C1 diagnostic. Run as a truncated *recurrent* rollout: in
  parallel teacher-forced mode the memory chain is broken (the memory input is
  the prefiller's output, not the decoder's own state), so a parallel gradient
  would see only within-window and cross-layer propagation and would understate
  influence for ``H > W``. ``tests/test_recurrence_equivalence.py`` pins that
  bound at ``L*(W-1)``.
* **B, saliency** -- what RWC actually uses at training time, because it costs
  one extra reduction over cached attention statistics rather than a backward
  pass.

Indexing convention, which is the easiest thing here to get wrong: the decoder
at position ``t`` consumes ``mem_in[t] = m'_{t-1}``. So the influence of memory
``m_j`` is the influence of the memory *input* at position ``j+1``. Everything
below computes in mem-input space and shifts once, in ``_to_memory_index``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import InfluenceConfig
from .losses import IGNORE_INDEX
from .model.maglev import MaglevModel

__all__ = [
    "grad_influence",
    "saliency_influence",
    "InfluenceEMA",
    "normalise",
    "spearman",
]


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


def spearman(a: Tensor, b: Tensor) -> float:
    """Spearman rank correlation between two 1-D tensors."""
    from scipy.stats import spearmanr

    rho = spearmanr(a.detach().cpu().numpy(), b.detach().cpu().numpy()).statistic
    return float(rho)


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


@torch.no_grad()
def saliency_influence(
    model: MaglevModel,
    tokens: Tensor,
    *,
    eps: float = 1e-8,
    normalise_out: bool = True,
) -> Tensor:
    """Gradient x input style saliency from cached attention statistics.

    ``w_j ~ | m_j * sum_l ( W_k_rec^T (attn_mass_j * g_rec_j)
                          + W_v_rec^T (attn_mass_j * g_rec_j) ) |``

    Returns ``(B, T, d)`` indexed by *memory* position ``j``.
    """
    out = model.forward_train(tokens, need_mass=True)
    mem_in = MaglevModel.shift_memory(out.m_prime)

    total = torch.zeros_like(mem_in)
    for layer, aux in enumerate(out.aux):
        inj = model.injections[layer]
        # attn_mass arrives per *query* head; fold it back onto kv heads, then
        # broadcast each kv head's mass across the d_head dims it owns.
        mass = _mass_to_kv_dims(aux.attn_mass, model.cfg)
        s = mass * aux.g_rec  # (B, T, d_kv)
        # Linear weight is (d_kv, d), so `s @ W` applies W^T -- the transpose in
        # the SPEC's formula, without materialising it.
        total = total + s @ inj.W_k_rec.weight + s @ inj.W_v_rec.weight

    mem_eff = mem_in
    gain = model.injections[0].carrier_gain
    if gain is not None:
        # The gain is part of the model's read path, so an estimator that
        # ignored it would be describing a network that does not exist.
        mem_eff = mem_eff * gain

    w = (mem_eff * total).abs()
    if normalise_out:
        w = normalise(w, eps)
    # After this, position T-1 is exactly zero: m'_{T-1} would be consumed at
    # position T, which does not exist. It is excluded from the loss, not
    # normalised to something.
    return _to_memory_index(w)


def _mass_to_kv_dims(mass: Tensor, cfg) -> Tensor:
    """(B, n_heads, T) attention mass -> (B, T, d_kv)."""
    b, n_heads, t = mass.shape
    n_rep = n_heads // cfg.n_kv_heads
    per_kv = mass.view(b, cfg.n_kv_heads, n_rep, t).mean(dim=2)  # (B, Hkv, T)
    per_kv = per_kv.permute(0, 2, 1)  # (B, T, Hkv)
    return per_kv.repeat_interleave(cfg.d_head, dim=-1)  # (B, T, d_kv)


# --------------------------------------------------------------------------
# EMA smoothing, so the per-step cost is negligible
# --------------------------------------------------------------------------


class InfluenceEMA:
    """Slow EMA of read-influence, refreshed every ``refresh_every`` steps.

    The trainer logs ``stats()`` every step so a collapse is caught early:
    ``w`` going uniform means RWC has degenerated into the uniform L2 it is
    meant to improve on, and ``w`` spiking onto one dimension means the
    consistency term has stopped constraining the memory at all. Both are
    failure modes, and both look like "training is fine" from the loss alone.
    """

    def __init__(self, cfg: InfluenceConfig, d_model: int, device: torch.device) -> None:
        self.cfg = cfg
        self.d_model = d_model
        self.value = torch.full((d_model,), 1.0 / d_model, device=device)
        self.n_updates = 0

    def should_refresh(self, step: int) -> bool:
        return step % self.cfg.refresh_every == 0

    def update(self, w: Tensor) -> Tensor:
        """Fold a fresh estimate in. ``w`` is reduced to ``(d,)`` and renormalised."""
        fresh = w.detach()
        while fresh.dim() > 1:
            fresh = fresh.mean(dim=0)
        fresh = normalise(fresh, self.cfg.eps)
        decay = self.cfg.ema_decay
        self.value = decay * self.value + (1.0 - decay) * fresh.to(self.value.device)
        self.value = normalise(self.value, self.cfg.eps)
        self.n_updates += 1
        return self.value

    def stats(self) -> Dict[str, float]:
        v = self.value
        uniform = 1.0 / self.d_model
        p = v / (v.sum() + self.cfg.eps)
        entropy = float(-(p * (p + self.cfg.eps).log()).sum())
        return {
            "w_mean": float(v.mean()),
            "w_std": float(v.std()),
            "w_max": float(v.max()),
            # 1.0 = perfectly uniform (RWC has degenerated to uniform L2),
            # 0.0 = all mass on one dimension.
            "w_entropy_ratio": entropy / float(torch.tensor(self.d_model).log()),
            "w_max_over_uniform": float(v.max()) / uniform,
        }

    def state_dict(self) -> Dict[str, object]:
        return {"value": self.value, "n_updates": self.n_updates}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.value = torch.as_tensor(state["value"]).to(self.value.device)
        self.n_updates = int(state["n_updates"])
