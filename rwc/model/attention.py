"""Sliding-window and full causal attention, RoPE, and the bounded KV cache.

Two attention backends are provided and selected by ``ModelConfig.attn_impl``:

* ``sdpa``  -- ``F.scaled_dot_product_attention``, the fast training path.
* ``math``  -- materialises the attention probabilities, which read-influence
  Estimator B needs in order to read ``attn_mass`` off (SPEC.md section 4).

Both must agree; the equivalence test exercises the ``math`` path because it is
the one with an exact, inspectable reduction order.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "RoPE",
    "SlidingKVCache",
    "attend",
    "causal_mask",
    "sliding_window_mask",
    "expand_kv_heads",
]


# --------------------------------------------------------------------------
# Rotary position embedding
# --------------------------------------------------------------------------


class RoPE(nn.Module):
    """Rotary embeddings, indexed by *absolute* position.

    Indexing by absolute position (rather than by offset within a window or a
    cache) is what lets the parallel and recurrent paths agree: both ask for the
    rotation of position ``t`` and get the identical tensor.
    """

    def __init__(self, d_head: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError(f"RoPE needs an even head dim, got {d_head}")
        half = d_head // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(pos, inv_freq)  # (max_seq_len, half)
        # Persistent=False: derived from config, no need to bloat checkpoints.
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def get(self, positions: Tensor) -> Tuple[Tensor, Tensor]:
        """Rotation factors for a 1-D tensor of absolute positions."""
        return self.cos[positions], self.sin[positions]

    @staticmethod
    def rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Rotate ``x`` of shape (B, H, T, d_head) given (T, d_head/2) factors."""
        x1, x2 = x.chunk(2, dim=-1)
        c = cos.to(x.dtype).view(1, 1, *cos.shape)
        s = sin.to(x.dtype).view(1, 1, *sin.shape)
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


# --------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------


def causal_mask(t_q: int, t_k: int, device: torch.device, q_offset: int = 0) -> Tensor:
    """Boolean keep-mask (t_q, t_k); True where the query may attend."""
    qi = torch.arange(t_q, device=device).unsqueeze(1) + q_offset
    kj = torch.arange(t_k, device=device).unsqueeze(0)
    return kj <= qi


def sliding_window_mask(
    t_q: int, t_k: int, window: int, device: torch.device, q_offset: int = 0
) -> Tensor:
    """Boolean keep-mask for Eq. 9's window ``k_bar[max(1, t-W+1) : t]``.

    In 0-based indexing that admits key ``j`` for query ``t`` exactly when
    ``0 <= t - j <= W - 1``, i.e. W entries including the query's own position.
    """
    qi = torch.arange(t_q, device=device).unsqueeze(1) + q_offset
    kj = torch.arange(t_k, device=device).unsqueeze(0)
    delta = qi - kj
    return (delta >= 0) & (delta < window)


def expand_kv_heads(x: Tensor, n_rep: int) -> Tensor:
    """Repeat grouped KV heads out to the number of query heads."""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


# --------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------


def attend(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    mask: Optional[Tensor] = None,
    impl: str = "sdpa",
    need_mass: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Attention over (B, H, T, d_head) queries and (B, Hkv, S, d_head) keys.

    Returns ``(out, attn_mass)``. ``attn_mass[b, h, j]`` is the total probability
    that queries place on key slot ``j``, summed over queries -- the quantity
    Estimator B calls ``attn_mass_j`` (SPEC.md section 4). It is only available
    from the ``math`` backend, since SDPA never materialises the probabilities.
    """
    n_rep = q.shape[1] // k.shape[1]
    k = expand_kv_heads(k, n_rep)
    v = expand_kv_heads(v, n_rep)

    if need_mass and impl != "math":
        raise ValueError("attn_mass requires impl='math'; SDPA never materialises it")

    if impl == "sdpa":
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return out, None

    if impl != "math":
        raise ValueError(f"unknown attention impl {impl!r}")

    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    mass = probs.sum(dim=-2) if need_mass else None  # sum over queries -> (B, H, S)
    return out, mass


# --------------------------------------------------------------------------
# Bounded KV cache
# --------------------------------------------------------------------------


class SlidingKVCache:
    """Bounded KV cache of size W for recurrent inference (Eq. 10).

    Entries are kept in chronological order and the oldest is shifted out, which
    costs an O(W) copy per step. A ring buffer would avoid the copy, but the
    window it returns would be out of order; attention is permutation-invariant
    over keys so that is *correct*, just harder to read and to debug against the
    parallel mask. Correctness first (SPEC.md section 10).
    """

    def __init__(
        self,
        batch: int,
        n_kv_heads: int,
        d_head: int,
        window: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.window = window
        self.n_filled = 0
        shape = (batch, n_kv_heads, window, d_head)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)

    def append(self, k_t: Tensor, v_t: Tensor) -> Tuple[Tensor, Tensor]:
        """Push one step's (B, Hkv, 1, d_head) K/V and return the valid window."""
        if self.n_filled < self.window:
            self.k[:, :, self.n_filled] = k_t[:, :, 0]
            self.v[:, :, self.n_filled] = v_t[:, :, 0]
            self.n_filled += 1
        else:
            self.k = torch.cat([self.k[:, :, 1:], k_t], dim=2)
            self.v = torch.cat([self.v[:, :, 1:], v_t], dim=2)
        return self.k[:, :, : self.n_filled], self.v[:, :, : self.n_filled]

    def reset(self) -> None:
        self.n_filled = 0
        self.k.zero_()
        self.v.zero_()
