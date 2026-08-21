"""RMSNorm, SwiGLU, self-attention, and the pre-norm transformer block.

Two deliberate factoring choices:

* The attention sublayer takes an optional ``inject`` callback, so the Maglev
  decoder can mix recurrent K/V in between projection and attention (Eq. 6-9)
  without this module knowing anything about memory.
* Layer kind ('S' sliding vs 'L' full-causal) is *not* stored on the block. It
  belongs to the path, not to the weights: under ``sharing="shared"`` one set of
  block weights is run by the prefiller under ``SLSL`` and by the decoder under
  ``SSSS``, so the caller supplies the mask.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Protocol, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ModelConfig
from .attention import RoPE, SlidingKVCache, attend

__all__ = [
    "RMSNorm",
    "SwiGLU",
    "SelfAttention",
    "TransformerBlock",
    "AttnAux",
    "InjectFn",
]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Normalise in fp32 regardless of autocast dtype; fp16 RMS is fragile.
        dtype = x.dtype
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class AttnAux(NamedTuple):
    """Per-layer statistics the read-influence estimators consume.

    ``attn_mass``  (B, H, S)    -- probability mass queries put on each key slot
    ``g_rec``      (B, T, d_kv) -- the recurrent gate of Eq. 7
    """

    attn_mass: Optional[Tensor]
    g_rec: Optional[Tensor]


class InjectFn(Protocol):
    """Mixes recurrent K/V into local K/V. Implemented by RecurrentInjection."""

    def __call__(
        self, a: Tensor, k: Tensor, v: Tensor, cos: Tensor, sin: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]: ...


class SelfAttention(nn.Module):
    """Multi-head (optionally grouped) attention with an injection hook."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.d_head = cfg.d_head

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_kv, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_kv, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

    def _split(self, x: Tensor, n_heads: int) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        *,
        rope: Optional[RoPE],
        positions: Tensor,
        mask: Optional[Tensor],
        impl: str,
        inject: Optional[InjectFn] = None,
        need_mass: bool = False,
        cache: Optional[SlidingKVCache] = None,
    ) -> Tuple[Tensor, AttnAux]:
        """``x`` is the *normalised* sublayer input; it also serves as ``a_t`` (Eq. 7)."""
        b, t, _ = x.shape
        q = self._split(self.q_proj(x), self.n_heads)
        k = self._split(self.k_proj(x), self.n_kv_heads)
        v = self._split(self.v_proj(x), self.n_kv_heads)

        if rope is not None:
            cos, sin = rope.get(positions)
            q = RoPE.rotate(q, cos, sin)
            k = RoPE.rotate(k, cos, sin)  # RoPE on local keys before mixing
        else:
            half = self.d_head // 2
            cos = torch.ones(t, half, device=x.device, dtype=torch.float32)
            sin = torch.zeros(t, half, device=x.device, dtype=torch.float32)

        g_rec: Optional[Tensor] = None
        if inject is not None:
            # Eq. 8: local k/v are replaced by the gated local+recurrent mixture.
            k, v, g_rec = inject(x, k, v, cos, sin)

        if cache is not None:
            k, v = cache.append(k, v)
            # The cache holds exactly the admissible window, so no mask is needed.
            # This is the one place parallel and recurrent could silently diverge.
            mask = None

        out, mass = attend(q, k, v, mask=mask, impl=impl, need_mass=need_mass)
        out = out.transpose(1, 2).reshape(b, t, self.n_heads * self.d_head)
        return self.o_proj(out), AttnAux(attn_mass=mass, g_rec=g_rec)


ResidualScale = Tuple[Union[Tensor, float], Union[Tensor, float]]


class TransformerBlock(nn.Module):
    """Pre-norm block: ``x + s_a * attn(norm(x))``, then ``h + s_f * ffn(norm(h))``.

    ``residual_scale`` is supplied by the caller rather than owned here, so the
    ``shared`` variant can run one set of block weights down two paths with
    different per-layer residual scaling (SPEC.md section 3).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = SelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(
        self,
        x: Tensor,
        *,
        rope: Optional[RoPE],
        positions: Tensor,
        mask: Optional[Tensor],
        impl: str,
        inject: Optional[InjectFn] = None,
        need_mass: bool = False,
        cache: Optional[SlidingKVCache] = None,
        residual_scale: Optional[ResidualScale] = None,
    ) -> Tuple[Tensor, AttnAux]:
        s_a, s_f = (1.0, 1.0) if residual_scale is None else residual_scale
        h, aux = self.attn(
            self.attn_norm(x),
            rope=rope,
            positions=positions,
            mask=mask,
            impl=impl,
            inject=inject,
            need_mass=need_mass,
            cache=cache,
        )
        x = x + s_a * h
        x = x + s_f * self.ffn(self.ffn_norm(x))
        return x, aux
