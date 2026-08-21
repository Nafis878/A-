"""Reference points: a plain sliding-window transformer and a full-attention one.

Neither has memory or a prefiller. They bracket the Maglev decoder: SWA-only is
what P degrades to if the recurrent channel carries nothing, and full-attention
is the unbounded-context ceiling. Priority 6 in the SPEC's grid.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from ..config import ModelConfig
from .attention import RoPE, causal_mask, sliding_window_mask
from .blocks import RMSNorm, TransformerBlock
from .maglev import soft_cap

__all__ = ["PlainTransformer", "swa_baseline", "full_attention_baseline"]


class PlainTransformer(nn.Module):
    """A block stack with no recurrent memory, run under a fixed layer pattern."""

    def __init__(self, cfg: ModelConfig, pattern: str) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        if cfg.n_layers % len(pattern) != 0:
            raise ValueError(f"pattern {pattern!r} does not tile n_layers={cfg.n_layers}")
        self.kinds = list(pattern * (cfg.n_layers // len(pattern)))

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RoPE(cfg.d_head, cfg.max_seq_len, cfg.rope_theta) if cfg.rope else None
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = None if cfg.tie_embeddings else nn.Linear(
            cfg.d_model, cfg.vocab_size, bias=False
        )

        self._mask_cache: Dict[Tuple[str, int, str], Tensor] = {}
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def _mask(self, kind: str, t: int, device: torch.device) -> Tensor:
        key = (kind, t, str(device))
        if key not in self._mask_cache:
            self._mask_cache[key] = (
                sliding_window_mask(t, t, self.cfg.window, device)
                if kind == "S"
                else causal_mask(t, t, device)
            )
        return self._mask_cache[key]

    def forward(self, tokens: Tensor, *, impl: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        """Returns ``(logits, h)`` where ``h`` is the post-final-norm state."""
        impl = impl or self.cfg.attn_impl
        t = tokens.shape[1]
        positions = torch.arange(t, device=tokens.device)
        h = self.embed(tokens)
        for block, kind in zip(self.blocks, self.kinds):
            h, _ = block(
                h,
                rope=self.rope,
                positions=positions,
                mask=self._mask(kind, t, tokens.device),
                impl=impl,
            )
        h = self.norm(h)
        w = self.embed.weight if self.lm_head is None else self.lm_head.weight
        return soft_cap(h @ w.t(), self.cfg.logit_cap), h


def swa_baseline(cfg: ModelConfig) -> PlainTransformer:
    """Sliding-window only -- the bounded-context floor."""
    return PlainTransformer(cfg, "S")


def full_attention_baseline(cfg: ModelConfig) -> PlainTransformer:
    """Full causal attention -- the unbounded-context ceiling."""
    return PlainTransformer(cfg, "L")
