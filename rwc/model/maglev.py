"""Maglev prefiller Q, decoder P, and the recurrent K/V injection.

Equation numbers throughout refer to SPEC.md section 3.

The central invariant of this file is the **shift**. The prefiller emits a
memory target at every position, ``m'_{1:T} = Q(x_{1:T})`` with ``m'_0 = 0``.
The decoder at position ``t`` consumes ``m'_{t-1}`` -- the *previous* position's
target -- and emits ``m_t`` (Eq. 3). That is the parallel, teacher-forced
simulation of inference, where at step ``t`` the decoder can only hold
``m_{t-1}``, since ``m_t`` is precisely what it is about to produce. Feed
``m'_t`` at position ``t`` instead and the consistency term collapses toward
triviality, because the decoder can learn to pass its input through to its
output. ``shift_memory`` is the only place this happens and it is asserted in
``tests/test_recurrence_equivalence.py``.
"""

from __future__ import annotations

from functools import partial
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..config import ModelConfig
from .attention import RoPE, SlidingKVCache, causal_mask, sliding_window_mask
from .blocks import AttnAux, RMSNorm, TransformerBlock

__all__ = ["RecurrentInjection", "MaglevModel", "MaglevOutput", "soft_cap"]


def soft_cap(logits: Tensor, cap: float) -> Tensor:
    """Eq. 4 logit soft-cap: ``cap * tanh(o / cap)``."""
    return cap * torch.tanh(logits / cap)


class MaglevOutput(NamedTuple):
    logits: Tensor  # (B, T, vocab), already soft-capped
    m: Tensor  # (B, T, d) decoder memory, post-final-norm
    m_prime: Optional[Tensor]  # (B, T, d) prefiller targets, None in recurrent mode
    aux: Tuple[AttnAux, ...]  # per decoder layer


class RecurrentInjection(nn.Module):
    """Layer-specific recurrent K/V injection, Eq. 6-8.

    ``W_k_rec``/``W_v_rec`` are shared across the sequence but layer-specific,
    per SPEC.md section 3.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_kv_heads = cfg.n_kv_heads
        self.d_head = cfg.d_head

        self.W_k_rec = nn.Linear(cfg.d_model, cfg.d_kv, bias=False)  # Eq. 6
        self.W_v_rec = nn.Linear(cfg.d_model, cfg.d_kv, bias=False)  # Eq. 6
        self.G_loc = nn.Linear(cfg.d_model, cfg.d_kv, bias=False)  # Eq. 7
        self.G_rec = nn.Linear(cfg.d_model, cfg.d_kv, bias=False)  # Eq. 7
        # RMSNorm of Eq. 6, computed per head over d_head with a per-key-dim gain.
        self.k_rec_gain = nn.Parameter(torch.ones(cfg.d_kv))

        # Ground-truth read structure for tests/test_influence_groundtruth.py.
        # Because W_k_rec and W_v_rec are the *only* route from memory into the
        # network, scaling memory by a known gain vector makes each dimension's
        # readability known by construction; a zero gain is provably unreadable.
        if cfg.carrier_gain_seed is None:
            self.register_buffer("carrier_gain", None, persistent=False)
        else:
            self.register_buffer(
                "carrier_gain", _carrier_gain(cfg), persistent=True
            )

    def _norm_k_rec(self, k_rec: Tensor) -> Tensor:
        """RMSNorm over d_head, per head (Eq. 6)."""
        b, t, _ = k_rec.shape
        x = k_rec.view(b, t, self.n_kv_heads, self.d_head).float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.cfg.norm_eps)
        x = x.view(b, t, -1) * self.k_rec_gain.float()
        return x.to(k_rec.dtype)

    def _split(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_kv_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        a: Tensor,
        k: Tensor,
        v: Tensor,
        cos: Tensor,
        sin: Tensor,
        *,
        mem: Tensor,
        detach_gate_input: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Mix recurrent K/V into local K/V.

        ``a``   (B, T, d)      normalised residual entering the sublayer (Eq. 7)
        ``k,v`` (B, Hkv, T, dh) local keys/values, RoPE already applied to ``k``
        ``mem`` (B, T, d)      memory read at each position, i.e. ``m'_{t-1}``

        ``detach_gate_input`` is used only by read-influence Estimator A: the
        gates are the one place a *local*-path dependence on memory leaks in, so
        detaching them confines the gradient to the recurrent channel.
        """
        if self.carrier_gain is not None:
            mem = mem * self.carrier_gain

        k_rec = self._norm_k_rec(self.W_k_rec(mem))  # Eq. 6
        v_rec = self.W_v_rec(mem)  # Eq. 6

        a_gate = a.detach() if detach_gate_input else a
        g_loc = 2.0 * torch.sigmoid(self.G_loc(a_gate))  # Eq. 7, neutral at 0
        g_rec = 2.0 * torch.sigmoid(self.G_rec(a_gate))  # Eq. 7

        k_rec = RoPE.rotate(self._split(k_rec), cos, sin)  # RoPE before mixing
        v_rec = self._split(v_rec)
        g_loc_h = self._split(g_loc)
        g_rec_h = self._split(g_rec)

        k_bar = g_loc_h * k + g_rec_h * k_rec  # Eq. 8, elementwise
        v_bar = g_loc_h * v + g_rec_h * v_rec  # Eq. 8
        return k_bar, v_bar, g_rec


def _carrier_gain(cfg: ModelConfig) -> Tensor:
    """Fixed, known per-dimension readability for the influence validity test.

    Graded rather than binary: a two-level mask would make the ground-truth
    ranking almost all ties, which strips Spearman of the power to detect that
    an estimator is ordering the readable dimensions correctly.
    """
    g = torch.Generator().manual_seed(int(cfg.carrier_gain_seed))
    d = cfg.d_model
    n_carrier = max(1, int(round(cfg.carrier_frac * d)))
    gain = torch.zeros(d)
    idx = torch.randperm(d, generator=g)[:n_carrier]
    # Log-uniform over two decades, so the carriers themselves are well separated.
    gain[idx] = torch.exp(torch.rand(n_carrier, generator=g) * 4.6 - 2.3)
    return gain


class MaglevModel(nn.Module):
    """Prefiller Q (training only) coupled to sliding-window decoder P."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.prefiller_kinds = cfg.layer_kinds("prefiller")
        self.decoder_kinds = cfg.layer_kinds("decoder")

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RoPE(cfg.d_head, cfg.max_seq_len, cfg.rope_theta) if cfg.rope else None

        if cfg.sharing == "shared":
            blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
            self.prefiller_blocks = blocks
            self.decoder_blocks = blocks
            self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.prefiller_norm = self.norm
            self.decoder_norm = self.norm
            # Separate residual scaling on the decoder path (SPEC.md section 3):
            # the shared blocks see a different input distribution there.
            self.dec_res_scale = nn.Parameter(torch.ones(cfg.n_layers, 2))
        else:
            self.prefiller_blocks = nn.ModuleList(
                TransformerBlock(cfg) for _ in range(cfg.n_layers)
            )
            self.decoder_blocks = nn.ModuleList(
                TransformerBlock(cfg) for _ in range(cfg.n_layers)
            )
            self.prefiller_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.decoder_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.dec_res_scale = None

        self.injections = nn.ModuleList(
            RecurrentInjection(cfg) for _ in range(cfg.n_layers)
        )

        if cfg.tie_embeddings:
            self.lm_head = None  # uses embed.weight
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self._mask_cache: Dict[Tuple[str, int, str], Tensor] = {}
        self.apply(self._init_weights)

    # ---- construction helpers --------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _mask(self, kind: str, t: int, device: torch.device) -> Tensor:
        key = (kind, t, str(device))
        if key not in self._mask_cache:
            if kind == "S":
                mask = sliding_window_mask(t, t, self.cfg.window, device)
            else:
                mask = causal_mask(t, t, device)
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    def _head(self, m: Tensor) -> Tensor:
        w = self.embed.weight if self.lm_head is None else self.lm_head.weight
        return soft_cap(m @ w.t(), self.cfg.logit_cap)  # Eq. 4

    # ---- the shift --------------------------------------------------------

    @staticmethod
    def shift_memory(m_prime: Tensor) -> Tensor:
        """Eq. 3's shift: ``mem_in[:, 0] = 0`` and ``mem_in[:, t] = m'[:, t-1]``.

        Built with a cat rather than in-place assignment so the graph back into
        the prefiller stays intact.
        """
        b, _, d = m_prime.shape
        zero = m_prime.new_zeros(b, 1, d)
        return torch.cat([zero, m_prime[:, :-1]], dim=1)

    # ---- forward passes ---------------------------------------------------

    def forward_prefiller(self, tokens: Tensor, *, impl: Optional[str] = None) -> Tensor:
        """``m'_{1:T} = Q(x_{1:T})`` -- the post-final-norm prefiller state."""
        impl = impl or self.cfg.attn_impl
        t = tokens.shape[1]
        positions = torch.arange(t, device=tokens.device)
        h = self.embed(tokens)
        for block, kind in zip(self.prefiller_blocks, self.prefiller_kinds):
            h, _ = block(
                h,
                rope=self.rope,
                positions=positions,
                mask=self._mask(kind, t, tokens.device),
                impl=impl,
            )
        return self.prefiller_norm(h)

    def forward_decoder(
        self,
        tokens: Tensor,
        mem_in: Tensor,
        *,
        positions: Optional[Tensor] = None,
        caches: Optional[Sequence[SlidingKVCache]] = None,
        need_mass: bool = False,
        detach_gate_input: bool = False,
        impl: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor, Tuple[AttnAux, ...]]:
        """``m_{1:T} = P(x_{1:T}, mem_in)``. ``mem_in`` is already shifted."""
        impl = impl or self.cfg.attn_impl
        if need_mass:
            impl = "math"  # attn_mass is only available from the math backend
        t = tokens.shape[1]
        if positions is None:
            positions = torch.arange(t, device=tokens.device)

        h = self.embed(tokens)
        aux: List[AttnAux] = []
        for i, (block, kind) in enumerate(zip(self.decoder_blocks, self.decoder_kinds)):
            inject = partial(
                self.injections[i], mem=mem_in, detach_gate_input=detach_gate_input
            )
            scale = None
            if self.dec_res_scale is not None:
                scale = (self.dec_res_scale[i, 0], self.dec_res_scale[i, 1])
            h, a = block(
                h,
                rope=self.rope,
                positions=positions,
                mask=None if caches is not None else self._mask(kind, t, tokens.device),
                impl=impl,
                inject=inject,
                need_mass=need_mass,
                cache=caches[i] if caches is not None else None,
                residual_scale=scale,
            )
            aux.append(a)

        m = self.decoder_norm(h)  # the decoder's post-final-norm state *is* m_t
        return self._head(m), m, tuple(aux)

    def forward_train(
        self,
        tokens: Tensor,
        *,
        need_mass: bool = False,
        detach_prefiller: bool = False,
        impl: Optional[str] = None,
    ) -> MaglevOutput:
        """The two sequence-parallel training passes of Eq. 3.

        ``detach_prefiller`` cuts the decoder's gradient into Q. Default off:
        with it on, Q would be trained only by the consistency term, which pulls
        it toward P and is degenerate (see README assumptions).
        """
        m_prime = self.forward_prefiller(tokens, impl=impl)
        mem_in = self.shift_memory(m_prime.detach() if detach_prefiller else m_prime)
        logits, m, aux = self.forward_decoder(
            tokens, mem_in, need_mass=need_mass, impl=impl
        )
        return MaglevOutput(logits=logits, m=m, m_prime=m_prime, aux=aux)

    # ---- recurrent inference (Eq. 10) -------------------------------------

    def make_caches(
        self, batch: int, *, device: torch.device, dtype: torch.dtype
    ) -> List[SlidingKVCache]:
        return [
            SlidingKVCache(
                batch,
                self.cfg.n_kv_heads,
                self.cfg.d_head,
                self.cfg.window,
                device=device,
                dtype=dtype,
            )
            for _ in range(self.cfg.n_layers)
        ]

    def forward_recurrent(
        self,
        tokens: Tensor,
        *,
        mem_in_override: Optional[Tensor] = None,
        impl: Optional[str] = None,
        keep_graph: bool = False,
    ) -> MaglevOutput:
        """Eq. 10: Q is discarded and P consumes its own ``m_{t-1}``.

        ``mem_in_override`` (B, T, d) replaces the model's own state at every
        step -- i.e. it is fed directly as ``mem_in``, already shifted. This is
        what ``test_recurrence_equivalence`` uses to force the two paths to see
        identical memory. ``keep_graph`` retains the autograd graph across steps,
        which read-influence Estimator A needs for its truncated rollout.
        """
        b, t = tokens.shape
        device = tokens.device
        dtype = self.embed.weight.dtype
        caches = self.make_caches(b, device=device, dtype=dtype)

        m_prev = torch.zeros(b, 1, self.cfg.d_model, device=device, dtype=dtype)
        logits_steps: List[Tensor] = []
        m_steps: List[Tensor] = []

        for step in range(t):
            mem_t = (
                mem_in_override[:, step : step + 1]
                if mem_in_override is not None
                else m_prev
            )
            pos = torch.arange(step, step + 1, device=device)
            logits_t, m_t, _ = self.forward_decoder(
                tokens[:, step : step + 1],
                mem_t,
                positions=pos,
                caches=caches,
                impl=impl,
            )
            logits_steps.append(logits_t)
            m_steps.append(m_t)
            m_prev = m_t if keep_graph else m_t.detach()

        return MaglevOutput(
            logits=torch.cat(logits_steps, dim=1),
            m=torch.cat(m_steps, dim=1),
            m_prime=None,
            aux=(),
        )

    def forward(self, tokens: Tensor, **kwargs) -> MaglevOutput:
        return self.forward_train(tokens, **kwargs)
