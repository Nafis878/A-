"""Parallel-vs-recurrent equivalence of the Maglev decoder — the critical test.

The whole training regime is a *parallel* simulation of a *sequential* process.
If the two paths do not compute the same function given the same memory, we
train one model and evaluate a different one: the consistency loss, the
read-influence estimates and every probe-accuracy-vs-distance curve would be
measuring a fiction, and they would fail silently, converging to plausible
numbers that mean nothing.

The realistic bugs are all quiet ones — an off-by-one in the shift, RoPE offsets
differing between windowed-parallel and cached-recurrent attention, a
`max(1, t-W+1)` boundary that is inclusive on one path and exclusive on the
other, a gate read off the wrong residual, KV-cache eviction misaligned with the
parallel mask. Every one of those still trains.
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

from rwc.config import ModelConfig
from rwc.model.attention import sliding_window_mask
from rwc.model.maglev import MaglevModel

# Deliberately smaller than the window in places and larger in others, so the
# partially-filled cache and the evicting cache are both exercised.
WINDOW = 8
SEQ_LEN = 24


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_mult=2.0,
        vocab_size=17,
        max_seq_len=64,
        window=WINDOW,
        prefiller_pattern="SL",
        decoder_pattern="SS",
        sharing="shared",
        rope=True,
        attn_impl="math",
    )
    base.update(kw)
    return ModelConfig(**base)


def build(dtype: torch.dtype = torch.float64, **kw) -> Tuple[MaglevModel, torch.Tensor]:
    torch.manual_seed(0)
    model = MaglevModel(tiny_cfg(**kw)).to(dtype).eval()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, SEQ_LEN))
    return model, tokens


def random_memory(model: MaglevModel, tokens: torch.Tensor, dtype: torch.dtype):
    g = torch.Generator().manual_seed(1234)
    return torch.randn(
        tokens.shape[0], tokens.shape[1], model.cfg.d_model, generator=g, dtype=dtype
    )


# --------------------------------------------------------------------------
# The shift
# --------------------------------------------------------------------------


def test_shift_is_exactly_one_position() -> None:
    """Eq. 3: at position t the decoder consumes m'_{t-1}, and m'_0 = 0."""
    m_prime = torch.randn(3, 7, 5)
    mem_in = MaglevModel.shift_memory(m_prime)

    assert mem_in.shape == m_prime.shape
    assert torch.all(mem_in[:, 0] == 0), "m'_0 must be zero"
    torch.testing.assert_close(mem_in[:, 1:], m_prime[:, :-1])
    # And the un-shifted target must NOT appear at its own position, which is
    # the bug that would make the consistency term trivial.
    assert not torch.allclose(mem_in[:, 1], m_prime[:, 1])


def test_shift_preserves_gradient_to_the_prefiller() -> None:
    m_prime = torch.randn(2, 5, 4, requires_grad=True)
    MaglevModel.shift_memory(m_prime).sum().backward()
    assert m_prime.grad is not None
    # Position T-1 is never read by anyone, so it must receive no gradient.
    assert torch.all(m_prime.grad[:, -1] == 0)
    assert torch.all(m_prime.grad[:, :-1] == 1)


def test_forward_train_uses_the_shifted_memory() -> None:
    """The decoder must see shift(m'), not m' itself."""
    model, tokens = build()
    out = model.forward_train(tokens)
    mem_in = MaglevModel.shift_memory(out.m_prime)

    _, m_shifted, _ = model.forward_decoder(tokens, mem_in)
    torch.testing.assert_close(m_shifted, out.m)

    _, m_unshifted, _ = model.forward_decoder(tokens, out.m_prime)
    assert not torch.allclose(m_unshifted, out.m), (
        "feeding unshifted m' produced the same result; the shift is not wired in"
    )


# --------------------------------------------------------------------------
# The equivalence itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("sharing", ["shared", "separate"])
@pytest.mark.parametrize("rope", [True, False])
def test_parallel_matches_recurrent(
    dtype: torch.dtype, sharing: str, rope: bool, request
) -> None:
    """Same memory in, same logits out, whichever way the decoder is run."""
    model, tokens = build(dtype=dtype, sharing=sharing, rope=rope)
    memory = random_memory(model, tokens, dtype)
    mem_in = MaglevModel.shift_memory(memory)

    with torch.no_grad():
        logits_par, m_par, _ = model.forward_decoder(tokens, mem_in)
        rec = model.forward_recurrent(tokens, mem_in_override=mem_in)

    dev_logits = (logits_par - rec.logits).abs().max().item()
    dev_m = (m_par - rec.m).abs().max().item()
    scale = logits_par.abs().max().item()
    print(
        f"\n[{dtype} {sharing} rope={rope}] max|Δlogits|={dev_logits:.3e} "
        f"max|Δm|={dev_m:.3e} (logit scale {scale:.3f})"
    )

    tol = 1e-10 if dtype == torch.float64 else 2e-4
    assert dev_logits < tol, f"parallel and recurrent paths disagree by {dev_logits:.3e}"
    assert dev_m < tol


def test_recurrent_rollout_is_self_consistent() -> None:
    """Roll out on its own memory, then replay that trajectory in parallel."""
    model, tokens = build()
    with torch.no_grad():
        rec = model.forward_recurrent(tokens)
        # The decoder produced m_t at each step while consuming m_{t-1}, so the
        # memory it actually saw is shift(rec.m). Replaying that in parallel
        # must reproduce the same outputs.
        logits_par, m_par, _ = model.forward_decoder(
            tokens, MaglevModel.shift_memory(rec.m)
        )

    dev = (logits_par - rec.logits).abs().max().item()
    print(f"\n[self-consistent rollout] max|Δlogits|={dev:.3e}")
    assert dev < 1e-10
    torch.testing.assert_close(m_par, rec.m, atol=1e-10, rtol=0)


def test_zero_memory_is_not_a_degenerate_no_op() -> None:
    """Guard the equivalence test against passing because memory does nothing."""
    model, tokens = build()
    memory = random_memory(model, tokens, torch.float64)
    with torch.no_grad():
        a, _, _ = model.forward_decoder(tokens, MaglevModel.shift_memory(memory))
        b, _, _ = model.forward_decoder(tokens, torch.zeros_like(memory))
    assert (a - b).abs().max().item() > 1e-6, "memory has no effect on the output"


# --------------------------------------------------------------------------
# Window boundary
# --------------------------------------------------------------------------


def test_sliding_window_mask_matches_equation_9() -> None:
    """Eq. 9's k_bar[max(1, t-W+1) : t] is W entries, inclusive of t."""
    mask = sliding_window_mask(10, 10, WINDOW, torch.device("cpu"))
    assert mask[0, 0] and not mask[0, 1]  # position 0 sees only itself
    assert mask[9, 9] and mask[9, 9 - WINDOW + 1]
    assert not mask[9, 9 - WINDOW]
    assert mask.sum(dim=1).tolist() == [min(t + 1, WINDOW) for t in range(10)]


def test_token_outside_window_cannot_reach_the_query() -> None:
    """With memory zeroed, the decoder is strictly local — verify it."""
    model, tokens = build()
    zero_mem = torch.zeros(tokens.shape[0], tokens.shape[1], model.cfg.d_model, dtype=torch.float64)

    # Two sliding layers stack their windows, so influence reaches back
    # n_layers*(W-1); perturb beyond that.
    query = SEQ_LEN - 1
    far = query - model.cfg.n_layers * (WINDOW - 1) - 1
    assert far >= 0

    perturbed = tokens.clone()
    perturbed[:, far] = (perturbed[:, far] + 1) % model.cfg.vocab_size

    with torch.no_grad():
        a, _, _ = model.forward_decoder(tokens, zero_mem)
        b, _, _ = model.forward_decoder(perturbed, zero_mem)

    assert torch.equal(a[:, query], b[:, query]), (
        "a token outside the stacked sliding windows changed the output"
    )
    assert not torch.equal(a[:, far], b[:, far]), "the perturbation had no effect at all"


def test_memory_reaches_past_the_window_but_is_bounded_in_parallel_mode() -> None:
    """Memory extends reach past W — but in parallel mode only by L*(W-1).

    ``m'_j`` enters at decoder position j+1, is read by queries within that
    layer's window, and each further sliding layer extends the reach by W-1.
    In parallel teacher-forced mode there is no chain beyond that, because the
    memory input is the prefiller's output rather than the decoder's own state.
    This is exactly why read-influence Estimator A must run as a *recurrent*
    rollout for horizons H > W (README assumption 11).
    """
    model, tokens = build()
    memory = random_memory(model, tokens, torch.float64)
    j, L, W = 0, model.cfg.n_layers, model.cfg.window
    last_reached = j + 1 + L * (W - 1)
    assert last_reached < SEQ_LEN - 1, "sequence too short to see the boundary"

    bumped = memory.clone()
    bumped[:, j] += 5.0
    with torch.no_grad():
        a, _, _ = model.forward_decoder(tokens, MaglevModel.shift_memory(memory))
        b, _, _ = model.forward_decoder(tokens, MaglevModel.shift_memory(bumped))
    delta = (a - b).abs().amax(dim=(0, 2))

    assert delta[j + 1] > 0, "memory had no effect at the position that reads it"
    assert delta[W] > 0, "memory did not reach the edge of the first window"
    assert delta[last_reached] > 0, f"memory did not reach position {last_reached}"
    assert delta[last_reached + 1] == 0, (
        f"memory reached position {last_reached + 1}, further than L*(W-1) allows"
    )


def test_recurrent_mode_carries_information_arbitrarily_far() -> None:
    """The memory *chain* only exists when the decoder consumes its own state."""
    model, tokens = build()
    L, W = model.cfg.n_layers, model.cfg.window
    query = SEQ_LEN - 1
    # Beyond anything local attention plus one teacher-forced read could carry.
    assert query > 1 + L * (W - 1)

    perturbed = tokens.clone()
    perturbed[:, 0] = (perturbed[:, 0] + 1) % model.cfg.vocab_size
    with torch.no_grad():
        a = model.forward_recurrent(tokens)
        b = model.forward_recurrent(perturbed)
    assert (a.logits[:, query] - b.logits[:, query]).abs().max().item() > 0, (
        "token 0 did not reach the final position; the recurrent chain is broken"
    )


# --------------------------------------------------------------------------
# Backend agreement
# --------------------------------------------------------------------------


def test_sdpa_and_math_backends_agree() -> None:
    """Estimator B runs under 'math'; training runs under 'sdpa'."""
    model, tokens = build()
    memory = random_memory(model, tokens, torch.float64)
    mem_in = MaglevModel.shift_memory(memory)
    with torch.no_grad():
        a, _, _ = model.forward_decoder(tokens, mem_in, impl="sdpa")
        b, _, _ = model.forward_decoder(tokens, mem_in, impl="math")
    dev = (a - b).abs().max().item()
    print(f"\n[sdpa vs math] max|Δlogits|={dev:.3e}")
    assert dev < 1e-10


def test_attention_mass_is_a_probability_sum() -> None:
    """attn_mass[j] is the total probability queries place on key slot j."""
    model, tokens = build()
    memory = random_memory(model, tokens, torch.float64)
    with torch.no_grad():
        _, _, aux = model.forward_decoder(
            tokens, MaglevModel.shift_memory(memory), need_mass=True
        )
    for layer in aux:
        assert layer.attn_mass is not None
        assert layer.g_rec is not None
        # Every query distributes exactly 1.0 of mass, so the grand total is T.
        total = layer.attn_mass.sum(dim=-1)
        torch.testing.assert_close(total, torch.full_like(total, float(SEQ_LEN)))
        assert torch.all(layer.attn_mass >= 0)
        # Eq. 7's gates live in (0, 2) and are neutral at 1.
        assert torch.all((layer.g_rec > 0) & (layer.g_rec < 2))


# --------------------------------------------------------------------------
# Shapes (M2 extension of tests/test_shapes.py's contract)
# --------------------------------------------------------------------------


def test_output_shapes() -> None:
    model, tokens = build()
    out = model.forward_train(tokens, need_mass=True)
    b, t = tokens.shape
    assert out.logits.shape == (b, t, model.cfg.vocab_size)
    assert out.m.shape == (b, t, model.cfg.d_model)
    assert out.m_prime.shape == (b, t, model.cfg.d_model)
    assert len(out.aux) == model.cfg.n_layers
    # Eq. 4's soft cap bounds the logits.
    assert out.logits.abs().max().item() <= model.cfg.logit_cap


def test_shared_and_separate_differ_in_parameter_count() -> None:
    shared = MaglevModel(tiny_cfg(sharing="shared"))
    separate = MaglevModel(tiny_cfg(sharing="separate"))
    n_shared = sum(p.numel() for p in shared.parameters())
    n_separate = sum(p.numel() for p in separate.parameters())
    assert n_separate > n_shared
    assert shared.prefiller_blocks is shared.decoder_blocks
    assert separate.prefiller_blocks is not separate.decoder_blocks


@pytest.mark.parametrize("name", ["tiny_synth", "small_lm", "medium_lm"])
def test_estimated_param_count_matches_reality(name: str) -> None:
    """n_params_estimate() is used to budget Colab runs; keep it honest."""
    from pathlib import Path

    from rwc.config import load_config

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / f"{name}.yaml", cuda_available=False).model
    model = MaglevModel(cfg)
    actual = sum(p.numel() for p in model.parameters())
    predicted = cfg.n_params_estimate()["total"]
    print(f"\n[{name}] actual {actual:,} predicted {predicted:,}")
    assert actual == predicted


@pytest.mark.parametrize("sharing", ["shared", "separate"])
def test_estimate_is_exact_for_both_sharing_modes(sharing: str) -> None:
    cfg = tiny_cfg(sharing=sharing)
    actual = sum(p.numel() for p in MaglevModel(cfg).parameters())
    assert actual == cfg.n_params_estimate()["total"]


# --------------------------------------------------------------------------
# Baselines and the ground-truth carrier gain
# --------------------------------------------------------------------------


def test_baselines_run_and_differ_in_reach() -> None:
    from rwc.model.baselines import full_attention_baseline, swa_baseline

    cfg = tiny_cfg()
    torch.manual_seed(0)
    swa = swa_baseline(cfg).double().eval()
    torch.manual_seed(0)
    full = full_attention_baseline(cfg).double().eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, SEQ_LEN))

    for model in (swa, full):
        logits, h = model(tokens)
        assert logits.shape == (2, SEQ_LEN, cfg.vocab_size)
        assert h.shape == (2, SEQ_LEN, cfg.d_model)

    # A far-back token must move the full-attention model and not the SWA one.
    far = 0
    query = SEQ_LEN - 1
    perturbed = tokens.clone()
    perturbed[:, far] = (perturbed[:, far] + 1) % cfg.vocab_size
    with torch.no_grad():
        assert torch.equal(swa(tokens)[0][:, query], swa(perturbed)[0][:, query])
        assert not torch.equal(full(tokens)[0][:, query], full(perturbed)[0][:, query])


def test_carrier_gain_makes_read_structure_known_by_construction() -> None:
    """Zero-gain dimensions must be provably unable to affect anything."""
    cfg = tiny_cfg(carrier_gain_seed=7, carrier_frac=0.25)
    torch.manual_seed(0)
    model = MaglevModel(cfg).double().eval()
    gain = model.injections[0].carrier_gain
    assert gain is not None
    assert torch.all(gain >= 0)
    assert int((gain > 0).sum()) == round(cfg.carrier_frac * cfg.d_model)
    for inj in model.injections:
        torch.testing.assert_close(inj.carrier_gain, gain)  # same structure everywhere

    tokens = torch.randint(0, cfg.vocab_size, (2, SEQ_LEN))
    memory = random_memory(model, tokens, torch.float64)
    dead = int((gain == 0).nonzero()[0])
    live = int((gain > 0).nonzero()[0])

    with torch.no_grad():
        base, _, _ = model.forward_decoder(tokens, MaglevModel.shift_memory(memory))
        for dim, expect_change in ((dead, False), (live, True)):
            bumped = memory.clone()
            bumped[:, :, dim] += 10.0
            out, _, _ = model.forward_decoder(tokens, MaglevModel.shift_memory(bumped))
            changed = (out - base).abs().max().item() > 0
            assert changed is expect_change, (
                f"dim {dim} (gain={gain[dim]:.3f}) changed={changed}"
            )
