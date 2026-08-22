"""The consistency loss, the propagation regulariser, and the diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from rwc.analysis import wilson_interval
from rwc.config import ConfigError, load_config
from rwc.evaluate import bits_per_byte
from rwc.losses import consistency_loss
from rwc.model.maglev import MaglevModel

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def diff_pair():
    torch.manual_seed(0)
    return torch.randn(2, 6, 16), torch.randn(2, 6, 16)


# --------------------------------------------------------------------------
# Weight scaling -- what makes the lambda sweep comparable
# --------------------------------------------------------------------------





def test_none_is_exactly_zero(diff_pair) -> None:
    m, m_prime = diff_pair
    assert float(consistency_loss(m, m_prime, kind="none")) == 0.0


def test_identical_memories_give_zero_penalty() -> None:
    m = torch.randn(2, 4, 8)
    cases = [
        ("uniform", {}),
    ]
    for kind, kw in cases:
        assert float(consistency_loss(m, m.clone(), kind=kind, **kw)) == pytest.approx(0.0)




def test_penalty_is_a_mean_of_per_position_norms(diff_pair) -> None:
    """Not a norm of the flattened tensor, which one position could dominate."""
    m, m_prime = diff_pair
    expected = (m - m_prime).pow(2).sum(-1).sqrt().mean() / math.sqrt(m.shape[-1])
    torch.testing.assert_close(consistency_loss(m, m_prime, kind="uniform"), expected)


def test_consistency_loss_is_differentiable(diff_pair) -> None:
    m, m_prime = diff_pair
    m = m.clone().requires_grad_(True)
    consistency_loss(m, m_prime, kind="uniform").backward()
    assert m.grad is not None and float(m.grad.abs().sum()) > 0


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        consistency_loss(torch.randn(2, 4, 8), torch.randn(2, 5, 8), kind="uniform")








def test_unknown_kinds_are_rejected(diff_pair) -> None:
    m, m_prime = diff_pair
    with pytest.raises(ValueError, match="unknown consistency kind"):
        consistency_loss(m, m_prime, kind="magic")








def _small_model() -> MaglevModel:
    cfg = load_config(
        REPO_ROOT / "configs" / "tiny_synth.yaml",
        overrides={
            "model.d_model": 32, "model.n_layers": 2, "model.n_heads": 4,
            "model.n_kv_heads": 2, "model.window": 8, "model.max_seq_len": 64,
            "model.prefiller_pattern": "SL", "model.decoder_pattern": "SS",
            "data.seq_len": 64, "data.distances": [4, 8, 16],
        },
        cuda_available=False,
    )
    torch.manual_seed(0)
    return MaglevModel(cfg.model).eval()








# --------------------------------------------------------------------------
# BPB and the baseline guard
# --------------------------------------------------------------------------


def test_bits_per_byte_matches_the_definition() -> None:
    assert bits_per_byte(math.log(2), 100, 100) == pytest.approx(1.0)
    assert bits_per_byte(2.0, 1_000_000, 4_000_000) == pytest.approx(
        2.0 / math.log(2) * 0.25
    )


@pytest.mark.parametrize("bad", [(1.0, 0, 10), (1.0, 10, 0)])
def test_bits_per_byte_rejects_empty_counts(bad) -> None:
    with pytest.raises(ValueError):
        bits_per_byte(*bad)


@pytest.mark.parametrize("kind", ["swa", "full"])
def test_baselines_reject_a_consistency_loss(kind: str) -> None:
    """No prefiller means no m' to be consistent with -- catch it at load."""
    with pytest.raises(ConfigError, match="no prefiller"):
        load_config(
            REPO_ROOT / "configs" / "tiny_synth.yaml",
            overrides={"model.kind": kind, "loss.consistency": "uniform", "loss.lam": 1.0},
            cuda_available=False,
        )
    load_config(
        REPO_ROOT / "configs" / "tiny_synth.yaml",
        overrides={"model.kind": kind},
        cuda_available=False,
    )






def test_wilson_interval_stays_inside_zero_one() -> None:
    """The normal approximation runs below zero at these accuracies; Wilson must not."""
    lo, hi = wilson_interval(0, 64)
    assert lo == 0.0 and 0.0 < hi < 0.10
    lo, hi = wilson_interval(64, 64)
    assert hi == 1.0 and lo > 0.9
    lo, hi = wilson_interval(4, 64)
    assert lo < 4 / 64 < hi


def test_wilson_narrows_with_more_samples() -> None:
    small = wilson_interval(10, 100)
    large = wilson_interval(100, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])










# --------------------------------------------------------------------------
# Stop-gradient on the distillation target
# --------------------------------------------------------------------------


def test_detach_target_blocks_gradient_to_the_prefiller_only(diff_pair) -> None:
    """With the pull symmetric, high lambda can be satisfied by degrading m'.

    Observed on the A100 headline batch: at lambda 0.1 the prefiller solved the
    memory-only buckets at 0.97 while the decoder inherited 0.16; at lambda 1.0
    both collapsed to 0.109. detach_target makes the prefiller a fixed teacher
    for this term so the decoder must chase it.
    """
    m, m_prime = diff_pair
    for detach, expect_target_grad in ((False, True), (True, False)):
        a = m.clone().requires_grad_(True)
        b = m_prime.clone().requires_grad_(True)
        consistency_loss(a, b, kind="uniform", detach_target=detach).backward()
        assert a.grad is not None and float(a.grad.abs().sum()) > 0
        got = b.grad is not None and float(b.grad.abs().sum()) > 0
        assert got is expect_target_grad


def test_detach_target_does_not_change_the_loss_value(diff_pair) -> None:
    """It is a gradient-routing change, not a different objective."""
    m, m_prime = diff_pair
    torch.testing.assert_close(
        consistency_loss(m, m_prime, kind="uniform", detach_target=True),
        consistency_loss(m, m_prime, kind="uniform", detach_target=False),
    )


def test_detach_target_is_part_of_the_config_identity() -> None:
    """Two arms differing only in this must be distinct manifest cells."""
    base = {"loss.consistency": "uniform", "loss.lam": 1.0}
    off = load_config(REPO_ROOT / "configs" / "tiny_synth.yaml",
                      overrides=base, cuda_available=False)
    on = load_config(REPO_ROOT / "configs" / "tiny_synth.yaml",
                     overrides={**base, "loss.detach_target": True}, cuda_available=False)
    assert off.config_hash != on.config_hash
    assert off.loss.detach_target is False and on.loss.detach_target is True




# --------------------------------------------------------------------------
# Recurrence diagnostics
# --------------------------------------------------------------------------

from rwc.analysis import (  # noqa: E402
    diagnose_recurrence,
    memory_ablation_effect,
    recurrence_gain,
)


def _tokens(model, n=2, t=32):
    return torch.randint(0, model.cfg.vocab_size, (n, t))


def test_severing_the_read_maps_zeroes_every_readout() -> None:
    """W_k_rec and W_v_rec are the only route from memory into the network.

    Zeroing them must drive all three diagnostics to their floor -- if any one
    still registers, it is reading memory through a path that should not exist.
    """
    model = _small_model().double()
    for inj in model.injections:
        torch.nn.init.zeros_(inj.W_k_rec.weight)
        torch.nn.init.zeros_(inj.W_v_rec.weight)
    d = diagnose_recurrence(model, _tokens(model))
    assert d.ablation_delta == pytest.approx(0.0, abs=1e-9)
    assert d.rho == pytest.approx(0.0, abs=1e-9)
    assert not d.is_read and not d.propagates
    assert "CHANNEL CLOSED" in d.verdict()


def test_an_intact_model_reads_and_propagates() -> None:
    model = _small_model().double()
    d = diagnose_recurrence(model, _tokens(model))
    assert d.is_read, "memory has no effect on the output at all"
    assert d.rho > 0 and d.sigma_max >= d.rho  # sigma_max always bounds rho


def test_read_without_write_is_distinguished_from_unused() -> None:
    """The failure this whole diagnostic exists to catch.

    Memory can be consumed at every position while nothing is carried across
    steps. Deployed accuracy cannot tell that apart from "memory is unused";
    ablation and rho together can, so the two must not move as one.
    """
    model = _small_model().double()
    tokens = _tokens(model)
    with torch.no_grad():
        mem_in = MaglevModel.shift_memory(model.forward_prefiller(tokens))
        # Keep the value path (memory still reaches the output) but cut the key
        # path, which is what carries memory into the next state's computation.
        for inj in model.injections:
            torch.nn.init.zeros_(inj.W_k_rec.weight)
    ablation = memory_ablation_effect(model, tokens, mem_in)
    assert ablation > 1e-3, "value path was severed too; the probe is not isolating"


def test_gain_is_invariant_to_batch_size() -> None:
    """Only the first sequence is used, so extra batch rows must not change it."""
    model = _small_model().double()
    torch.manual_seed(1)
    tokens = _tokens(model, n=4)
    with torch.no_grad():
        mem = MaglevModel.shift_memory(model.forward_prefiller(tokens))
    a = recurrence_gain(model, tokens, mem, positions=(8,))
    b = recurrence_gain(model, tokens[:1], mem[:1], positions=(8,))
    assert a[0] == pytest.approx(b[0], rel=1e-9)


def test_scaling_the_read_maps_scales_the_gain() -> None:
    """rho should track the strength of the recurrent path, not be an artifact."""
    model = _small_model().double()
    tokens = _tokens(model)
    with torch.no_grad():
        mem = MaglevModel.shift_memory(model.forward_prefiller(tokens))
    strong = recurrence_gain(model, tokens, mem, positions=(8,))[0]
    with torch.no_grad():
        for inj in model.injections:
            inj.W_k_rec.weight.mul_(0.05)
            inj.W_v_rec.weight.mul_(0.05)
    weak = recurrence_gain(model, tokens, mem, positions=(8,))[0]
    assert weak < strong, f"weakening the read maps did not lower rho ({weak} vs {strong})"


def test_diagnostics_do_not_leave_the_model_in_eval_mode() -> None:
    model = _small_model().double()
    model.train()
    diagnose_recurrence(model, _tokens(model))
    assert model.training, "diagnostic left the model in eval mode"


def test_a_constant_memory_trajectory_makes_consistency_exactly_zero() -> None:
    """The degenerate optimum, stated as a test.

    lam * ||m - m'|| is globally minimised by ANY constant trajectory, which
    carries no information. Nothing in the objective excludes it -- which is
    why trained models collapse into it, prefiller spread falling ~1800x below
    random init at lam >= 0.3.
    """
    const = torch.randn(1, 1, 16).expand(4, 12, 16).contiguous()
    assert float(consistency_loss(const, const.clone(), kind="uniform")) == 0.0
    # And it beats any informative trajectory that is not exactly matched.
    informative = torch.randn(4, 12, 16)
    assert float(consistency_loss(informative, const, kind="uniform")) > 0.0


def test_memory_information_detects_a_collapsed_trajectory() -> None:
    from rwc.analysis import memory_information

    model = _small_model().double()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 48))
    live = memory_information(model, tokens)
    assert live["spread"] > 1e-3, "a fresh model should have a varying trajectory"

    # Force the prefiller output to a constant by zeroing the final norm gain.
    with torch.no_grad():
        model.prefiller_norm.weight.zero_()
    dead = memory_information(model, tokens)
    assert dead["spread"] < live["spread"] / 100
