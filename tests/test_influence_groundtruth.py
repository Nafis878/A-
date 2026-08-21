"""Read-influence estimators against a read structure known by construction.

SPEC.md section 4: if these fail, RWC is weighting noise and the whole paper
collapses. The actual correlations are printed, not just a pass/fail.

**How the ground truth is known by construction.** ``W_k_rec`` and ``W_v_rec``
are the *only* route from memory into the network -- the gates read the residual
stream, not memory -- so scaling the memory vector by a fixed, known gain
``g`` before those projections makes each dimension's readability an exact
property of the model rather than a guess. A dimension with ``g[i] = 0`` is
provably unreadable, which ``test_carrier_gain_makes_read_structure_known_by_
construction`` verifies directly. ``g`` is graded (log-uniform over two decades)
rather than binary, so the ground-truth ranking is not almost-all ties and
Spearman retains the power to detect that an estimator orders the *readable*
dimensions correctly, not merely that it finds the dead ones.

The task is delayed-recall past the stacked local reach, so the model has no
route to the answer except through memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pytest
import torch

from rwc.config import Config, load_config
from rwc.influence import (
    InfluenceEMA,
    default_anchors,
    grad_influence,
    normalise,
    saliency_influence,
    scored_positions,
    spearman,
)
from rwc.model.maglev import MaglevModel
from rwc.train import Trainer

REPO_ROOT = Path(__file__).resolve().parents[1]

# SPEC.md section 4 thresholds. Not to be relaxed to make the suite green.
RHO_A_VS_TRUTH = 0.7
RHO_B_VS_A = 0.6

TRAIN_STEPS = 120
HORIZON_H = 24


def carrier_config(tmp_path: Path, **extra) -> Config:
    """Small enough for a CPU gate, structured enough for the claim to mean something."""
    overrides = {
        "model.d_model": 64,
        "model.n_layers": 2,
        "model.n_heads": 4,
        "model.n_kv_heads": 2,
        "model.window": 8,
        "model.max_seq_len": 96,
        "model.prefiller_pattern": "SL",
        "model.decoder_pattern": "SS",
        "model.carrier_gain_seed": 11,
        "model.carrier_frac": 0.25,
        "data.task": "carrier_copy",
        "data.seq_len": 96,
        # Every distance past the stacked local reach L*(W-1) = 2*7 = 14, so the
        # answer is unreachable without the recurrent channel.
        "data.distances": [16, 24, 32, 48],
        "data.loss_on_answer_only": True,
        "data.n_distractors": 2,
        "optim.total_steps": TRAIN_STEPS,
        "optim.warmup": 10,
        "optim.lr": 3e-3,
        "optim.global_tokens_per_step": 8 * 96,
        "optim.micro_batch": 8,
        "run.out_dir": str(tmp_path),
        "run.ckpt_every": 10_000,
        "run.log_every": 10_000,
        "run.seed": 0,
    }
    overrides.update(extra)
    return load_config(
        REPO_ROOT / "configs" / "tiny_synth.yaml",
        overrides=overrides,
        cuda_available=False,
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> Tuple[MaglevModel, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train briefly, then hand back the model, a batch, and the true gains."""
    tmp_path = tmp_path_factory.mktemp("carrier")
    cfg = carrier_config(tmp_path)
    trainer = Trainer(cfg, device=torch.device("cpu"))
    trainer.run(quiet=True)

    model = trainer.model
    batch = trainer.data.batch(step=TRAIN_STEPS + 1, batch_size=8)
    gain = model.injections[0].carrier_gain.clone()
    assert gain is not None and float(gain.sum()) > 0
    return model, batch.tokens, batch.targets, gain


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def test_estimator_a_recovers_the_true_read_structure(trained) -> None:
    """SPEC.md section 4: Spearman rho > 0.7 against the known gains."""
    model, tokens, targets, gain = trained
    w = grad_influence(model, tokens, targets, horizon_H=HORIZON_H).mean(dim=0)

    rho = spearman(w, gain)
    live = gain > 0
    rho_live = spearman(w[live], gain[live])
    prec = _precision_at_k(w, gain)

    print(
        f"\n[influence] Estimator A vs ground truth"
        f"\n    Spearman rho (all {len(gain)} dims) = {rho:+.4f}   threshold {RHO_A_VS_TRUTH}"
        f"\n    Spearman rho (live carriers only)  = {rho_live:+.4f}"
        f"\n    precision@k (k = #carriers)        = {prec:.3f}"
    )
    assert rho > RHO_A_VS_TRUTH, f"Estimator A recovered the read structure at rho={rho:.4f}"


def test_estimator_b_agrees_with_estimator_a(trained) -> None:
    """SPEC.md section 4: Spearman rho > 0.6 between the cheap and exact estimators."""
    model, tokens, targets, gain = trained
    a = grad_influence(model, tokens, targets, horizon_H=HORIZON_H).mean(dim=0)
    b = saliency_influence(model, tokens).mean(dim=(0, 1))

    rho = spearman(b, a)
    live = gain > 0
    rho_live = spearman(b[live], a[live])
    rho_b_truth = spearman(b, gain)

    print(
        f"\n[influence] Estimator B vs Estimator A"
        f"\n    Spearman rho (all dims)           = {rho:+.4f}   threshold {RHO_B_VS_A}"
        f"\n    Spearman rho (live carriers only) = {rho_live:+.4f}"
        f"\n    B vs ground truth (reference)     = {rho_b_truth:+.4f}"
    )
    assert rho > RHO_B_VS_A, f"Estimator B agrees with A only at rho={rho:.4f}"


def _precision_at_k(w: torch.Tensor, gain: torch.Tensor) -> float:
    k = int((gain > 0).sum())
    top = set(torch.topk(w, k).indices.tolist())
    true = set(torch.topk(gain, k).indices.tolist())
    return len(top & true) / k


# --------------------------------------------------------------------------
# Mechanics the gate depends on
# --------------------------------------------------------------------------


def test_dead_dimensions_get_no_influence(trained) -> None:
    """A provably unreadable dimension must score ~0 under both estimators."""
    model, tokens, targets, gain = trained
    dead = gain == 0
    assert int(dead.sum()) > 0

    a = grad_influence(model, tokens, targets, horizon_H=HORIZON_H).mean(dim=0)
    b = saliency_influence(model, tokens).mean(dim=(0, 1))
    for name, w in (("A", a), ("B", b)):
        assert float(w[dead].max()) <= 1e-9, f"Estimator {name} scored a dead dimension"
        assert float(w[~dead].max()) > 0, f"Estimator {name} is identically zero"


def test_estimators_are_normalised_per_position(trained) -> None:
    """SPEC.md section 4 requires the weights to sum to 1 over d."""
    model, tokens, targets, _ = trained
    a = grad_influence(model, tokens, targets, horizon_H=HORIZON_H)
    torch.testing.assert_close(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5, rtol=0)

    b = saliency_influence(model, tokens)
    # All but the final memory position, which is exactly zero because nothing
    # ever reads m'_{T-1}.
    head = b[:, :-1].sum(-1)
    torch.testing.assert_close(head, torch.ones_like(head), atol=1e-5, rtol=0)
    assert float(b[:, -1].abs().max()) == 0.0


def test_estimators_carry_no_gradient(trained) -> None:
    """Used as loss weights, so they must be detached (SPEC.md section 4)."""
    model, tokens, targets, _ = trained
    assert not grad_influence(model, tokens, targets, horizon_H=HORIZON_H).requires_grad
    assert not saliency_influence(model, tokens).requires_grad


def test_grad_influence_leaves_no_gradient_on_the_model(trained) -> None:
    """The reference estimator must not pollute the training gradient."""
    model, tokens, targets, _ = trained
    model.zero_grad(set_to_none=True)
    grad_influence(model, tokens, targets, horizon_H=HORIZON_H)
    assert all(p.grad is None for p in model.parameters())


def test_last_memory_position_is_never_read(trained) -> None:
    """m'_{T-1} is consumed at position T, which does not exist."""
    model, tokens, _, _ = trained
    b = saliency_influence(model, tokens, normalise_out=False)
    assert float(b[:, -1].abs().max()) == 0.0


def test_detach_local_changes_the_estimate(trained) -> None:
    """Guard against the flag being silently inert."""
    model, tokens, targets, _ = trained
    on = grad_influence(model, tokens, targets, horizon_H=HORIZON_H, detach_local=True)
    off = grad_influence(model, tokens, targets, horizon_H=HORIZON_H, detach_local=False)
    assert not torch.allclose(on, off), "detach_local had no effect on the gradient"
    # Both should still find the same broad structure.
    print(f"\n[influence] detach_local on vs off: rho = "
          f"{spearman(on.mean(0), off.mean(0)):+.4f}")


def test_horizon_matters(trained) -> None:
    """Influence is defined over j+1..j+H; H must actually enter the answer.

    Run against full-sequence targets: under the answer-only loss used for the
    gate, widening H past the single scored position adds nothing to score, so
    the estimate correctly does *not* change and the test would be vacuous.
    """
    model, tokens, _, _ = trained
    full = torch.full_like(tokens, -100)
    full[:, :-1] = tokens[:, 1:]
    short = grad_influence(model, tokens, full, horizon_H=4, anchors=[20])
    long_ = grad_influence(model, tokens, full, horizon_H=48, anchors=[20])
    assert not torch.allclose(short, long_)
    rho = spearman(short[0], long_[0])
    print(f"\n[influence] horizon 4 vs 48: rho = {rho:+.4f}")


def test_anchors_without_a_scored_position_fail_loudly(trained) -> None:
    """An all-zero influence vector reads as 'memory is never used'. It must not
    be possible to get one by accident."""
    model, tokens, targets, _ = trained
    with pytest.raises(ValueError, match="identically"):
        grad_influence(model, tokens, targets, horizon_H=2, anchors=[0])


def test_useless_anchors_are_dropped_not_averaged_in(trained) -> None:
    """One unscored anchor must not turn the whole estimate into NaN."""
    model, tokens, targets, _ = trained
    answer = int(scored_positions(targets)[-1])
    mixed = grad_influence(
        model, tokens, targets, horizon_H=HORIZON_H, anchors=[0, answer - 2]
    )
    assert torch.isfinite(mixed).all()
    assert mixed.shape[0] == 1, "the unscored anchor should have been dropped"


def test_default_anchors_cover_the_scored_positions(trained) -> None:
    _, _, targets, _ = trained
    scored = set(scored_positions(targets).tolist())
    anchors = default_anchors(targets, HORIZON_H)
    assert anchors
    for j in anchors:
        assert any(j < s <= j + HORIZON_H for s in scored)


# --------------------------------------------------------------------------
# EMA smoothing
# --------------------------------------------------------------------------


def test_ema_converges_and_stays_normalised(tmp_path: Path) -> None:
    cfg = carrier_config(tmp_path)
    ema = InfluenceEMA(cfg.loss.influence, cfg.model.d_model, torch.device("cpu"))
    assert ema.stats()["w_entropy_ratio"] == pytest.approx(1.0, abs=1e-6)

    target = normalise(torch.rand(cfg.model.d_model))
    for _ in range(200):
        ema.update(target.clone())
    torch.testing.assert_close(ema.value, target, atol=1e-4, rtol=0)
    assert float(ema.value.sum()) == pytest.approx(1.0, abs=1e-5)


def test_ema_stats_flag_both_failure_modes(tmp_path: Path) -> None:
    """Uniform collapse and spike collapse must both be visible in the logs."""
    cfg = carrier_config(tmp_path)
    d = cfg.model.d_model

    uniform = InfluenceEMA(cfg.loss.influence, d, torch.device("cpu"))
    for _ in range(200):
        uniform.update(torch.full((d,), 1.0 / d))
    assert uniform.stats()["w_entropy_ratio"] > 0.99
    assert uniform.stats()["w_max_over_uniform"] == pytest.approx(1.0, abs=1e-3)

    spike = InfluenceEMA(cfg.loss.influence, d, torch.device("cpu"))
    one_hot = torch.zeros(d)
    one_hot[3] = 1.0
    for _ in range(400):
        spike.update(one_hot.clone())
    assert spike.stats()["w_entropy_ratio"] < 0.05
    assert spike.stats()["w_max_over_uniform"] > 0.5 * d


def test_ema_survives_a_checkpoint_roundtrip(tmp_path: Path) -> None:
    cfg = carrier_config(tmp_path)
    a = InfluenceEMA(cfg.loss.influence, cfg.model.d_model, torch.device("cpu"))
    a.update(normalise(torch.rand(cfg.model.d_model)))
    b = InfluenceEMA(cfg.loss.influence, cfg.model.d_model, torch.device("cpu"))
    b.load_state_dict(a.state_dict())
    torch.testing.assert_close(a.value, b.value)
    assert b.n_updates == a.n_updates
