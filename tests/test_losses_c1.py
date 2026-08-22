"""Consistency losses and the C1 decorrelation curve (SPEC.md section 5, M5)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from rwc.analysis import (
    _slope,
    c1_curve,
    consistency_error_by_dim,
    effective_rank,
    full_sequence_targets,
    memory_spectrum,
)
from rwc.config import ConfigError, load_config
from rwc.evaluate import bits_per_byte
from rwc.losses import consistency_loss, influence_mask
from rwc.model.maglev import MaglevModel

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def diff_pair():
    torch.manual_seed(0)
    return torch.randn(2, 6, 16), torch.randn(2, 6, 16)


# --------------------------------------------------------------------------
# Weight scaling -- what makes the lambda sweep comparable
# --------------------------------------------------------------------------


def test_rwc_with_flat_weights_reproduces_uniform_exactly(diff_pair) -> None:
    """RWC must be a strict generalisation of Maglev's uniform L2.

    Without the mean-1 rescaling, sum-to-1 weights would make the RWC penalty
    ~d times weaker at the same lambda, and the C3 comparison would be measuring
    penalty strength rather than allocation.
    """
    m, m_prime = diff_pair
    d = m.shape[-1]
    flat = torch.full((d,), 1.0 / d)
    torch.testing.assert_close(
        consistency_loss(m, m_prime, kind="rwc", weights=flat),
        consistency_loss(m, m_prime, kind="uniform"),
    )


def test_rwc_reweights_toward_high_influence_dims(diff_pair) -> None:
    m, m_prime = diff_pair
    peaked = torch.zeros(m.shape[-1])
    peaked[0] = 1.0
    assert float(consistency_loss(m, m_prime, kind="rwc", weights=peaked)) != pytest.approx(
        float(consistency_loss(m, m_prime, kind="uniform"))
    )


def test_full_mask_reduces_to_uniform(diff_pair) -> None:
    """mask_topk at k=100 must be exactly the uniform loss."""
    m, m_prime = diff_pair
    d = m.shape[-1]
    mask = influence_mask(torch.rand(d), "mask_topk", 100.0)
    assert int(mask.sum()) == d
    torch.testing.assert_close(
        consistency_loss(m, m_prime, kind="mask_topk", mask=mask),
        consistency_loss(m, m_prime, kind="uniform"),
    )


def test_none_is_exactly_zero(diff_pair) -> None:
    m, m_prime = diff_pair
    assert float(consistency_loss(m, m_prime, kind="none")) == 0.0


def test_identical_memories_give_zero_penalty() -> None:
    m = torch.randn(2, 4, 8)
    cases = [
        ("uniform", {}),
        ("rwc", {"weights": torch.rand(8)}),
        ("mask_topk", {"mask": influence_mask(torch.rand(8), "mask_topk", 50.0)}),
    ]
    for kind, kw in cases:
        assert float(consistency_loss(m, m.clone(), kind=kind, **kw)) == pytest.approx(0.0)


def test_weights_must_be_detached(diff_pair) -> None:
    """They are loss weights; gradient must not flow into the influence estimate."""
    m, m_prime = diff_pair
    w = torch.rand(m.shape[-1], requires_grad=True)
    with pytest.raises(ValueError, match="detached"):
        consistency_loss(m, m_prime, kind="rwc", weights=w)
    with pytest.raises(ValueError, match="detached"):
        consistency_loss(m, m_prime, kind="mask_topk", mask=w)


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


# --------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k_pct, expect", [(10.0, 3), (25.0, 8), (50.0, 16), (100.0, 32)])
def test_mask_selects_the_right_count(k_pct: float, expect: int) -> None:
    assert int(influence_mask(torch.rand(32), "mask_topk", k_pct).sum()) == expect


def test_top_and_bottom_masks_are_disjoint_below_half() -> None:
    w = torch.rand(64)
    top = influence_mask(w, "mask_topk", 25.0)
    bottom = influence_mask(w, "mask_bottomk", 25.0)
    assert float((top * bottom).sum()) == 0.0
    assert float(w[top.bool()].min()) > float(w[bottom.bool()].max())


def test_random_mask_is_reproducible_from_a_generator() -> None:
    w = torch.rand(64)
    a = influence_mask(w, "mask_randomk", 25.0, generator=torch.Generator().manual_seed(3))
    b = influence_mask(w, "mask_randomk", 25.0, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(a, b)


def test_random_mask_ignores_influence_ordering() -> None:
    """The falsification control must not secretly track influence."""
    w = torch.arange(64, dtype=torch.float)
    rand = influence_mask(w, "mask_randomk", 25.0, generator=torch.Generator().manual_seed(1))
    assert not torch.equal(rand, influence_mask(w, "mask_topk", 25.0))


@pytest.mark.parametrize("k_pct", [0.0, -5.0, 101.0])
def test_mask_rejects_invalid_k(k_pct: float) -> None:
    with pytest.raises(ValueError, match="k_pct"):
        influence_mask(torch.rand(8), "mask_topk", k_pct)


def test_unknown_kinds_are_rejected(diff_pair) -> None:
    m, m_prime = diff_pair
    with pytest.raises(ValueError, match="unknown consistency kind"):
        consistency_loss(m, m_prime, kind="magic")
    with pytest.raises(ValueError, match="unknown mask kind"):
        influence_mask(torch.rand(8), "mask_middlek", 25.0)


# --------------------------------------------------------------------------
# Effective rank
# --------------------------------------------------------------------------


def test_effective_rank_of_isotropic_noise_is_near_full() -> None:
    torch.manual_seed(0)
    assert effective_rank(torch.randn(4096, 32)) > 0.9 * 32


def test_effective_rank_of_a_rank_one_matrix_is_one() -> None:
    m = torch.randn(256, 1) @ torch.randn(1, 16)
    assert effective_rank(m, center=False) == pytest.approx(1.0, abs=1e-3)


def test_centering_removes_a_shared_offset() -> None:
    """A common bias vector is not capacity in use, and must not read as rank 1."""
    torch.manual_seed(0)
    m = torch.randn(1024, 16) + 50.0 * torch.randn(1, 16)
    assert effective_rank(m, center=False) < 2.0
    assert effective_rank(m, center=True) > 10.0
    spec = memory_spectrum(m)
    assert spec["effective_rank"] > spec["effective_rank_uncentred"]
    assert spec["n_dims"] == 16


# --------------------------------------------------------------------------
# C1
# --------------------------------------------------------------------------


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


def test_c1_curve_reports_one_entry_per_horizon() -> None:
    model = _small_model()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 64))
    out = c1_curve(model, tokens, horizons=[4, 8, 16], n_anchors=3)
    assert [p["horizon"] for p in out["per_horizon"]] == [4, 8, 16]
    assert 0 < len(out["anchors"]) <= 3
    assert all(-1.0 <= p["rho_mean"] <= 1.0 for p in out["per_horizon"])
    assert "effective_rank" in out["memory_spectrum"]


def test_c1_uses_one_anchor_set_across_all_horizons() -> None:
    """Different anchors per H would make the curve incomparable across H."""
    model = _small_model()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 64))
    out = c1_curve(model, tokens, horizons=[4, 16], n_anchors=3)
    counts = {p["n_anchors"] for p in out["per_horizon"]}
    assert len(counts) == 1


def test_c1_rejects_answer_only_targets() -> None:
    """The failure this guards against produced a perfectly flat curve.

    With one scored position, every horizon that reaches it scores the identical
    set, so grad_influence returns bit-identical values for H=8 and H=128 and
    the curve comes out flat for a reason unrelated to the hypothesis.
    """
    model = _small_model()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 64))
    answer_only = torch.full_like(tokens, -100)
    answer_only[:, 60] = tokens[:, 61]
    with pytest.raises(ValueError, match="scored positions"):
        c1_curve(model, tokens, answer_only, horizons=[4, 8, 16])


def test_c1_horizons_actually_differ_under_full_targets() -> None:
    """Guard against the flat-curve bug returning."""
    model = _small_model()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 64))
    a, b = c1_curve(model, tokens, horizons=[4, 32], n_anchors=3)["per_horizon"]
    assert a["rho_mean"] != b["rho_mean"], "H had no effect on the estimate"


def test_full_sequence_targets_are_next_token() -> None:
    tokens = torch.randint(0, 20, (2, 8))
    targets = full_sequence_targets(tokens)
    torch.testing.assert_close(targets[:, :-1], tokens[:, 1:])
    assert torch.all(targets[:, -1] == -100)


def test_consistency_error_is_kept_per_position() -> None:
    m, m_prime = torch.randn(3, 5, 7), torch.randn(3, 5, 7)
    err = consistency_error_by_dim(m, m_prime)
    assert err.shape == (5, 7)
    torch.testing.assert_close(err, (m - m_prime).abs().mean(dim=0))


def test_slope_sign_is_the_c1_verdict() -> None:
    assert _slope([1, 2, 3], [0.9, 0.5, 0.1]) < 0  # decays -> C1 holds
    assert _slope([1, 2, 3], [0.1, 0.5, 0.9]) > 0
    assert _slope([1, 2, 3], [0.5, 0.5, 0.5]) == pytest.approx(0.0)


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


# --------------------------------------------------------------------------
# C3 statistics
# --------------------------------------------------------------------------

from rwc.analysis import paired_slope_test, wilson_interval  # noqa: E402

DIST = [16, 24, 32, 48, 64, 96]


def _arms(n_seeds, control_fn, treat_fn):
    return (
        {s: [control_fn(s, i) for i in range(len(DIST))] for s in range(n_seeds)},
        {s: [treat_fn(s, i) for i in range(len(DIST))] for s in range(n_seeds)},
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


def test_identical_arms_give_zero_slope_and_p_one() -> None:
    c, t = _arms(8, lambda s, i: 0.1, lambda s, i: 0.1)
    r = paired_slope_test(DIST, c, t, n_permutations=2000)
    assert r.slope_mean == pytest.approx(0.0)
    assert r.p_value == pytest.approx(1.0)
    assert not r.significant


def test_a_flat_advantage_is_not_a_growing_one() -> None:
    """The distinction C3 turns on: a constant gain means RWC is a regulariser."""
    c, t = _arms(8, lambda s, i: 0.1, lambda s, i: 0.2)
    r = paired_slope_test(DIST, c, t, n_permutations=2000)
    assert r.mean_delta == pytest.approx(0.1)  # there IS an advantage
    assert r.slope_mean == pytest.approx(0.0)  # but it does not grow
    assert not r.significant
    assert "NOT SIGNIFICANT" in r.verdict()


def test_a_growing_advantage_is_detected() -> None:
    c, t = _arms(8, lambda s, i: 0.1, lambda s, i: 0.1 + 0.05 * i)
    r = paired_slope_test(DIST, c, t, n_permutations=4000)
    assert r.slope_mean > 0
    assert r.significant
    assert r.verdict() == "GROWS with distance"
    assert r.ci_low > 0


def test_a_shrinking_advantage_is_detected() -> None:
    c, t = _arms(8, lambda s, i: 0.1, lambda s, i: 0.4 - 0.05 * i)
    r = paired_slope_test(DIST, c, t, n_permutations=4000)
    assert r.slope_mean < 0
    assert r.verdict() == "SHRINKS with distance"


def test_five_seeds_cannot_reach_significance() -> None:
    """The permutation floor is 2/2**n_seeds; at 5 seeds that is above 0.05.

    Without surfacing this, an underpowered design reports p=0.0625 and reads
    as a null result when it is really 'this test could never have said yes'.
    """
    c, t = _arms(5, lambda s, i: 0.1, lambda s, i: 0.1 + 0.5 * i)
    r = paired_slope_test(DIST, c, t, n_permutations=4000)
    assert r.underpowered
    assert r.min_achievable_p == pytest.approx(0.0625)
    assert not r.significant
    assert "UNDERPOWERED" in r.verdict()

    c8, t8 = _arms(8, lambda s, i: 0.1, lambda s, i: 0.1 + 0.5 * i)
    r8 = paired_slope_test(DIST, c8, t8, n_permutations=4000)
    assert not r8.underpowered and r8.significant


def test_seed_noise_does_not_manufacture_a_slope() -> None:
    """Pure noise, paired by seed, must not come out significant."""
    import random

    rng = random.Random(0)
    c, t = _arms(8, lambda s, i: rng.random() * 0.2, lambda s, i: rng.random() * 0.2)
    r = paired_slope_test(DIST, c, t, n_permutations=4000)
    assert not r.significant, f"noise produced p={r.p_value:.4f}"


def test_pairing_is_by_seed_and_mismatches_are_rejected() -> None:
    c, t = _arms(4, lambda s, i: 0.1, lambda s, i: 0.2)
    del t[3]
    with pytest.raises(ValueError, match="same seeds"):
        paired_slope_test(DIST, c, t, n_permutations=100)


def test_wrong_number_of_distances_is_rejected() -> None:
    c, t = _arms(4, lambda s, i: 0.1, lambda s, i: 0.2)
    t[0] = [0.1, 0.2]
    with pytest.raises(ValueError, match="expected"):
        paired_slope_test(DIST, c, t, n_permutations=100)


def test_log_axis_is_the_default_and_changes_the_slope() -> None:
    """The probe grid is geometric; a linear fit lets the last bucket dominate."""
    c, t = _arms(8, lambda s, i: 0.1, lambda s, i: 0.1 + 0.05 * i)
    log = paired_slope_test(DIST, c, t, n_permutations=500)
    lin = paired_slope_test(DIST, c, t, n_permutations=500, log_x=False)
    assert log.slope_mean != pytest.approx(lin.slope_mean)
