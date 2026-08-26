"""The char-LM loader must be a pure function of its index.

`Trainer._micro_batch` calls `data.batch(step * accum + micro, micro_batch)` and
saves no data position, so an exact resume depends entirely on that call being
deterministic in the index. If it ever acquires hidden state, resumed LM runs
drift silently -- the same class of bug the synthetic path is guarded against in
test_resume.py.

These build their own corpus in tmp_path rather than reading lm_data/, so they
run without the 95 MB text8 download.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from rwc.config import ConfigError, load_config
from rwc.data.lm import CharShardLoader, load_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
ALPHABET = " abcdefghijklmnopqrstuvwxyz"


def make_corpus(root: Path, n: int = 40_000) -> Path:
    """A deterministic pseudo-text over text8's alphabet."""
    root.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, len(ALPHABET), (n,), generator=g).tolist()
    (root / "text8").write_text("".join(ALPHABET[i] for i in idx), encoding="ascii")
    idx_v = torch.randint(0, len(ALPHABET), (n // 4,), generator=g).tolist()
    (root / "text8.valid").write_text(
        "".join(ALPHABET[i] for i in idx_v), encoding="ascii"
    )
    return root


def cfg_for(root: Path, seq_len: int = 64):
    return load_config(
        REPO_ROOT / "configs" / "tiny_synth.yaml",
        overrides={
            "data.kind": "lm", "data.shard_dir": str(root),
            "data.seq_len": seq_len, "data.val_tokens": 5_000,
            "model.max_seq_len": seq_len, "model.vocab_size": 27,
            "model.d_model": 32, "model.n_layers": 2, "model.n_heads": 4,
            "model.n_kv_heads": 2, "model.window": 4,
            "model.prefiller_pattern": "SL", "model.decoder_pattern": "SS",
        },
        cuda_available=False,
    )


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    return make_corpus(tmp_path / "lm")


# --------------------------------------------------------------------------
# Purity in the index -- the property resume depends on
# --------------------------------------------------------------------------


def test_batch_is_deterministic_in_the_index(corpus: Path) -> None:
    cfg = cfg_for(corpus)
    a = CharShardLoader(cfg.data, seed=3)
    assert torch.equal(a.batch(11, 4).tokens, a.batch(11, 4).tokens)


def test_batch_is_identical_across_instances(corpus: Path) -> None:
    """A resumed process builds a NEW loader; it must see the same data."""
    cfg = cfg_for(corpus)
    a = CharShardLoader(cfg.data, seed=3)
    b = CharShardLoader(cfg.data, seed=3)
    for i in (0, 1, 7, 128):
        assert torch.equal(a.batch(i, 4).tokens, b.batch(i, 4).tokens), f"index {i}"


def test_different_indices_give_different_data(corpus: Path) -> None:
    cfg = cfg_for(corpus)
    a = CharShardLoader(cfg.data, seed=3)
    assert not torch.equal(a.batch(0, 4).tokens, a.batch(1, 4).tokens)


def test_seed_changes_the_permutation(corpus: Path) -> None:
    cfg = cfg_for(corpus)
    assert not torch.equal(
        CharShardLoader(cfg.data, seed=0).batch(0, 4).tokens,
        CharShardLoader(cfg.data, seed=1).batch(0, 4).tokens,
    )


# --------------------------------------------------------------------------
# Shape and content
# --------------------------------------------------------------------------


def test_targets_are_the_next_character(corpus: Path) -> None:
    """Off-by-one here would make BPB meaningless while still looking sane."""
    cfg = cfg_for(corpus)
    b = CharShardLoader(cfg.data, seed=0).batch(5, 3)
    assert torch.equal(b.targets[:, :-1], b.tokens[:, 1:])
    assert b.targets.shape == b.tokens.shape == (3, cfg.data.seq_len)


def test_ids_stay_inside_the_vocabulary(corpus: Path) -> None:
    cfg = cfg_for(corpus)
    loader = CharShardLoader(cfg.data, seed=0)
    b = loader.batch(0, 8)
    assert int(b.tokens.min()) >= 0
    assert int(b.tokens.max()) < loader.vocab_size <= cfg.model.vocab_size


def test_every_position_is_scored(corpus: Path) -> None:
    """LM training is full-sequence CE, unlike the answer-only synthetic runs."""
    cfg = cfg_for(corpus)
    b = CharShardLoader(cfg.data, seed=0).batch(0, 2)
    assert bool(b.loss_mask.all())


def test_train_and_validation_do_not_overlap(corpus: Path) -> None:
    """BPB on trained-on text would make the memory comparison meaningless."""
    cfg = cfg_for(corpus)
    tr = CharShardLoader(cfg.data, seed=0, split="train")
    va = CharShardLoader(cfg.data, seed=0, split="valid")
    train_text = bytes(tr.ids.tolist())
    val_text = bytes(va.ids[:200].tolist())
    assert val_text not in train_text


def test_tail_split_when_no_validation_file(tmp_path: Path) -> None:
    root = tmp_path / "solo"
    root.mkdir()
    (root / "text8").write_text("abc def " * 5000, encoding="ascii")
    train, valid, source = load_corpus(str(root))
    assert "tail split" in source
    assert len(valid) > 0 and len(train) > 0
    # Truncated so the two cannot overlap.
    assert len(train) + len(valid) == len("abc def " * 5000)


def test_missing_corpus_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no corpus"):
        load_corpus(str(tmp_path / "nothing"))


def test_corpus_too_short_for_seq_len(corpus: Path) -> None:
    cfg = cfg_for(corpus, seq_len=64)
    cfg.data.seq_len = 10**9
    with pytest.raises(ConfigError, match="too few"):
        CharShardLoader(cfg.data, seed=0)


# --------------------------------------------------------------------------
# BPB
# --------------------------------------------------------------------------


def test_untrained_model_scores_the_uniform_baseline(corpus: Path) -> None:
    """The gate that catches a miscomputed bits_per_byte.

    An untrained model over a 27-symbol alphabet must sit near log2(27)=4.755
    bits per character. This caught a real error: with the default vocab_size of
    64 the model spreads mass over 37 ids that can never occur and scores
    log2(64)=6.0, which would have silently inflated every LM number.
    """
    from rwc.evaluate import lm_memory_benefit
    from rwc.model.maglev import MaglevModel

    cfg = cfg_for(corpus)
    val = CharShardLoader(cfg.data, seed=0, split="valid")
    model = MaglevModel(cfg.model).eval()
    out = lm_memory_benefit(model, val, n_batches=2, batch_size=4,
                            buckets=[0, 8, 32, 64])
    assert out["bpb_live"] == pytest.approx(math.log2(27), abs=0.4)
    assert out["bpb_dead"] == pytest.approx(math.log2(27), abs=0.4)
    assert out["n_positions"] == cfg.data.seq_len
    assert [b["lo"] for b in out["by_position"]] == [0, 8, 32]


def test_memory_ablation_is_the_local_model(corpus: Path) -> None:
    """Zeroing memory must actually change the computation, not silently no-op."""
    from rwc.model.maglev import MaglevModel

    cfg = cfg_for(corpus)
    val = CharShardLoader(cfg.data, seed=0, split="valid")
    model = MaglevModel(cfg.model).eval()
    tokens = val.batch(0, 2).tokens
    with torch.no_grad():
        live = model.forward_recurrent(tokens).logits
        zeros = torch.zeros(*tokens.shape, cfg.model.d_model)
        dead = model.forward_recurrent(tokens, mem_in_override=zeros).logits
    assert not torch.allclose(live, dead)


# --------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------


def test_lm_resume_reproduces_the_loss_trace(corpus: Path, tmp_path: Path) -> None:
    """Same gate as test_resume.py, on the LM path."""
    from rwc.train import Trainer

    def build(run: str):
        cfg = cfg_for(corpus)
        cfg.optim.total_steps = 8
        cfg.optim.warmup = 2
        cfg.optim.micro_batch = 2
        cfg.optim.global_tokens_per_step = 2 * cfg.data.seq_len
        cfg.optim.train_mode = "bptt"
        cfg.run.out_dir = str(tmp_path / run)
        cfg.run.ckpt_every = 4
        cfg.run.log_every = 1000
        cfg.run.seed = 1
        return Trainer(cfg, device=torch.device("cpu"))

    clean = build("clean")
    clean.run(quiet=True)
    expected = [m.ce for m in clean.history]

    part = build("resumed")
    part.run(max_steps=4, quiet=True)
    first = [m.ce for m in part.history]
    del part

    revived = build("resumed")
    assert revived.maybe_resume()
    assert revived.step == 4
    revived.run(quiet=True)

    combined = first + [m.ce for m in revived.history]
    assert len(combined) == len(expected) == 8
    worst = max(abs(a - b) for a, b in zip(expected, combined))
    assert worst == 0.0, f"LM resume diverged by {worst:.3e}"
