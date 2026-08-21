"""Shape and configuration contracts.

M1 covers the config layer: the shipped YAMLs load, derived geometry is
consistent, every validation rule actually rejects, and the config hash is
stable in the right way. M2 extends this file with real module shape tests.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from rwc.config import (
    Config,
    ConfigError,
    DataConfig,
    InfluenceConfig,
    LossConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    config_from_dict,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

# (config file, non-embedding parameter band). Bands are deliberately generous:
# the SPEC states targets as "~5M" / "~25M" / "~60M", and the point of the test
# is to catch a geometry change that blows the Colab budget, not to pin a number.
PARAM_BANDS = {
    "tiny_synth.yaml": (3_000_000, 7_000_000),
    "small_lm.yaml": (20_000_000, 32_000_000),
    "medium_lm.yaml": (50_000_000, 70_000_000),
}


def _cuda_free(raw: Dict[str, Any]) -> Config:
    """Build a config as if no GPU were present, so tests are host-independent."""
    return config_from_dict(raw, cuda_available=False)


def _base_dict() -> Dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / "tiny_synth.yaml").read_text(encoding="utf-8"))


def _mutated(**dotted: Any) -> Dict[str, Any]:
    raw = _base_dict()
    for key, value in dotted.items():
        section, _, leaf = key.partition("__")
        node = raw[section]
        while "__" in leaf:
            head, _, leaf = leaf.partition("__")
            node = node[head]
        node[leaf] = value
    return raw


# --------------------------------------------------------------------------
# The shipped configs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PARAM_BANDS))
def test_shipped_config_loads_and_validates(name: str) -> None:
    cfg = load_config(CONFIG_DIR / name, cuda_available=False)
    assert cfg.run.name == name.replace(".yaml", "")


@pytest.mark.parametrize("name", sorted(PARAM_BANDS))
def test_param_count_within_budget(name: str) -> None:
    cfg = load_config(CONFIG_DIR / name, cuda_available=False)
    counts = cfg.model.n_params_estimate()
    lo, hi = PARAM_BANDS[name]
    assert lo <= counts["non_embedding"] <= hi, (
        f"{name}: non-embedding params {counts['non_embedding']:,} outside "
        f"[{lo:,}, {hi:,}]"
    )


def test_every_config_file_is_covered() -> None:
    """A new config must be added to PARAM_BANDS, not silently untested."""
    on_disk = {p.name for p in CONFIG_DIR.glob("*.yaml")}
    assert on_disk == set(PARAM_BANDS)


# --------------------------------------------------------------------------
# Derived geometry
# --------------------------------------------------------------------------


def test_derived_head_and_ffn_shapes() -> None:
    m = ModelConfig(d_model=512, n_heads=8, n_kv_heads=4, ffn_mult=2.6667)
    assert m.d_head == 64
    assert m.d_head * m.n_heads == m.d_model
    assert m.d_kv == 4 * 64
    assert m.ffn_hidden == 1408  # ceil(2.6667*512 / 64) * 64
    assert m.ffn_hidden % 64 == 0


def test_layer_pattern_tiles_to_n_layers() -> None:
    m = ModelConfig(n_layers=8, prefiller_pattern="SLSL", decoder_pattern="SSSS")
    pre = m.layer_kinds("prefiller")
    dec = m.layer_kinds("decoder")
    assert len(pre) == len(dec) == 8
    assert pre == list("SLSLSLSL")
    assert set(dec) == {"S"}


def test_separate_sharing_doubles_block_count() -> None:
    shared = ModelConfig(sharing="shared").n_params_estimate()["non_embedding"]
    separate = ModelConfig(sharing="separate").n_params_estimate()["non_embedding"]
    assert separate > shared


def test_grad_accum_hits_exact_token_budget() -> None:
    optim = OptimConfig(global_tokens_per_step=65536, micro_batch=16)
    assert optim.grad_accum(seq_len=512) == 8
    assert 8 * 16 * 512 == 65536


# --------------------------------------------------------------------------
# Validation: one rejection test per rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, needle",
    [
        (_mutated(model__d_model=250), "divisible by n_heads"),
        (_mutated(model__n_kv_heads=3), "divisible by"),
        (_mutated(model__prefiller_pattern="SXSX"), "only 'S' and 'L'"),
        (_mutated(model__prefiller_pattern="SLS"), "does not tile"),
        (_mutated(model__decoder_pattern="SLSL"), "sliding-window only"),
        (_mutated(model__window=9999), "exceeds max_seq_len"),
        (_mutated(data__seq_len=99999), "exceeds"),
        (_mutated(loss__lam=-1.0, loss__consistency="uniform"), "non-negative"),
        (_mutated(loss__lam=1.0), "no effect with consistency='none'"),
        (_mutated(loss__consistency="mask_topk", loss__lam=1.0), "requires loss.mask_k_pct"),
        (_mutated(loss__mask_k_pct=25.0), "does not use a mask"),
        (_mutated(loss__consistency="rwc", loss__lam=1.0, loss__influence__refresh_every=0), "refresh_every"),
        (_mutated(loss__consistency="rwc", loss__lam=1.0, loss__influence__ema_decay=1.0), "ema_decay"),
        (_mutated(loss__consistency="rwc", loss__lam=1.0, loss__influence__horizon_H=0), "horizon_H"),
        (_mutated(data__distances=[8, 700]), "does not fit"),
        (_mutated(optim__total_steps=100, optim__warmup=200), "warmup"),
        (_mutated(optim__global_tokens_per_step=1000), "not divisible by seq_len"),
        (_mutated(optim__micro_batch=17), "grad accumulation"),
        (_mutated(optim__schedule="magic"), "unknown optim.schedule"),
        (_mutated(run__seed=-1), "seed"),
    ],
)
def test_invalid_config_is_rejected(raw: Dict[str, Any], needle: str) -> None:
    with pytest.raises(ConfigError) as exc:
        _cuda_free(raw)
    assert needle in str(exc.value), f"unexpected message: {exc.value}"


def test_delayed_recall_rejects_distance_inside_window() -> None:
    """The task's whole point is a fact local attention provably cannot reach."""
    raw = _mutated(data__task="delayed_recall")  # tiny_synth has distances < W=64
    with pytest.raises(ConfigError) as exc:
        _cuda_free(raw)
    assert "outside the window" in str(exc.value)

    ok = _mutated(data__task="delayed_recall", data__distances=[96, 128, 192, 384])
    _cuda_free(ok)


def test_fp16_requires_cuda() -> None:
    raw = _mutated(optim__precision="fp16")
    with pytest.raises(ConfigError) as exc:
        config_from_dict(raw, cuda_available=False)
    assert "requires CUDA" in str(exc.value)
    config_from_dict(raw, cuda_available=True)  # fine on a GPU host


@pytest.mark.parametrize(
    "raw, path",
    [
        ({"model": {"d_modle": 256}}, "model"),
        ({"data": {"distancez": [8]}}, "data"),
        ({"loss": {"influence": {"emma_decay": 0.9}}}, "loss.influence"),
        ({"optim": {"learning_rate": 1e-3}}, "optim"),
    ],
)
def test_unknown_key_is_rejected(raw: Dict[str, Any], path: str) -> None:
    """A typo'd hyperparameter must be loud, not silently ignored."""
    with pytest.raises(ConfigError) as exc:
        _cuda_free(raw)
    assert path in str(exc.value)
    assert "unknown key" in str(exc.value)


def test_unknown_top_level_key_is_rejected() -> None:
    with pytest.raises(ConfigError) as exc:
        _cuda_free({"modle": {}})
    assert "unknown top-level key" in str(exc.value)


def test_missing_file_raises_config_error() -> None:
    with pytest.raises(ConfigError):
        load_config(CONFIG_DIR / "does_not_exist.yaml", cuda_available=False)


# --------------------------------------------------------------------------
# Config hash
# --------------------------------------------------------------------------


def test_hash_is_stable_across_reload() -> None:
    a = load_config(CONFIG_DIR / "tiny_synth.yaml", cuda_available=False)
    b = load_config(CONFIG_DIR / "tiny_synth.yaml", cuda_available=False)
    assert a.config_hash == b.config_hash
    assert len(a.config_hash) == 64


@pytest.mark.parametrize(
    "override",
    [
        {"loss.lam": 1.0, "loss.consistency": "uniform"},
        {"run.seed": 3},
        {"model.window": 32},
        {"optim.total_steps": 4001},
        {"data.distances": [8, 16]},
    ],
)
def test_hash_changes_with_scientific_fields(override: Dict[str, Any]) -> None:
    base = load_config(CONFIG_DIR / "tiny_synth.yaml", cuda_available=False)
    other = load_config(
        CONFIG_DIR / "tiny_synth.yaml", overrides=override, cuda_available=False
    )
    assert base.config_hash != other.config_hash


@pytest.mark.parametrize(
    "override",
    [
        {"run.out_dir": "/somewhere/else"},
        {"run.drive_dir": "/content/drive/MyDrive/rwc"},
        {"run.name": "renamed"},
        {"run.wandb": True},
    ],
)
def test_hash_ignores_bookkeeping_fields(override: Dict[str, Any]) -> None:
    base = load_config(CONFIG_DIR / "tiny_synth.yaml", cuda_available=False)
    other = load_config(
        CONFIG_DIR / "tiny_synth.yaml", overrides=override, cuda_available=False
    )
    assert base.config_hash == other.config_hash


def test_overrides_do_not_mutate_the_file() -> None:
    before = (CONFIG_DIR / "tiny_synth.yaml").read_text(encoding="utf-8")
    load_config(
        CONFIG_DIR / "tiny_synth.yaml", overrides={"run.seed": 9}, cuda_available=False
    )
    assert (CONFIG_DIR / "tiny_synth.yaml").read_text(encoding="utf-8") == before


def test_roundtrip_through_dict() -> None:
    cfg = load_config(CONFIG_DIR / "small_lm.yaml", cuda_available=False)
    again = config_from_dict(cfg.to_dict(), cuda_available=False)
    assert again.config_hash == cfg.config_hash
