"""Dataclass configuration, YAML loading, and validation.

Plain dataclasses rather than Hydra/OmegaConf, per SPEC.md section 10. The one
non-obvious choice here is that loading *rejects unknown keys at every nesting
level*: a silently-ignored typo in a hyperparameter name is exactly the failure
mode that quietly poisons a research result, so it is made loud.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, get_type_hints

import yaml

__all__ = [
    "ConfigError",
    "ModelConfig",
    "DataConfig",
    "InfluenceConfig",
    "LossConfig",
    "OptimConfig",
    "RunConfig",
    "Config",
    "load_config",
    "config_from_dict",
]


class ConfigError(ValueError):
    """Raised for any invalid or internally inconsistent configuration."""


# Fields excluded from the config hash: they name *where* a run writes, not what
# it computes, so two runs differing only in these are scientifically identical.
_HASH_EXCLUDED_RUN_FIELDS = frozenset({"out_dir", "drive_dir", "wandb", "name"})

_LAYER_KINDS = frozenset({"S", "L"})

CONSISTENCY_KINDS = (
    "none",
    "uniform",
    "rwc",
    "mask_topk",
    "mask_bottomk",
    "mask_randomk",
)
_MASK_KINDS = frozenset({"mask_topk", "mask_bottomk", "mask_randomk"})
# Loss kinds that consume read-influence weights and therefore need InfluenceConfig.
_INFLUENCE_KINDS = frozenset({"rwc"}) | _MASK_KINDS

SYNTHETIC_TASKS = ("needle", "two_hop", "delayed_recall", "carrier_copy")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Transformer geometry plus the Maglev prefiller/decoder coupling."""

    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 4  # equal to n_heads => plain MHA; smaller => GQA
    ffn_mult: float = 2.6667  # SwiGLU: 8/3, so param count matches a 4x GELU FFN
    vocab_size: int = 64
    max_seq_len: int = 512
    window: int = 64  # W in Eq. 9

    # Layer patterns are strings so a different interleave is a one-line change
    # (SPEC.md section 3). They tile to n_layers.
    prefiller_pattern: str = "SLSL"
    decoder_pattern: str = "SSSS"

    sharing: Literal["shared", "separate"] = "shared"

    rope: bool = True
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    logit_cap: float = 15.0  # Eq. 4: softmax(15 * tanh(o / 15))
    tie_embeddings: bool = True

    # "math" materialises attention probabilities, which Estimator B needs to
    # read attn_mass off (SPEC.md section 4). "sdpa" is the fast training path.
    attn_impl: Literal["sdpa", "math"] = "sdpa"

    # Set only by the influence ground-truth task: fixes, by construction, which
    # memory dimensions can be read at all. None means "no constraint".
    carrier_gain_seed: Optional[int] = None
    carrier_frac: float = 0.25

    # ---- derived geometry -------------------------------------------------

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_kv(self) -> int:
        """Width of the (possibly grouped) key/value projection."""
        return self.n_kv_heads * self.d_head

    @property
    def ffn_hidden(self) -> int:
        # Round to a multiple of 64 for kernel friendliness; done here rather
        # than in the YAML so every config gets the same rounding rule.
        raw = self.ffn_mult * self.d_model
        return int(math.ceil(raw / 64) * 64)

    def layer_kinds(self, which: Literal["prefiller", "decoder"]) -> List[str]:
        """Expand a pattern string to exactly n_layers entries of 'S' or 'L'."""
        pattern = (
            self.prefiller_pattern if which == "prefiller" else self.decoder_pattern
        )
        reps = self.n_layers // len(pattern)
        return list(pattern * reps)

    def n_params_estimate(self) -> Dict[str, int]:
        """Parameter counts, split embedding vs non-embedding.

        A real function rather than arithmetic in a YAML comment, so the tests
        can assert the SPEC's parameter budgets instead of trusting a comment.
        """
        d, dkv, h = self.d_model, self.d_kv, self.ffn_hidden

        attn = d * d + 2 * (d * dkv) + d * d  # q, k, v, o
        ffn = 3 * d * h  # SwiGLU: gate, up, down
        norms = 2 * d  # pre-attn + pre-ffn RMSNorm gains
        per_block = attn + ffn + norms

        # Recurrent injection, decoder-only, layer-specific (SPEC.md section 3).
        # W_k_rec, W_v_rec: d -> d_kv.  G_loc, G_rec: d -> d_kv.  + RMSNorm gain.
        per_injection = 2 * (d * dkv) + 2 * (d * dkv) + dkv

        n_blocks = self.n_layers if self.sharing == "shared" else 2 * self.n_layers
        non_emb = n_blocks * per_block
        non_emb += self.n_layers * per_injection
        if self.sharing == "shared":
            non_emb += d  # one final norm, used by both paths
            non_emb += 2 * self.n_layers  # decoder-path residual scales (2 per block)
        else:
            non_emb += 2 * d  # an independent final norm per path

        emb = self.vocab_size * d
        if not self.tie_embeddings:
            emb += self.vocab_size * d

        return {"embedding": emb, "non_embedding": non_emb, "total": emb + non_emb}

    def validate(self) -> None:
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ConfigError("d_model, n_layers and n_heads must all be positive")
        if self.d_model % self.n_heads != 0:
            raise ConfigError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )
        if self.n_kv_heads <= 0 or self.n_heads % self.n_kv_heads != 0:
            raise ConfigError(
                f"n_heads={self.n_heads} must be divisible by "
                f"n_kv_heads={self.n_kv_heads}"
            )
        for which, pattern in (
            ("prefiller_pattern", self.prefiller_pattern),
            ("decoder_pattern", self.decoder_pattern),
        ):
            if not pattern:
                raise ConfigError(f"{which} must not be empty")
            bad = set(pattern) - _LAYER_KINDS
            if bad:
                raise ConfigError(
                    f"{which}={pattern!r} contains {sorted(bad)}; only 'S' and 'L' allowed"
                )
            if self.n_layers % len(pattern) != 0:
                raise ConfigError(
                    f"{which}={pattern!r} of length {len(pattern)} does not tile "
                    f"n_layers={self.n_layers}"
                )
        # The decoder is what gets deployed, and it is sliding-window only
        # (SPEC.md section 3). A full-causal layer here would leak unbounded
        # context into the "bounded memory" model and invalidate every result.
        if "L" in self.decoder_pattern:
            raise ConfigError(
                f"decoder_pattern={self.decoder_pattern!r} contains a full-causal "
                "layer; the deployed decoder must be sliding-window only"
            )
        if self.window <= 0:
            raise ConfigError("window must be positive")
        if self.window > self.max_seq_len:
            raise ConfigError(
                f"window={self.window} exceeds max_seq_len={self.max_seq_len}"
            )
        if self.vocab_size <= 0:
            raise ConfigError("vocab_size must be positive")
        if self.logit_cap <= 0:
            raise ConfigError("logit_cap must be positive")
        if not 0.0 < self.carrier_frac <= 1.0:
            raise ConfigError("carrier_frac must lie in (0, 1]")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Synthetic probe generation, or FineWeb-Edu shard streaming."""

    kind: Literal["synthetic", "lm"] = "synthetic"
    seq_len: int = 512

    # -- synthetic --
    task: Optional[Literal["needle", "two_hop", "delayed_recall", "carrier_copy"]] = (
        "needle"
    )
    # Retrieval distances. This list is the x-axis of every headline figure and
    # results are always reported per bucket, never averaged (SPEC.md section 6).
    distances: List[int] = field(
        default_factory=lambda: [8, 16, 32, 48, 64, 96, 128, 192, 256, 384]
    )
    n_keys: int = 16
    n_values: int = 16
    n_filler: int = 16

    # -- lm --
    shard_dir: Optional[str] = None
    tokenizer_path: Optional[str] = None
    val_tokens: int = 2_000_000

    def validate(self, model: ModelConfig) -> None:
        if self.seq_len <= 0:
            raise ConfigError("seq_len must be positive")
        if self.seq_len > model.max_seq_len:
            raise ConfigError(
                f"data.seq_len={self.seq_len} exceeds "
                f"model.max_seq_len={model.max_seq_len}"
            )

        if self.kind == "synthetic":
            if self.task is None:
                raise ConfigError("synthetic data requires data.task")
            if self.task not in SYNTHETIC_TASKS:
                raise ConfigError(
                    f"unknown task {self.task!r}; expected one of {SYNTHETIC_TASKS}"
                )
            if not self.distances:
                raise ConfigError("synthetic data requires a non-empty distances grid")
            if any(d <= 0 for d in self.distances):
                raise ConfigError("all distances must be positive")
            # Leave room for the query and answer tokens at the tail.
            if max(self.distances) >= self.seq_len - 2:
                raise ConfigError(
                    f"max distance {max(self.distances)} does not fit in "
                    f"seq_len={self.seq_len}"
                )
            # The point of delayed_recall is that the fact is *provably*
            # unreachable by local attention (SPEC.md section 6), so a distance
            # inside the window would silently weaken the task.
            if self.task == "delayed_recall":
                short = [d for d in self.distances if d <= model.window]
                if short:
                    raise ConfigError(
                        f"delayed_recall distances {short} are <= window="
                        f"{model.window}; the fact must be outside the window"
                    )
            if min(self.n_keys, self.n_values, self.n_filler) <= 0:
                raise ConfigError("n_keys, n_values and n_filler must be positive")

        elif self.kind == "lm":
            if self.shard_dir is None:
                raise ConfigError("lm data requires data.shard_dir")
            if self.val_tokens <= 0:
                raise ConfigError("val_tokens must be positive")
        else:
            raise ConfigError(f"unknown data.kind {self.kind!r}")


# --------------------------------------------------------------------------
# Influence + loss
# --------------------------------------------------------------------------


@dataclass
class InfluenceConfig:
    """Read-influence estimation (SPEC.md section 4)."""

    estimator: Literal["saliency", "grad"] = "saliency"
    horizon_H: int = 64  # positions j+1 .. j+H that the memory is read over
    refresh_every: int = 50
    ema_decay: float = 0.9
    detach_local: bool = True
    eps: float = 1e-8

    def validate(self, model: ModelConfig) -> None:
        if self.estimator not in ("saliency", "grad"):
            raise ConfigError(f"unknown influence.estimator {self.estimator!r}")
        if self.horizon_H < 1:
            raise ConfigError("influence.horizon_H must be >= 1")
        if self.refresh_every < 1:
            raise ConfigError("influence.refresh_every must be >= 1")
        if not 0.0 < self.ema_decay < 1.0:
            raise ConfigError("influence.ema_decay must lie strictly in (0, 1)")
        if self.eps <= 0:
            raise ConfigError("influence.eps must be positive")


@dataclass
class LossConfig:
    """CE plus the consistency term under test (SPEC.md section 5)."""

    lam: float = 0.0
    consistency: Literal[
        "none", "uniform", "rwc", "mask_topk", "mask_bottomk", "mask_randomk"
    ] = "none"
    mask_k_pct: Optional[float] = None
    influence: InfluenceConfig = field(default_factory=InfluenceConfig)

    def validate(self, model: ModelConfig) -> None:
        if self.consistency not in CONSISTENCY_KINDS:
            raise ConfigError(
                f"unknown loss.consistency {self.consistency!r}; "
                f"expected one of {CONSISTENCY_KINDS}"
            )
        if self.lam < 0:
            raise ConfigError("loss.lam must be non-negative")
        if self.consistency == "none" and self.lam != 0:
            raise ConfigError(
                f"loss.lam={self.lam} has no effect with consistency='none'; "
                "set consistency or set lam=0"
            )
        if self.consistency in _MASK_KINDS:
            if self.mask_k_pct is None:
                raise ConfigError(
                    f"loss.consistency={self.consistency!r} requires loss.mask_k_pct"
                )
            if not 0.0 < self.mask_k_pct <= 100.0:
                raise ConfigError(
                    f"loss.mask_k_pct={self.mask_k_pct} must lie in (0, 100]"
                )
        elif self.mask_k_pct is not None:
            raise ConfigError(
                f"loss.mask_k_pct is set but consistency={self.consistency!r} "
                "does not use a mask"
            )
        if self.consistency in _INFLUENCE_KINDS:
            self.influence.validate(model)

    @property
    def uses_influence(self) -> bool:
        return self.consistency in _INFLUENCE_KINDS


# --------------------------------------------------------------------------
# Optimisation + run
# --------------------------------------------------------------------------


@dataclass
class OptimConfig:
    lr: float = 3e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.1
    warmup: int = 200
    schedule: Literal["cosine", "linear", "constant"] = "cosine"
    grad_clip: float = 1.0
    total_steps: int = 4000

    # Fixed global token count per step regardless of GPU (SPEC.md section 7);
    # micro_batch=None means the trainer picks it from available VRAM.
    global_tokens_per_step: int = 131_072
    micro_batch: Optional[int] = None

    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    compile: bool = False  # default off: compile time is punishing on Colab

    def validate(self, data: DataConfig, *, cuda_available: bool) -> None:
        if self.lr <= 0:
            raise ConfigError("optim.lr must be positive")
        if len(self.betas) != 2 or not all(0.0 <= b < 1.0 for b in self.betas):
            raise ConfigError("optim.betas must be two values in [0, 1)")
        if self.weight_decay < 0:
            raise ConfigError("optim.weight_decay must be non-negative")
        if self.total_steps <= 0:
            raise ConfigError("optim.total_steps must be positive")
        if self.warmup < 0 or self.warmup >= self.total_steps:
            raise ConfigError("optim.warmup must satisfy 0 <= warmup < total_steps")
        if self.schedule not in ("cosine", "linear", "constant"):
            raise ConfigError(f"unknown optim.schedule {self.schedule!r}")
        if self.grad_clip < 0:
            raise ConfigError("optim.grad_clip must be non-negative")
        if self.global_tokens_per_step <= 0:
            raise ConfigError("optim.global_tokens_per_step must be positive")
        if self.global_tokens_per_step % data.seq_len != 0:
            raise ConfigError(
                f"global_tokens_per_step={self.global_tokens_per_step} is not "
                f"divisible by seq_len={data.seq_len}"
            )
        if self.micro_batch is not None:
            if self.micro_batch <= 0:
                raise ConfigError("optim.micro_batch must be positive")
            per_micro = self.micro_batch * data.seq_len
            if self.global_tokens_per_step % per_micro != 0:
                raise ConfigError(
                    f"global_tokens_per_step={self.global_tokens_per_step} is not "
                    f"divisible by micro_batch*seq_len={per_micro}; grad accumulation "
                    "would not hit the target token count exactly"
                )
        if self.precision not in ("auto", "bf16", "fp16", "fp32"):
            raise ConfigError(f"unknown optim.precision {self.precision!r}")
        if self.precision == "fp16" and not cuda_available:
            raise ConfigError("optim.precision='fp16' requires CUDA")

    def grad_accum(self, seq_len: int) -> int:
        """Micro-steps per optimizer step, once micro_batch is known."""
        if self.micro_batch is None:
            raise ConfigError("grad_accum requires micro_batch to be resolved first")
        return self.global_tokens_per_step // (self.micro_batch * seq_len)


@dataclass
class RunConfig:
    name: str = "unnamed"
    seed: int = 0
    out_dir: str = "results"
    drive_dir: Optional[str] = None
    ckpt_every: int = 250
    log_every: int = 10
    eval_every: int = 500
    heartbeat_secs: int = 60
    wandb: bool = False

    def validate(self, optim: OptimConfig) -> None:
        if not self.name:
            raise ConfigError("run.name must be non-empty")
        if self.seed < 0:
            raise ConfigError("run.seed must be non-negative")
        for attr in ("ckpt_every", "log_every", "eval_every", "heartbeat_secs"):
            if getattr(self, attr) < 1:
                raise ConfigError(f"run.{attr} must be >= 1")


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def validate(self, *, cuda_available: Optional[bool] = None) -> "Config":
        if cuda_available is None:
            cuda_available = _cuda_available()
        self.model.validate()
        self.data.validate(self.model)
        self.loss.validate(self.model)
        self.optim.validate(self.data, cuda_available=cuda_available)
        self.run.validate(self.optim)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def hashable_dict(self) -> Dict[str, Any]:
        """The config minus fields that only name where output goes.

        Seed is deliberately *included*: the run manifest must treat different
        seeds as different runs (SPEC.md section 8, priority 7).
        """
        d = self.to_dict()
        d["run"] = {
            k: v for k, v in d["run"].items() if k not in _HASH_EXCLUDED_RUN_FIELDS
        }
        return d

    @property
    def config_hash(self) -> str:
        blob = json.dumps(self.hashable_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

_T = TypeVar("_T")


def _build(cls: Type[_T], raw: Any, path: str) -> _T:
    """Instantiate a dataclass from a dict, rejecting unknown keys."""
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(raw).__name__}")

    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(
            f"{path}: unknown key(s) {unknown}; valid keys are {sorted(known)}"
        )

    # `from __future__ import annotations` makes f.type a string, so nested
    # dataclass fields are resolved through get_type_hints rather than f.type.
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for key, value in raw.items():
        hint = hints.get(key)
        if isinstance(hint, type) and is_dataclass(hint):
            kwargs[key] = _build(hint, value, f"{path}.{key}")
        else:
            kwargs[key] = value
    return cls(**kwargs)


def config_from_dict(
    raw: Dict[str, Any], *, cuda_available: Optional[bool] = None
) -> Config:
    """Build and validate a Config from a nested dict."""
    if not isinstance(raw, dict):
        raise ConfigError(f"expected a mapping at the top level, got {type(raw).__name__}")
    unknown = sorted(set(raw) - {f.name for f in fields(Config)})
    if unknown:
        raise ConfigError(f"unknown top-level key(s) {unknown}")

    cfg = Config(
        model=_build(ModelConfig, raw.get("model"), "model"),
        data=_build(DataConfig, raw.get("data"), "data"),
        loss=_build(LossConfig, raw.get("loss"), "loss"),
        optim=_build(OptimConfig, raw.get("optim"), "optim"),
        run=_build(RunConfig, raw.get("run"), "run"),
    )
    return cfg.validate(cuda_available=cuda_available)


def load_config(
    path: str | Path,
    *,
    overrides: Optional[Dict[str, Any]] = None,
    cuda_available: Optional[bool] = None,
) -> Config:
    """Load a YAML config file, apply dotted overrides, and validate.

    Overrides use dotted keys, e.g. {"loss.lam": 1.0, "run.seed": 3}, which is
    what scripts/run_queue.py uses to expand the experiment grid from a small
    number of base configs.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if overrides:
        raw = _apply_overrides(raw, overrides)
    return config_from_dict(raw, cuda_available=cuda_available)


def _apply_overrides(raw: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(raw))  # deep copy of plain YAML data
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node = out
        for part in parts[:-1]:
            nxt = node.get(part)
            if nxt is None:
                nxt = {}
                node[part] = nxt
            elif not isinstance(nxt, dict):
                raise ConfigError(f"override {dotted!r} traverses non-mapping {part!r}")
            node = nxt
        node[parts[-1]] = value
    return out


def _cuda_available() -> bool:
    # Imported lazily so config loading does not pay the torch import cost in
    # contexts (figure scripts, manifest surgery) that never touch a GPU.
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
