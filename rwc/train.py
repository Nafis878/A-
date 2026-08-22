"""Single-run training loop, resumable across Colab disconnects.

SPEC.md section 7 treats disconnection as the normal case, and that drives the
whole design here:

* Every source of state is checkpointed -- model, optimizer, grad scaler, LR
  schedule position, dataloader position, and the RNG state of torch/CUDA/numpy/
  python -- so a resumed run is bit-comparable to an uninterrupted one.
* Metrics go to a CSV that is flushed every write, because stdout dies with the
  session.
* Checkpoint writes are atomic (temp file + os.replace), because Drive is slow
  and a session can vanish mid-write.
* A heartbeat file lets scripts/run_queue.py reclaim a run whose session died.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import random
import signal
import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from .config import Config, ConfigError, load_config
from .data.synthetic import ProbeBatch, SyntheticGenerator
from .evaluate import probe_accuracy
from .influence import InfluenceEMA, grad_influence, saliency_influence
from .losses import MASK_KINDS, consistency_loss, cross_entropy, influence_mask
from .model.baselines import full_attention_baseline, swa_baseline
from .model.maglev import MaglevModel

__all__ = ["Trainer", "resolve_precision", "resolve_batching", "seed_everything", "main"]

CHECKPOINT_VERSION = 1
CSV_COLUMNS = [
    "step",
    "loss",
    "ce",
    "lr",
    "grad_norm",
    "tokens",
    "elapsed_s",
    # Logged every step so a collapse of w is visible early: entropy_ratio ~1
    # means RWC has degenerated into the uniform L2 it is meant to improve on,
    # and ~0 means it has spiked onto one dimension. Both look fine in the loss.
    "w_mean",
    "w_std",
    "w_max",
    "w_entropy_ratio",
    "w_max_over_uniform",
    "consistency",
]


# --------------------------------------------------------------------------
# Setup helpers
# --------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_precision(cfg: Config, device: torch.device) -> Tuple[str, bool]:
    """Returns ``(amp_dtype_name, use_grad_scaler)``.

    bf16 on A100-class hardware, fp16 with a scaler on a T4, fp32 on CPU
    (SPEC.md section 7).
    """
    want = cfg.optim.precision
    if device.type != "cuda":
        if want in ("bf16", "fp16"):
            raise ConfigError(f"precision={want!r} requires CUDA")
        return "fp32", False
    if want == "auto":
        want = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return want, want == "fp16"


def gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def resolve_batching(cfg: Config, device: torch.device) -> Tuple[int, int]:
    """Pick ``(micro_batch, grad_accum)`` hitting the token budget exactly.

    The global token count per step is fixed by config and must not depend on
    which GPU Colab handed out, or runs stop being comparable. So micro_batch is
    only ever chosen from the *divisors* of the total sequence count.
    """
    seq_len = cfg.data.seq_len
    total_seqs = cfg.optim.global_tokens_per_step // seq_len

    if cfg.optim.micro_batch is not None:
        micro = cfg.optim.micro_batch
    else:
        if device.type != "cuda":
            micro = min(8, total_seqs)
        else:
            free = torch.cuda.get_device_properties(0).total_memory
            params = cfg.model.n_params_estimate()["total"]
            # Rough: activations dominate and scale with micro*seq_len*d*layers.
            # Deliberately conservative -- an OOM on Colab costs a whole run.
            per_seq = 24 * seq_len * cfg.model.d_model * cfg.model.n_layers
            budget = max(free - 20 * params, free // 4)
            micro = max(1, min(total_seqs, int(budget // max(per_seq, 1))))
        micro = _largest_divisor_at_most(total_seqs, micro)

    if total_seqs % micro != 0:
        raise ConfigError(
            f"micro_batch={micro} does not divide {total_seqs} sequences per step"
        )
    return micro, total_seqs // micro


def _largest_divisor_at_most(n: int, cap: int) -> int:
    cap = max(1, min(cap, n))
    for candidate in range(cap, 0, -1):
        if n % candidate == 0:
            return candidate
    return 1


def build_model(cfg: Config) -> nn.Module:
    kind = cfg.model.kind
    if kind == "maglev":
        return MaglevModel(cfg.model)
    if kind == "swa":
        return swa_baseline(cfg.model)
    if kind == "full":
        return full_attention_baseline(cfg.model)
    raise ConfigError(f"unknown model.kind {kind!r}")


def forward_logits(model: nn.Module, tokens: Tensor):
    """One interface over Maglev and the baselines, for *training*.

    Returns ``(logits, m, m_prime)``; the baselines have no prefiller so
    ``m_prime`` is None and the consistency term is inapplicable to them.
    """
    if isinstance(model, MaglevModel):
        out = model.forward_train(tokens)
        return out.logits, out.m, out.m_prime
    logits, h = model(tokens)
    return logits, h, None


def deployed_logits(model: nn.Module, tokens: Tensor) -> Tensor:
    """Logits from the model as actually *deployed* (Eq. 10).

    The prefiller Q is discarded and P runs recurrently on its own ``m_{t-1}``.
    This is not the same function as ``forward_logits``: there, Q has full
    causal attention over the whole sequence and hands the decoder a memory that
    already contains the answer, so probe accuracy saturates near 1.0 at every
    retrieval distance and the distance axis carries no signal at all. Measuring
    the deployed path is what makes the C3 slope observable.

    The baselines have no prefiller and no memory, so their training forward
    already *is* the deployed model.
    """
    if isinstance(model, MaglevModel):
        return model.forward_recurrent(tokens).logits
    return model(tokens)[0]


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------


@dataclass
class StepMetrics:
    step: int
    loss: float
    ce: float
    lr: float
    grad_norm: float
    tokens: int
    elapsed_s: float
    w_mean: float = float("nan")
    w_std: float = float("nan")
    w_max: float = float("nan")
    w_entropy_ratio: float = float("nan")
    w_max_over_uniform: float = float("nan")
    consistency: float = 0.0


class Trainer:
    def __init__(
        self,
        cfg: Config,
        *,
        device: Optional[torch.device] = None,
        run_dir: Optional[Path] = None,
    ) -> None:
        cfg.validate(cuda_available=torch.cuda.is_available())
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_dtype, self.use_scaler = resolve_precision(cfg, self.device)
        self.micro_batch, self.accum = resolve_batching(cfg, self.device)

        # Seed *before* building the model, so init is reproducible from config.
        seed_everything(cfg.run.seed)
        self.model = build_model(cfg).to(self.device)
        if cfg.optim.compile:
            self.model = torch.compile(self.model)  # off by default on Colab
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.optim.lr,
            betas=tuple(cfg.optim.betas),
            weight_decay=cfg.optim.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        self.data = self._build_data()
        self.influence = (
            InfluenceEMA(cfg.loss.influence, cfg.model.d_model, self.device)
            if cfg.loss.uses_influence
            else None
        )

        # A fixed subset for mask_randomk -- the control for "the top k%" is
        # "some k%", not a fresh k% each step (see losses.influence_mask).
        self._mask_generator = torch.Generator(device="cpu").manual_seed(cfg.run.seed)
        self._fixed_mask: Optional[Tensor] = None

        self.step = 0
        self.history: List[StepMetrics] = []
        self._t0 = time.time()
        self._elapsed_before_resume = 0.0
        self._stop_requested = False
        self._last_heartbeat = 0.0

        self.run_dir = Path(run_dir) if run_dir else Path(cfg.run.out_dir) / cfg.run.name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._install_signal_handlers()

    # ---- data -------------------------------------------------------------

    def _build_data(self):
        if self.cfg.data.kind == "synthetic":
            return SyntheticGenerator(
                self.cfg.data,
                self.cfg.model,
                seed=self.cfg.run.seed,
                n_distractors=self.cfg.data.n_distractors,
            )
        from .data.lm import ShardLoader

        return ShardLoader(self.cfg.data, seed=self.cfg.run.seed)

    def _micro_batch(self, step: int, micro: int) -> ProbeBatch:
        """Data position is a pure function of (step, micro), hence resumable."""
        return self.data.batch(step * self.accum + micro, self.micro_batch)

    # ---- schedule ---------------------------------------------------------

    def lr_at(self, step: int) -> float:
        o = self.cfg.optim
        if step < o.warmup:
            return o.lr * (step + 1) / max(1, o.warmup)
        if o.schedule == "constant":
            return o.lr
        progress = (step - o.warmup) / max(1, o.total_steps - o.warmup)
        progress = min(1.0, max(0.0, progress))
        if o.schedule == "linear":
            return o.lr * (1.0 - progress)
        return o.lr * 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine

    # ---- one step ---------------------------------------------------------

    def _autocast(self):
        if self.amp_dtype == "fp32":
            return nullcontext()
        dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16
        return torch.autocast(self.device.type, dtype=dtype)

    def train_step(self) -> StepMetrics:
        lr = self.lr_at(self.step)
        for group in self.opt.param_groups:
            group["lr"] = lr

        self._maybe_refresh_influence()

        self.opt.zero_grad(set_to_none=True)
        ce_total = 0.0
        cons_total = 0.0
        for micro in range(self.accum):
            batch = self._micro_batch(self.step, micro)
            tokens = batch.tokens.to(self.device)
            targets = batch.targets.to(self.device)
            with self._autocast():
                logits, m, m_prime = forward_logits(self.model, tokens)
                ce = cross_entropy(logits, targets)
                cons = self._consistency_terms(tokens, m, m_prime)
                loss = ce + self.cfg.loss.lam * cons
            self.scaler.scale(loss / self.accum).backward()
            ce_total += float(ce.detach()) / self.accum
            cons_total += float(cons.detach()) / self.accum

        if self.use_scaler:
            self.scaler.unscale_(self.opt)
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
        )
        self.scaler.step(self.opt)
        self.scaler.update()

        self.step += 1
        metrics = StepMetrics(
            step=self.step,
            loss=ce_total + self.cfg.loss.lam * cons_total,
            ce=ce_total,
            lr=lr,
            grad_norm=grad_norm,
            tokens=self.step * self.cfg.optim.global_tokens_per_step,
            elapsed_s=self.elapsed,
            consistency=cons_total,
            **(self.influence.stats() if self.influence is not None else {}),
        )
        self.history.append(metrics)
        return metrics

    def _consistency(self, m: Tensor, m_prime: Optional[Tensor]) -> Tensor:
        """The consistency term of SPEC.md section 5, or zero if inapplicable."""
        kind = self.cfg.loss.consistency
        if kind == "none" or self.cfg.loss.lam == 0:
            return m.new_zeros(())
        if m_prime is None:
            # SWA/full baselines have no prefiller, so there is no target to be
            # consistent with. Rejected at config load, but asserted here too.
            raise ConfigError(
                f"model.kind={self.cfg.model.kind!r} has no prefiller, so "
                f"loss.consistency={kind!r} is not applicable"
            )
        weights = mask = None
        if kind == "rwc":
            weights = self.influence.value.detach()
        elif kind in MASK_KINDS:
            mask = self._mask_for_step()
        return consistency_loss(
            m, m_prime, kind=kind, weights=weights, mask=mask,
            detach_target=self.cfg.loss.detach_target,
        )

    def _consistency_terms(
        self, tokens: Tensor, m: Tensor, m_prime: Optional[Tensor]
    ) -> Tensor:
        """Maglev's one-step term, plus the closed-loop iterates when k > 1.

        Iterate j feeds back the decoder's own memory from iterate j-1, which is
        exactly the closed loop for the first j positions. The fed-back memory is
        *detached*: the objective is "recover the target from the state you
        actually produced", so each pass carries a one-step gradient and the
        graph does not deepen with k. That keeps the cost k forward passes and
        the memory cost flat, and it is the same truncation scheduled sampling
        makes -- the difference is that this stays parallel.
        """
        k = self.cfg.loss.unroll_k
        total = self._consistency(m, m_prime)
        if k == 1 or m_prime is None or self.cfg.loss.lam == 0:
            return total

        prev = m
        for _ in range(2, k + 1):
            mem_in = MaglevModel.shift_memory(prev.detach())
            _, prev, _ = self.model.forward_decoder(tokens, mem_in)
            total = total + self.cfg.loss.unroll_weight * self._consistency(prev, m_prime)
        return total / (1.0 + (k - 1) * self.cfg.loss.unroll_weight)

    def _mask_for_step(self) -> Tensor:
        """Top-k / bottom-k track the EMA; random-k is drawn once and kept."""
        kind = self.cfg.loss.consistency
        k_pct = self.cfg.loss.mask_k_pct
        if kind == "mask_randomk":
            if self._fixed_mask is None:
                self._fixed_mask = influence_mask(
                    self.influence.value, kind, k_pct, generator=self._mask_generator
                ).to(self.device)
            return self._fixed_mask
        return influence_mask(self.influence.value.detach(), kind, k_pct)

    def _maybe_refresh_influence(self) -> None:
        """Recompute read-influence on a slow EMA (SPEC.md section 5).

        Refreshing every step would cost a whole extra pass; the EMA makes the
        amortised cost negligible while keeping the weights current.
        """
        if self.influence is None or not self.influence.should_refresh(self.step):
            return
        batch = self._micro_batch(self.step, 0)
        tokens = batch.tokens.to(self.device)
        cfg = self.cfg.loss.influence
        self.model.eval()
        try:
            if cfg.estimator == "saliency":
                w = saliency_influence(self.model, tokens, eps=cfg.eps)
            else:
                w = grad_influence(
                    self.model,
                    tokens,
                    batch.targets.to(self.device),
                    horizon_H=cfg.horizon_H,
                    detach_local=cfg.detach_local,
                    eps=cfg.eps,
                )
        finally:
            self.model.train()
        self.influence.update(w)

    @property
    def elapsed(self) -> float:
        return self._elapsed_before_resume + (time.time() - self._t0)

    # ---- the loop ---------------------------------------------------------

    def run(self, *, max_steps: Optional[int] = None, quiet: bool = False) -> Dict[str, Any]:
        stop_at = self.cfg.optim.total_steps
        if max_steps is not None:
            stop_at = min(stop_at, self.step + max_steps)

        while self.step < stop_at and not self._stop_requested:
            metrics = self.train_step()
            self._log_csv(metrics)
            if not quiet and self.step % self.cfg.run.log_every == 0:
                print(self._format(metrics), flush=True)
            if self.step % self.cfg.run.ckpt_every == 0:
                self.save_checkpoint()
                if not quiet:
                    self._banner(metrics)
            self._maybe_heartbeat()

        if self._stop_requested:
            self.save_checkpoint()
            print("\n[rwc] stop requested; checkpointed and exiting.", flush=True)
        elif self.step >= self.cfg.optim.total_steps:
            self.save_checkpoint()

        return self.result()

    def result(self) -> Dict[str, Any]:
        """The per-run JSON of SPEC.md section 8."""
        return {
            "config": self.cfg.to_dict(),
            "config_hash": self.cfg.config_hash,
            "git_commit": _git_commit(),
            "gpu": gpu_name(),
            "precision": self.amp_dtype,
            "micro_batch": self.micro_batch,
            "grad_accum": self.accum,
            "seed": self.cfg.run.seed,
            "steps": self.step,
            "wall_clock_s": self.elapsed,
            "final_ce": self.history[-1].ce if self.history else None,
            "final_consistency": self.history[-1].consistency if self.history else None,
            # Headline: the deployed model (Eq. 10). The parallel figure is kept
            # alongside it purely as a diagnostic -- a large gap between the two
            # means the memory chain is not carrying what the prefiller wrote,
            # which is precisely what the consistency loss is meant to fix.
            "probe_accuracy_by_distance": self.evaluate(mode="deployed"),
            "probe_accuracy_parallel": self.evaluate(mode="parallel"),
        }

    def evaluate(
        self, *, n_per_distance: int = 64, mode: str = "deployed"
    ) -> Dict[str, Dict[str, float]]:
        """Probe accuracy per distance bucket.

        ``mode="deployed"`` runs Eq. 10 -- Q discarded, P recurrent on its own
        memory -- and is the headline metric. ``mode="parallel"`` keeps the
        prefiller and is a diagnostic only: it saturates near 1.0 at every
        distance, because Q's full attention hands the decoder the answer.
        """
        if self.cfg.data.kind != "synthetic":
            return {}
        if mode not in ("deployed", "parallel"):
            raise ValueError(f"unknown evaluate mode {mode!r}")
        forward = (
            (lambda tokens: deployed_logits(self.model, tokens))
            if mode == "deployed"
            else (lambda tokens: forward_logits(self.model, tokens)[0])
        )
        self.model.eval()
        try:
            # Eval holds no gradients, so it is not bound by micro_batch --
            # which is sized for the backward pass and can be very small. Wide
            # batches matter here because the deployed path walks the sequence
            # one step at a time.
            batches = self.data.eval_batches(
                batch_size=min(n_per_distance, max(self.micro_batch, 64)),
                n_per_distance=n_per_distance,
            )
            with torch.no_grad():
                return probe_accuracy(forward, batches, device=self.device)
        finally:
            self.model.train()

    # ---- reporting --------------------------------------------------------

    def _format(self, m: StepMetrics) -> str:
        eta = self._eta()
        return (
            f"[rwc] step {m.step:>6}/{self.cfg.optim.total_steps} "
            f"ce {m.ce:.4f} lr {m.lr:.2e} gnorm {m.grad_norm:.2f} "
            f"{m.tokens/1e6:.1f}M tok  eta {eta}"
        )

    def _eta(self) -> str:
        if self.step == 0:
            return "?"
        per_step = self.elapsed / self.step
        remaining = (self.cfg.optim.total_steps - self.step) * per_step
        return f"{int(remaining // 3600)}h{int(remaining % 3600 // 60):02d}m"

    def _banner(self, m: StepMetrics) -> None:
        print(
            "\n" + "=" * 62 + "\n"
            f"  SAFE TO DISCONNECT  --  checkpoint at step {m.step}\n"
            f"  {self.ckpt_path}\n"
            f"  elapsed {self.elapsed/60:.1f} min   eta {self._eta()}\n"
            + "=" * 62 + "\n",
            flush=True,
        )

    def _log_csv(self, m: StepMetrics) -> None:
        path = self.run_dir / "metrics.csv"
        new = not path.exists()
        # Flushed every write: stdout dies with the Colab session, this must not.
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            if new:
                writer.writeheader()
            writer.writerow({k: getattr(m, k) for k in CSV_COLUMNS})
            fh.flush()

    def _maybe_heartbeat(self) -> None:
        now = time.time()
        if now - self._last_heartbeat < self.cfg.run.heartbeat_secs:
            return
        self._last_heartbeat = now
        _atomic_write_text(self.run_dir / "heartbeat", json.dumps({"t": now, "step": self.step}))

    # ---- checkpointing ----------------------------------------------------

    @property
    def ckpt_path(self) -> Path:
        return self.run_dir / "checkpoint.pt"

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": CHECKPOINT_VERSION,
            "config_hash": self.cfg.config_hash,
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "elapsed_s": self.elapsed,
            "data": self.data.state_dict() if hasattr(self.data, "state_dict") else {},
            "influence": self.influence.state_dict() if self.influence else None,
            "fixed_mask": self._fixed_mask,
            "mask_generator": self._mask_generator.get_state(),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError(f"checkpoint version {state.get('version')} not supported")
        if state["config_hash"] != self.cfg.config_hash:
            raise ValueError(
                "checkpoint was written by a different config "
                f"({state['config_hash'][:12]} vs {self.cfg.config_hash[:12]})"
            )
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.step = state["step"]
        self._elapsed_before_resume = state.get("elapsed_s", 0.0)
        self._t0 = time.time()
        if hasattr(self.data, "load_state_dict"):
            self.data.load_state_dict(state.get("data", {}))
        if self.influence is not None and state.get("influence"):
            self.influence.load_state_dict(state["influence"])
        # The random-k subset is part of the experiment, not a nuisance: a
        # resumed run that redrew it would be running a different ablation.
        if state.get("fixed_mask") is not None:
            self._fixed_mask = state["fixed_mask"].to(self.device)
        if state.get("mask_generator") is not None:
            self._mask_generator.set_state(state["mask_generator"])

        rng = state["rng"]
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])

    def save_checkpoint(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path else self.ckpt_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)  # atomic: a killed session never leaves a torn file
        _atomic_write_text(
            self.run_dir / "heartbeat", json.dumps({"t": time.time(), "step": self.step})
        )
        return path

    def maybe_resume(self, path: Optional[Path] = None) -> bool:
        """Resume if a checkpoint exists. Returns whether it did."""
        path = Path(path) if path else self.ckpt_path
        if not path.exists():
            return False
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
        print(f"[rwc] resumed from {path} at step {self.step}", flush=True)
        return True

    # ---- signals ----------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):  # noqa: ANN001
            self._stop_requested = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError, AttributeError):
                # Not the main thread, or the platform does not deliver it.
                pass


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Train one RWC run.")
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="dotted.key=value",
        help="config override, parsed as JSON then as a string",
    )
    args = p.parse_args(argv)

    overrides = {}
    for item in args.set:
        key, _, raw = item.partition("=")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw

    cfg = load_config(args.config, overrides=overrides)
    trainer = Trainer(
        cfg,
        device=torch.device(args.device) if args.device else None,
        run_dir=Path(args.run_dir) if args.run_dir else None,
    )
    trainer.maybe_resume()
    result = trainer.run(max_steps=args.max_steps, quiet=args.quiet)
    _atomic_write_text(trainer.run_dir / "result.json", json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
