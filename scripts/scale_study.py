"""Does the write-rule result survive a larger model?

Everything established so far is at d=128, L=2 on CPU. That is the one objection
the CPU work cannot answer, so this repeats the two load-bearing measurements at
`small_lm` geometry (d=512, L=8) on a GPU.

At CPU scale, on delayed recall with every distance beyond the stacked local
reach:

    Maglev write rule    3/10 seeds reach deployed 1.000, the rest sit at chance
    + identity path     10/10                                p = 1.6e-3

The claim under test here is that this is a property of the write rule -- the
state is rebuilt from scratch each step, so remembering requires learning the
identity function -- and not an artefact of a two-layer model.

Geometry is chosen to keep the task's shape identical: W=4 with L=8 gives a
stacked local reach of 24, and the probe distances (32..96) span 1.3x to 4x that
reach, matching the CPU study's 1.3x-4x.

Groups, in priority order so a truncated run still answers the main question:

  1  reliability   5 seeds x {Maglev write rule, identity path}, BPTT
  2  generality    16 values with distractors, both arms
  3  collapse      the consistency loss at lambda=0.1, both arms, teacher-forced

Usage:
    python scripts/scale_study.py --out-dir /content/drive/MyDrive/rwc/scale
    python scripts/scale_study.py --out-dir ... --group 1     # just reliability
    python scripts/scale_study.py --out-dir ... --summary     # read results back
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from rwc.analysis import (  # noqa: E402
    diagnose_recurrence,
    memory_information,
    wilson_interval,
)
from rwc.config import load_config  # noqa: E402
from rwc.data.synthetic import SyntheticGenerator  # noqa: E402
from rwc.evaluate import probe_accuracy  # noqa: E402
from rwc.model.maglev import MaglevModel  # noqa: E402
from rwc.train import Trainer  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# small_lm width and depth, with the synthetic probe's vocabulary and a window
# small enough that the probe distances stay outside the stacked local reach.
BASE: Dict[str, Any] = {
    "model.d_model": 512, "model.n_layers": 8,
    "model.n_heads": 8, "model.n_kv_heads": 4,
    "model.window": 4,                      # L*(W-1) = 24
    "model.prefiller_pattern": "SL", "model.decoder_pattern": "SS",
    "data.task": "delayed_recall",
    "data.seq_len": 128, "model.max_seq_len": 128,
    "data.distances": [32, 48, 64, 96],     # 1.3x .. 4x the reach
    "data.loss_on_answer_only": True,
    "data.n_values": 2, "data.n_keys": 4, "data.n_filler": 8,
    "data.n_distractors": 0,
    "optim.total_steps": 1500, "optim.warmup": 100, "optim.lr": 3e-3,
    "optim.global_tokens_per_step": 32 * 128, "optim.micro_batch": 32,
    "optim.train_mode": "bptt",
    "run.ckpt_every": 100, "run.log_every": 200,
}
HARD = {"data.n_values": 16, "data.n_keys": 16, "data.n_filler": 16,
        "data.n_distractors": 2}
RESIDUAL = {"model.memory_residual": True}


def cells(group: Optional[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def add(g: int, name: str, **over: Any) -> None:
        if group in (None, g):
            out.append({"group": g, "name": name, "over": over})

    for seed in range(5):
        add(1, f"s1_plain_s{seed}", **{"run.seed": seed})
        add(1, f"s1_res_s{seed}", **RESIDUAL, **{"run.seed": seed})
    add(2, "s2_plain_v16", **HARD)
    add(2, "s2_res_v16", **RESIDUAL, **HARD)
    # Teacher-forced with the consistency loss: Maglev's actual training scheme.
    add(3, "s3_lam01_plain", **{"optim.train_mode": "parallel",
                                "loss.consistency": "uniform", "loss.lam": 0.1})
    add(3, "s3_lam01_res", **RESIDUAL, **{"optim.train_mode": "parallel",
                                          "loss.consistency": "uniform",
                                          "loss.lam": 0.1})
    return out


def run_cell(cell: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    result = out_dir / cell["name"] / "result.json"
    if result.exists():
        return json.loads(result.read_text(encoding="utf-8"))

    over = dict(BASE)
    over.update(cell["over"])
    over["run.name"] = cell["name"]
    over["run.out_dir"] = str(out_dir)
    cfg = load_config(CONFIGS / "tiny_synth.yaml", overrides=over)

    started = time.time()
    trainer = Trainer(cfg)
    if trainer.maybe_resume():
        print(f"  resumed {cell['name']} at step {trainer.step}", flush=True)
    trainer.run(quiet=True)

    payload = {
        "name": cell["name"], "group": cell["group"],
        "final_ce": trainer.history[-1].ce,
        "deployed": trainer.evaluate(n_per_distance=128, mode="deployed"),
        "parallel": trainer.evaluate(n_per_distance=128, mode="parallel"),
        "secs": time.time() - started,
        "config": cfg.to_dict(),
    }
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def verdict(payload: Dict[str, Any]) -> Dict[str, Any]:
    buckets = payload["deployed"]
    mean = sum(v["acc"] for v in buckets.values()) / len(buckets)
    n = sum(v["n"] for v in buckets.values())
    chance = 1.0 / payload["config"]["data"]["n_values"]
    hi = wilson_interval(round(chance * n), n)[1]
    return {"mean": mean, "chance": chance, "hi": hi, "works": mean > hi}


def summarise(out_dir: Path) -> None:
    rows = []
    for f in sorted(out_dir.glob("*/result.json")):
        p = json.loads(f.read_text(encoding="utf-8"))
        rows.append((p, verdict(p)))
    if not rows:
        print("no results yet")
        return

    print(f"\n{'cell':<22} {'ce':>8} {'deployed':>9} {'chance':>7} {'':>4}  per-distance")
    print("-" * 92)
    for p, v in rows:
        per = "  ".join(f"D{k}:{b['acc']:.2f}" for k, b in p["deployed"].items())
        tag = "OK " if v["works"] else "   "
        print(f"{p['name']:<22} {p['final_ce']:>8.4f} {v['mean']:>9.3f} "
              f"{v['chance']:>7.3f} {tag:>4}  {per}")

    for arm, prefix in (("Maglev write rule", "s1_plain"), ("+ identity path", "s1_res")):
        hits = [v["works"] for p, v in rows if p["name"].startswith(prefix)]
        if hits:
            lo, hi = wilson_interval(sum(hits), len(hits))
            print(f"\n{arm:<22} {sum(hits)}/{len(hits)}   95% CI [{lo:.2f}, {hi:.2f}]")
    print("\nCPU reference (d=128, L=2):  Maglev 3/10   identity path 10/10   p=1.6e-3")


def diagnostics(out_dir: Path) -> None:
    """Checkpoint-only readouts: no task has to succeed for these to be valid."""
    print(f"\n{'cell':<22} {'rho':>7} {'g_rec':>7} {'ablate':>10} {'spread':>9}  state")
    print("-" * 78)
    for f in sorted(out_dir.glob("*/result.json")):
        p = json.loads(f.read_text(encoding="utf-8"))
        ckpt = f.parent / "checkpoint.pt"
        if not ckpt.exists():
            continue
        from rwc.config import config_from_dict

        cfg = config_from_dict(p["config"])
        model = MaglevModel(cfg.model).eval()
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        tokens = torch.randint(0, cfg.model.vocab_size, (2, cfg.data.seq_len),
                               generator=torch.Generator().manual_seed(0))
        d = diagnose_recurrence(model, tokens)
        info = memory_information(model, tokens)
        print(f"{p['name']:<22} {d.rho:>7.3f} {d.g_rec:>7.3f} "
              f"{d.ablation_delta:>10.2e} {info['spread']:>9.5f}  {d.verdict()}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--group", type=int, default=None, choices=(1, 2, 3))
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--diagnostics", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summary or args.diagnostics:
        if args.summary:
            summarise(out_dir)
        if args.diagnostics:
            diagnostics(out_dir)
        return 0

    todo = cells(args.group)
    print(f"{len(todo)} cells -> {out_dir}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    for cell in todo:
        payload = run_cell(cell, out_dir)
        v = verdict(payload)
        print(f"[g{cell['group']}] {payload['name']:<22} ce={payload['final_ce']:.4f} "
              f"deployed={v['mean']:.3f} (chance {v['chance']:.3f}) "
              f"{'ABOVE CHANCE' if v['works'] else 'at chance'} "
              f"[{payload['secs']:.0f}s]", flush=True)
    summarise(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
