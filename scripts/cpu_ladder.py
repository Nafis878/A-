"""Does the write-rule result survive a 55x larger model? A CPU width ladder.

The one objection the existing work cannot answer is scale: everything is at
d=128, L=2, ~0.5M non-embedding parameters. On delayed recall, at every probe
distance beyond the stacked local reach:

    Maglev write rule    3/10 seeds reach deployed 1.000
    + identity path     10/10                            p = 1.6e-3

A reviewer's reading of that is "a two-layer artifact". This answers it on CPU
by scaling WIDTH at a task held literally fixed.

Why width. The stacked local reach is ``L*(W-1)`` -- a depth quantity. Scaling
depth moves the reach, which forces the probe distances and the sequence length
to move with it, so the task changes at every rung and parameter count is
confounded with task difficulty. Holding ``L=2, W=4`` fixes the reach at 6, so
``seq_len 48`` and distances ``[8,12,16,24]`` (1.3x..4x reach) are IDENTICAL at
every rung, down to the sampled batches: the generator is a pure function of
(seed, step), so two arms at the same seed see the same data. Parameter count is
then the only variable that moves.

    d_model    non-emb    s/step (this CPU)    10 cells
       128        0.5M          0.90              3.8 h*
       256        1.7M          1.36              5.7 h
       512        6.9M          2.99             12.5 h
       768       15.6M          5.49             22.9 h
      1024       27.4M          9.92             41.3 h

* Measured over 3 steps after one warm-up step, which still carries allocator
and lazy-init overhead: in the actual run d=128 cells take ~10 min rather than
the 22.5 predicted, so the whole ladder lands near 43 h, not 95.

The top rung is 27.4M non-embedding -- the same scale as `small_lm`, and 55x
the model every published number here was measured on.

What this does NOT establish: depth scaling. L stays at 2, deliberately, because
that is what keeps the task comparable. scripts/scale_study.py covers the depth
axis and needs a GPU.

Each rung runs both arms across seeds -- ten per arm at d=128 and d=256, five at
the three expensive rungs -- so every rung yields a rate with a confidence
interval and a Fisher test rather than a single anecdote. Ten is not cosmetic:
at five seeds the smallest attainable p-value is 5/5 vs 0/5 = 1/252 = 4e-3, so
such a rung cannot individually clear 1e-3 however clean the separation. The
expensive rungs lean on the trend across rungs instead.

Rungs are ordered cheapest-first and arms are paired within a seed, so an
interrupted run still leaves balanced arms at every rung it finished.

Alongside accuracy each cell records the deterministic diagnostics (rho, the
Eq. 7 gate, the ablation delta, memory spread). Those are computed from the
trained weights and need no task to succeed, so they stay informative even at a
rung where every cell sits at chance.

Usage:
    python scripts/cpu_ladder.py --out-dir ladder_runs --plan
    python scripts/cpu_ladder.py --out-dir ladder_runs
    python scripts/cpu_ladder.py --out-dir ladder_runs --rung 1024
    python scripts/cpu_ladder.py --out-dir ladder_runs --summary
"""

from __future__ import annotations

import argparse
import json
import shutil
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
from rwc.train import Trainer  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# The CPU headline task, verbatim. Nothing in here varies across rungs.
BASE: Dict[str, Any] = {
    "model.n_layers": 2, "model.window": 4,          # stacked local reach = 6
    "model.prefiller_pattern": "SL", "model.decoder_pattern": "SS",
    "data.task": "delayed_recall",
    "data.seq_len": 48, "model.max_seq_len": 48,
    "data.distances": [8, 12, 16, 24],               # 1.3x .. 4x the reach
    "data.loss_on_answer_only": True,
    "data.n_values": 2, "data.n_keys": 4, "data.n_filler": 8,
    "data.n_distractors": 0,
    "optim.total_steps": 1500, "optim.warmup": 100, "optim.lr": 3e-3,
    "optim.global_tokens_per_step": 32 * 48, "optim.micro_batch": 32,
    "optim.train_mode": "bptt",
    "run.ckpt_every": 100, "run.log_every": 500,
}

# (d_model, n_heads, n_kv_heads, measured s/step on this CPU at 8 threads)
RUNGS = [
    (128, 4, 2, 0.90),
    (256, 8, 4, 1.36),
    (512, 8, 4, 2.99),
    (768, 12, 6, 5.49),
    (1024, 16, 8, 9.92),
]
PARAMS_AT = {128: 527_000, 256: 1_710_000, 512: 6_880_000,
             768: 15_600_000, 1024: 27_400_000}
# Seeds per arm, per rung. The cheap rungs can afford the archived study's ten,
# and the difference is not cosmetic: with five seeds the SMALLEST attainable
# p-value is 5/5 vs 0/5 = 1/252 = 4e-3, so a rung cannot individually clear 1e-3
# no matter how clean the separation. Ten seeds reach 1.6e-3 at 10/10 vs 3/10.
# The expensive rungs stay at five and lean on the trend across rungs instead.
SEEDS_AT = {128: 10, 256: 10, 512: 5, 768: 5, 1024: 5}
ARMS = {"plain": {}, "res": {"model.memory_residual": True}}

# A cell needs room for its checkpoint (weights + two Adam moments) written
# atomically, so twice that transiently, plus headroom. Estimated from the
# parameter count rather than fixed: 0.5 GB is ample at d=128 and not enough at
# d=1024, where a checkpoint is ~330 MB.
MIN_FREE_GB = 0.5


def cells(only: Optional[int]) -> List[Dict[str, Any]]:
    """Cheapest rung first; arms paired within a seed so truncation stays balanced."""
    out: List[Dict[str, Any]] = []
    for d, h, kv, secs in RUNGS:
        if only is not None and d != only:
            continue
        for seed in range(SEEDS_AT.get(d, 5)):
            for arm, extra in ARMS.items():
                over = {"model.d_model": d, "model.n_heads": h, "model.n_kv_heads": kv,
                        "run.seed": seed}
                over.update(extra)
                out.append({"name": f"d{d}_{arm}_s{seed}", "d": d, "arm": arm,
                            "seed": seed, "secs": secs, "over": over})
    return out


def _free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def _needs_gb(d_model: int) -> float:
    """Disk a cell needs: fp32 weights + two Adam moments, written atomically."""
    n = PARAMS_AT.get(d_model, 30_000_000)
    return MIN_FREE_GB + 2 * (3 * n * 4) / 1e9


def run_cell(cell: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    result = out_dir / cell["name"] / "result.json"
    if result.exists():
        return json.loads(result.read_text(encoding="utf-8"))

    over = dict(BASE)
    over.update(cell["over"])
    over["run.name"] = cell["name"]
    over["run.out_dir"] = str(out_dir)
    cfg = load_config(CONFIGS / "tiny_synth.yaml", overrides=over, cuda_available=False)

    started = time.time()
    trainer = Trainer(cfg, device=torch.device("cpu"))
    if trainer.maybe_resume():
        print(f"    resumed at step {trainer.step}/{cfg.optim.total_steps}", flush=True)
    trainer.run(quiet=True)

    n_par = sum(p.numel() for n, p in trainer.model.named_parameters()
                if "embed" not in n and "lm_head" not in n)

    # Diagnose on real task batches, not random tokens: rho and spread are
    # properties of the trajectory the model actually runs on.
    gen = SyntheticGenerator(cfg.data, cfg.model, seed=1234)
    tokens = gen.batch(0, 2).tokens
    diag = diagnose_recurrence(trainer.model, tokens)
    info = memory_information(trainer.model, tokens)

    payload = {
        "name": cell["name"], "d_model": cell["d"], "arm": cell["arm"],
        "seed": cell["seed"], "n_params": n_par,
        "final_ce": trainer.history[-1].ce,
        "deployed": trainer.evaluate(n_per_distance=128, mode="deployed"),
        "parallel": trainer.evaluate(n_per_distance=128, mode="parallel"),
        "rho": diag.rho, "sigma_max": diag.sigma_max,
        "ablation": diag.ablation_delta, "g_rec": diag.g_rec, "g_loc": diag.g_loc,
        "verdict": diag.verdict(),
        "spread": info["spread"], "mem_norm": info["norm"], "mem_rank": info["rank"],
        "secs": time.time() - started,
        "config": cfg.to_dict(),
    }
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # 5.7 GB free at the top rung's ~330 MB per checkpoint: the cell is cached by
    # result.json now, so its checkpoint has no further use.
    ckpt = trainer.ckpt_path
    if ckpt.exists():
        ckpt.unlink()
    return payload


def scored(payload: Dict[str, Any]) -> Dict[str, Any]:
    buckets = payload["deployed"]
    mean = sum(v["acc"] for v in buckets.values()) / len(buckets)
    n = sum(v["n"] for v in buckets.values())
    chance = 1.0 / payload["config"]["data"]["n_values"]
    hi = wilson_interval(round(chance * n), n)[1]
    return {"mean": mean, "chance": chance, "hi": hi, "works": mean > hi}


def summarise(out_dir: Path) -> None:
    rows = [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(out_dir.glob("*/result.json"))]
    if not rows:
        print("no results yet")
        return

    print(f"\n{'cell':<20} {'non-emb':>9} {'ce':>8} {'deployed':>9} {'rho':>7} "
          f"{'g_rec':>7} {'spread':>9}  per-distance")
    print("-" * 108)
    for p in sorted(rows, key=lambda r: (r["d_model"], r["arm"], r["seed"])):
        s = scored(p)
        per = " ".join(f"D{k}:{b['acc']:.2f}" for k, b in p["deployed"].items())
        print(f"{p['name']:<20} {p['n_params']/1e6:>8.1f}M {p['final_ce']:>8.4f} "
              f"{s['mean']:>9.3f} {p['rho']:>7.3f} {p['g_rec']:>7.3f} "
              f"{p['spread']:>9.5f}  {per}")

    print(f"\n{'rung':>7} {'non-emb':>9} {'Maglev rule':>13} {'identity path':>15} "
          f"{'Fisher p':>10}")
    print("-" * 60)
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        fisher_exact = None

    for d, _, _, _ in RUNGS:
        at = [p for p in rows if p["d_model"] == d]
        if not at:
            continue
        n_par = at[0]["n_params"]
        cnt = {}
        for arm in ("plain", "res"):
            hits = [scored(p)["works"] for p in at if p["arm"] == arm]
            cnt[arm] = (sum(hits), len(hits))
        (kp, np_), (kr, nr) = cnt["plain"], cnt["res"]
        p_txt = "  --"
        if fisher_exact is not None and np_ and nr:
            # One-sided: is the identity path MORE likely to learn the task?
            _, pv = fisher_exact([[kr, nr - kr], [kp, np_ - kp]], alternative="greater")
            p_txt = f"{pv:.2e}"
        print(f"{d:>7} {n_par/1e6:>8.1f}M {kp:>6}/{np_:<6} {kr:>8}/{nr:<6} {p_txt:>10}")

    print("\nreference, d=128 L=2 (established):  Maglev 3/10   identity 10/10   p=1.6e-3")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rung", type=int, default=None,
                    choices=[d for d, _, _, _ in RUNGS])
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop starting new cells after this much wall-clock")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summary:
        summarise(out_dir)
        return 0

    todo = cells(args.rung)
    if args.plan:
        total = 0.0
        print(f"{'cells':>6} {'rung':>7} {'est hours':>10} {'cumulative':>11}")
        for d, _, _, secs in RUNGS:
            if args.rung is not None and d != args.rung:
                continue
            n = len([c for c in todo if c["d"] == d])
            h = n * secs * BASE["optim.total_steps"] / 3600
            total += h
            print(f"{n:>6} {d:>7} {h:>9.1f} h {total:>10.1f} h")
        print(f"\n{len(todo)} cells, {total:.1f} h estimated")
        return 0

    torch.set_num_threads(8)  # 16 threads measured 23% SLOWER: memory-bound
    done = sum(1 for c in todo if (out_dir / c["name"] / "result.json").exists())
    print(f"{len(todo)} cells -> {out_dir}   ({done} already complete)", flush=True)
    print(f"free disk: {_free_gb(out_dir):.1f} GB\n", flush=True)

    t0 = time.time()
    skipped: List[str] = []
    for i, cell in enumerate(todo, 1):
        cached = (out_dir / cell["name"] / "result.json").exists()
        if not cached and args.max_hours and (time.time() - t0) / 3600 > args.max_hours:
            print(f"\nstopping: --max-hours {args.max_hours} reached", flush=True)
            break
        if not cached:
            free, need = _free_gb(out_dir), _needs_gb(cell["d"])
            if free < need:
                print(f"[{i}/{len(todo)}] SKIP {cell['name']}: needs {need:.1f} GB, "
                      f"{free:.1f} GB free", flush=True)
                skipped.append(cell["name"])
                continue
            print(f"[{i}/{len(todo)}] {cell['name']}  "
                  f"(~{cell['secs'] * BASE['optim.total_steps'] / 3600:.1f} h, "
                  f"needs {need:.1f} GB, {free:.1f} free)", flush=True)
        payload = run_cell(cell, out_dir)
        s = scored(payload)
        print(f"[{i}/{len(todo)}] {payload['name']:<20} ce={payload['final_ce']:.4f} "
              f"deployed={s['mean']:.3f} rho={payload['rho']:.3f} "
              f"{'LEARNED' if s['works'] else 'at chance'} "
              f"[{payload['secs'] / 60:.0f} min, {_free_gb(out_dir):.1f} GB free]",
              flush=True)

    if skipped:
        print(f"\nskipped {len(skipped)} cells for disk: {', '.join(skipped)}")
        print("free space and re-run the same command to pick them up.")
    summarise(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
