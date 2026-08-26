"""Two objections that the width ladder cannot answer, on one harness.

    --study breadth   one task, one difficulty?
    --study depth     two layers everywhere?

BREADTH. Every synthetic number so far is `delayed_recall` with n_values=2, so
chance is 0.5 and a coin-flip readout carries the whole result. This repeats the
comparison at 16 values (chance 0.0625) with distractors, on all three tasks the
generator implements -- delayed_recall, needle, and two_hop, the last requiring
two chained memory reads rather than one. Geometry is the width ladder's base
rung exactly (d=128, L=2, W=4, reach 6, distances 8..24, seq_len 48), so only the
task changes.

DEPTH. The width ladder holds L=2 at every rung, deliberately: the stacked local
reach is L*(W-1), so scaling depth moves the reach and drags the probe distances
and sequence length with it. That is the right call for isolating parameter
count, and it leaves "your model is two layers" unanswered. Here depth is the
variable and the distance/reach RATIO is what is held fixed at 1.3x-4x:

    L=2   reach 6    distances [8,12,16,24]    seq_len 48
    L=4   reach 12   distances [16,24,32,48]   seq_len 96
    L=8   reach 24   distances [32,48,64,96]   seq_len 192

L=2 is re-run here rather than borrowed from ladder_runs, so all three rungs come
from one table at one batch size and no cross-study caveat is needed.

Depth and parameter count move together in this study (L=8 at d=128 is ~2M), so
it does NOT isolate parameters -- the width ladder already does that. What it
isolates is depth, which is what the objection asks about.

Usage:
    python scripts/variant_study.py --study breadth --out-dir breadth_runs
    python scripts/variant_study.py --study depth   --out-dir depth_runs
    python scripts/variant_study.py --study depth   --out-dir depth_runs --summary
"""

from __future__ import annotations

import argparse
import itertools
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

from cpu_ladder import stratified_exact_p  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

COMMON: Dict[str, Any] = {
    "model.d_model": 128, "model.window": 4,
    "model.n_heads": 4, "model.n_kv_heads": 2,
    "model.prefiller_pattern": "SL", "model.decoder_pattern": "SS",
    "data.loss_on_answer_only": True,
    "optim.total_steps": 1500, "optim.warmup": 100, "optim.lr": 3e-3,
    "optim.train_mode": "bptt",
    "run.ckpt_every": 100, "run.log_every": 500,
}

# 16 values and keys puts chance at 0.0625 instead of 0.5, and two distractor
# pairs mean the query is only answerable by binding the right key.
HARD = {"data.n_values": 16, "data.n_keys": 16, "data.n_filler": 16,
        "data.n_distractors": 2}
BASE_GEOM = {"data.seq_len": 48, "model.max_seq_len": 48,
             "data.distances": [8, 12, 16, 24],
             "optim.global_tokens_per_step": 32 * 48, "optim.micro_batch": 32}

STUDIES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "breadth": {
        "dr16": {**BASE_GEOM, **HARD, "model.n_layers": 2,
                 "data.task": "delayed_recall"},
        "needle16": {**BASE_GEOM, **HARD, "model.n_layers": 2,
                     "data.task": "needle"},
        "hop16": {**BASE_GEOM, **HARD, "model.n_layers": 2,
                  "data.task": "two_hop"},
    },
    # micro_batch 16 throughout: at L=8, seq 192 the retained BPTT graph at
    # batch 32 is ~2.5 GB, and a rung that OOMs is not a comparison.
    "depth": {
        "L2": {"model.n_layers": 2, "data.task": "delayed_recall",
               "data.seq_len": 48, "model.max_seq_len": 48,
               "data.distances": [8, 12, 16, 24],
               "data.n_values": 2, "data.n_keys": 4, "data.n_filler": 8,
               "data.n_distractors": 0,
               "optim.global_tokens_per_step": 16 * 48, "optim.micro_batch": 16},
        "L4": {"model.n_layers": 4, "data.task": "delayed_recall",
               "data.seq_len": 96, "model.max_seq_len": 96,
               "data.distances": [16, 24, 32, 48],
               "data.n_values": 2, "data.n_keys": 4, "data.n_filler": 8,
               "data.n_distractors": 0,
               "optim.global_tokens_per_step": 16 * 96, "optim.micro_batch": 16},
        "L8": {"model.n_layers": 8, "data.task": "delayed_recall",
               "data.seq_len": 192, "model.max_seq_len": 192,
               "data.distances": [32, 48, 64, 96],
               "data.n_values": 2, "data.n_keys": 4, "data.n_filler": 8,
               "data.n_distractors": 0,
               "optim.global_tokens_per_step": 16 * 192, "optim.micro_batch": 16},
    },
}
ARMS = {"plain": {}, "res": {"model.memory_residual": True}}
SEEDS = range(5)
MIN_FREE_GB = 0.6


def cells(study: str) -> List[Dict[str, Any]]:
    """Variants in declaration order, arms paired within a seed."""
    out = []
    for variant, over in STUDIES[study].items():
        for seed in SEEDS:
            for arm, extra in ARMS.items():
                merged = {**over, "run.seed": seed, **extra}
                out.append({"name": f"{variant}_{arm}_s{seed}", "variant": variant,
                            "arm": arm, "seed": seed, "over": merged})
    return out


def check_reach(cfg) -> None:
    """Every probe distance must exceed the stacked local reach.

    SyntheticGenerator enforces this for delayed_recall only. `needle` and
    `two_hop` get no such check, so without this a "memory" result could be
    solvable by local attention alone and nobody would notice.
    """
    reach = cfg.model.local_reach
    bad = [d for d in cfg.data.distances if d <= reach]
    if bad:
        raise SystemExit(
            f"distances {bad} are within stacked local reach L*(W-1)={reach} "
            f"for task {cfg.data.task!r}: the task would not require memory"
        )


def _free_gb(p: Path) -> float:
    return shutil.disk_usage(p).free / 1e9


def run_cell(cell: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    result = out_dir / cell["name"] / "result.json"
    if result.exists():
        return json.loads(result.read_text(encoding="utf-8"))

    over = dict(COMMON)
    over.update(cell["over"])
    over["run.name"] = cell["name"]
    over["run.out_dir"] = str(out_dir)
    cfg = load_config(CONFIGS / "tiny_synth.yaml", overrides=over, cuda_available=False)
    check_reach(cfg)

    started = time.time()
    trainer = Trainer(cfg, device=torch.device("cpu"))
    if trainer.maybe_resume():
        print(f"    resumed at step {trainer.step}/{cfg.optim.total_steps}", flush=True)
    trainer.run(quiet=True)

    gen = SyntheticGenerator(cfg.data, cfg.model, seed=1234)
    tokens = gen.batch(0, 2).tokens
    diag = diagnose_recurrence(trainer.model, tokens)
    info = memory_information(trainer.model, tokens)

    payload = {
        "name": cell["name"], "variant": cell["variant"], "arm": cell["arm"],
        "seed": cell["seed"], "task": cfg.data.task,
        "n_layers": cfg.model.n_layers, "local_reach": cfg.model.local_reach,
        "n_params": sum(p.numel() for n, p in trainer.model.named_parameters()
                        if "embed" not in n and "lm_head" not in n),
        "final_ce": trainer.history[-1].ce,
        "deployed": trainer.evaluate(n_per_distance=128, mode="deployed"),
        "rho": diag.rho, "ablation": diag.ablation_delta,
        "g_rec": diag.g_rec, "verdict": diag.verdict(),
        "spread": info["spread"],
        "secs": time.time() - started,
        "config": cfg.to_dict(),
    }
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if trainer.ckpt_path.exists():
        trainer.ckpt_path.unlink()
    return payload


def scored(p: Dict[str, Any]) -> Dict[str, Any]:
    buckets = p["deployed"]
    mean = sum(v["acc"] for v in buckets.values()) / len(buckets)
    n = sum(v["n"] for v in buckets.values())
    chance = 1.0 / p["config"]["data"]["n_values"]
    hi = wilson_interval(round(chance * n), n)[1]
    return {"mean": mean, "chance": chance, "hi": hi, "works": mean > hi}


def summarise(study: str, out_dir: Path) -> None:
    rows = [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(out_dir.glob("*/result.json"))]
    if not rows:
        print("no results yet")
        return
    order = list(STUDIES[study])

    print(f"\n{'cell':<20} {'task':<15} {'L':>2} {'reach':>6} {'deployed':>9} "
          f"{'chance':>7} {'rho':>7}  per-distance")
    print("-" * 110)
    for p in sorted(rows, key=lambda r: (order.index(r["variant"]) if r["variant"]
                                         in order else 99, r["arm"], r["seed"])):
        s = scored(p)
        per = " ".join(f"D{k}:{b['acc']:.2f}" for k, b in p["deployed"].items())
        print(f"{p['name']:<20} {p['task']:<15} {p['n_layers']:>2} "
              f"{p['local_reach']:>6} {s['mean']:>9.3f} {s['chance']:>7.3f} "
              f"{p['rho']:>7.3f}  {per}")

    print(f"\n{'variant':>10} {'task':<15} {'params':>8} {'Maglev':>9} "
          f"{'identity':>10} {'Fisher p':>10}")
    print("-" * 68)
    strata = []
    for variant in order:
        at = [p for p in rows if p["variant"] == variant]
        if not at:
            continue
        kp = sum(scored(p)["works"] for p in at if p["arm"] == "plain")
        npl = sum(1 for p in at if p["arm"] == "plain")
        kr = sum(scored(p)["works"] for p in at if p["arm"] == "res")
        nr = sum(1 for p in at if p["arm"] == "res")
        if npl and nr:
            strata.append((kr, nr, kp, npl))
            pv = f"{stratified_exact_p([(kr, nr, kp, npl)]):.2e}"
        else:
            pv = "  --"
        print(f"{variant:>10} {at[0]['task']:<15} {at[0]['n_params'] / 1e6:>7.1f}M "
              f"{kp:>5}/{npl:<3} {kr:>6}/{nr:<3} {pv:>10}")

    # At 16 values the binary "works" rate is the wrong readout. It was built for
    # the 2-value task, whose outcome is bimodal -- 1.000 or chance -- so a
    # threshold loses nothing. With 16 values the outcome is GRADED and the
    # Wilson bound for chance sits near 0.09, so a cell at 0.143 counts as
    # "LEARNED" beside one at 0.906 and a sixfold gap is thrown away. Report the
    # accuracy itself, paired by seed.
    print(f"\n{'variant':>10} {'plain acc':>10} {'identity acc':>13} "
          f"{'ratio':>7} {'paired p':>9}")
    print("-" * 54)
    for variant in order:
        at = [p for p in rows if p["variant"] == variant]
        if not at:
            continue
        pl = {p["seed"]: scored(p)["mean"] for p in at if p["arm"] == "plain"}
        rs = {p["seed"]: scored(p)["mean"] for p in at if p["arm"] == "res"}
        shared = sorted(set(pl) & set(rs))
        if not shared:
            continue
        mp = sum(pl[s] for s in shared) / len(shared)
        mr = sum(rs[s] for s in shared) / len(shared)
        d = [rs[s] - pl[s] for s in shared]
        mean = sum(d) / len(d)
        signs = list(itertools.product([1, -1], repeat=len(d)))
        pv = sum(1 for sg in signs
                 if sum(a * b for a, b in zip(d, sg)) / len(d) >= mean) / len(signs)
        print(f"{variant:>10} {mp:>10.3f} {mr:>13.3f} "
              f"{mr / max(mp, 1e-9):>6.1f}x {pv:>9.4f}")
    print("paired permutation over seeds; floor is 1/2^n_seeds")

    if len(strata) > 1:
        print(f"\nacross {len(strata)} variants, stratified:")
        print(f"  Maglev write rule  {sum(s[2] for s in strata)}/"
              f"{sum(s[3] for s in strata)}")
        print(f"  + identity path    {sum(s[0] for s in strata)}/"
              f"{sum(s[1] for s in strata)}")
        print(f"  exact conditional test:  p = {stratified_exact_p(strata):.2e}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", required=True, choices=sorted(STUDIES))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.summary:
        summarise(args.study, out_dir)
        return 0

    todo = cells(args.study)
    if args.plan:
        for variant in STUDIES[args.study]:
            n = len([c for c in todo if c["variant"] == variant])
            print(f"  {variant:<10} {n} cells")
        print(f"{len(todo)} cells total")
        return 0

    torch.set_num_threads(8)
    done = sum(1 for c in todo if (out_dir / c["name"] / "result.json").exists())
    print(f"[{args.study}] {len(todo)} cells -> {out_dir} ({done} cached), "
          f"free {_free_gb(out_dir):.1f} GB", flush=True)
    for i, cell in enumerate(todo, 1):
        if not (out_dir / cell["name"] / "result.json").exists():
            if _free_gb(out_dir) < MIN_FREE_GB:
                print(f"[{i}] SKIP {cell['name']}: low disk", flush=True)
                continue
            print(f"[{i}/{len(todo)}] {cell['name']}", flush=True)
        p = run_cell(cell, out_dir)
        s = scored(p)
        print(f"[{i}/{len(todo)}] {p['name']:<20} ce={p['final_ce']:.4f} "
              f"deployed={s['mean']:.3f} (chance {s['chance']:.3f}) "
              f"rho={p['rho']:.3f} "
              f"{'LEARNED' if s['works'] else 'at chance'} "
              f"[{p['secs'] / 60:.0f} min]", flush=True)
    summarise(args.study, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
