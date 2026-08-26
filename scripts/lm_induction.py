"""Does the write rule fail at long-range exact retrieval IN NATURAL LANGUAGE?

scripts/lm_study.py measured aggregate held-out BPB on text8 and found the write
rule healthy: rho 0.27-0.43, memory buying it 1.52 bits, and a BETTER language
model than the identity path (1.9089 vs 1.9723 bpc, 0/5 seeds). Taken at face
value that is a null for the LM domain.

But aggregate BPB cannot test this hypothesis, and that is measurable rather than
arguable. Only 3.87% of held-out positions are ones where the next character is
fixed by a substring that last occurred beyond the decoder's reach; and most of
those are common phrases whose continuation local statistics already supply --
an UNTRAINED model scores better on them than on ordinary positions. Filtering to
substrings absent from 3M characters of training leaves 0.49% of positions,
median retrieval distance 72 characters, 12x the reach of 6. Those are the ones
where the continuation can only come from this context. A defect confined to them
is invisible in a mean over the other 99.5%.

THE PREDICTION, stated before the run. If the write rule fails specifically at
long-range exact retrieval, then against the identity path it should lose on
`induction_novel`, lose by more as distance grows, and may still win on `other`.
If it wins or ties on `induction_novel`, the mechanism does not transfer to
language and the claim stays confined to the synthetic probe. Both are reported.

Three groups, always reported together:

    induction_novel   repeat, substring absent from training  -> the real test
    induction_common  repeat, substring common in training    -> control
    other             everything else

An arm that wins on `induction_novel` while losing on `other` has made a
trade-off, and reporting only the flattering half would repeat the error of
quoting the confounded `benefit` metric in lm_study.py.

THIRD ARM. `res_b0` is the identity path with memory_gate_bias 0.0 instead of
2.0. Not a search for a better number: sigmoid(2)=0.88 retains 88% of memory per
step, the right prior for holding one fact across 24 steps and the wrong one for
language, where memory must turn over constantly. If the identity path's
aggregate deficit comes from that prior rather than the mechanism, bias 0.0
should close it. All three arms are reported whatever happens.

Checkpoints are KEPT here, unlike lm_study.py, whose deletion is what forced this
study to retrain from scratch.

Usage:
    python scripts/lm_induction.py --out-dir lm_ind_runs
    python scripts/lm_induction.py --out-dir lm_ind_runs --summary
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from rwc.analysis import diagnose_recurrence  # noqa: E402
from rwc.config import load_config  # noqa: E402
from rwc.data.lm import CharShardLoader, load_corpus  # noqa: E402
from rwc.evaluate import (  # noqa: E402
    build_reference,
    lm_induction_bpb,
    lm_memory_benefit,
)
from rwc.train import Trainer  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
MIN_MATCH = 8

BASE: Dict[str, Any] = {
    "data.kind": "lm", "data.shard_dir": "lm_data",
    "data.seq_len": 256, "data.val_tokens": 1_000_000,
    "model.max_seq_len": 256, "model.vocab_size": 27,
    "model.d_model": 128, "model.n_layers": 2, "model.window": 4,
    "model.n_heads": 4, "model.n_kv_heads": 2,
    "model.prefiller_pattern": "SL", "model.decoder_pattern": "SS",
    "optim.total_steps": 1500, "optim.warmup": 100, "optim.lr": 3e-3,
    "optim.train_mode": "bptt",
    "optim.global_tokens_per_step": 16 * 256, "optim.micro_batch": 16,
    "run.ckpt_every": 100, "run.log_every": 500,
}
ARMS = {
    "plain": {},
    "res": {"model.memory_residual": True},
    "res_b0": {"model.memory_residual": True, "model.memory_gate_bias": 0.0},
}
SEEDS = range(5)
_REFERENCE: Optional[set] = None


def reference_set() -> set:
    """Training-corpus substrings, built once and reused across cells."""
    global _REFERENCE
    if _REFERENCE is None:
        train, _, _ = load_corpus("lm_data")
        _REFERENCE = build_reference(train, min_match=MIN_MATCH)
        print(f"    reference: {len(_REFERENCE):,} distinct "
              f"{MIN_MATCH}-grams from training", flush=True)
    return _REFERENCE


def cells() -> List[Dict[str, Any]]:
    """plain and res first, paired by seed; res_b0 is the follow-up."""
    out = []
    for arm in ("plain", "res"):
        for seed in SEEDS:
            out.append({"name": f"{arm}_s{seed}", "arm": arm, "seed": seed})
    for seed in SEEDS:
        out.append({"name": f"res_b0_s{seed}", "arm": "res_b0", "seed": seed})
    return out


def run_cell(cell: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    result = out_dir / cell["name"] / "result.json"
    if result.exists():
        return json.loads(result.read_text(encoding="utf-8"))

    over = dict(BASE)
    over.update(ARMS[cell["arm"]])
    over["run.seed"] = cell["seed"]
    over["run.name"] = cell["name"]
    over["run.out_dir"] = str(out_dir)
    cfg = load_config(CONFIGS / "tiny_synth.yaml", overrides=over, cuda_available=False)

    started = time.time()
    trainer = Trainer(cfg, device=torch.device("cpu"))
    if trainer.maybe_resume():
        print(f"    resumed at step {trainer.step}/{cfg.optim.total_steps}", flush=True)
    trainer.run(quiet=True)

    val = CharShardLoader(cfg.data, seed=0, split="valid")
    ind = lm_induction_bpb(trainer.model, val, reference=reference_set(),
                           n_batches=200, batch_size=8, min_match=MIN_MATCH)
    agg = lm_memory_benefit(trainer.model, val, n_batches=16, batch_size=8)
    diag = diagnose_recurrence(trainer.model, val.batch(0, 2).tokens)

    payload = {
        "name": cell["name"], "arm": cell["arm"], "seed": cell["seed"],
        "final_ce": trainer.history[-1].ce,
        "bpb_novel": ind["induction_novel"],
        "bpb_common": ind["induction_common"],
        "bpb_other": ind["other"],
        "bpb_all": ind["all"],
        "n_novel": ind["n_novel"], "n_common": ind["n_common"],
        "n_other": ind["n_other"], "novel_fraction": ind["novel_fraction"],
        "by_distance": ind["by_distance"],
        "reach": ind["reach"], "min_match": ind["min_match"],
        "bpb_live": agg["bpb_live"], "bpb_dead": agg["bpb_dead"],
        "rho": diag.rho, "g_rec": diag.g_rec,
        "secs": time.time() - started,
        "config": cfg.to_dict(),
    }
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload            # checkpoint deliberately kept


def paired_p(deltas: List[float]) -> float:
    """One-sided paired permutation; the 1/2^n floor is printed beside it."""
    mean = sum(deltas) / len(deltas)
    signs = list(itertools.product([1, -1], repeat=len(deltas)))
    return sum(1 for s in signs
               if sum(a * b for a, b in zip(deltas, s)) / len(deltas) >= mean) / len(signs)


def summarise(out_dir: Path) -> None:
    rows = [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(out_dir.glob("*/result.json"))]
    if not rows:
        print("no results yet")
        return
    by: Dict[str, Dict[int, dict]] = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["seed"]] = r

    r0 = rows[0]
    print(f"\ninduction = {r0['min_match']}-char context repeats from beyond "
          f"reach {r0['reach']}; NOVEL = that substring is absent from training")
    print(f"{r0['n_novel']} novel, {r0['n_common']} common, {r0['n_other']} other "
          f"({r0['novel_fraction'] * 100:.2f}% novel)\n")
    print(f"{'cell':<16} {'NOVEL':>9} {'common':>9} {'other':>9} {'all':>9} {'rho':>7}")
    print("-" * 64)
    for arm in ("plain", "res", "res_b0"):
        for _, r in sorted(by.get(arm, {}).items()):
            print(f"{r['name']:<16} {r['bpb_novel']:>9.4f} {r['bpb_common']:>9.4f} "
                  f"{r['bpb_other']:>9.4f} {r['bpb_all']:>9.4f} {r['rho']:>7.3f}")

    print(f"\n{'arm':<10} {'n':>3} {'NOVEL':>9} {'common':>9} {'other':>9} {'all':>9}")
    print("-" * 52)
    for arm in ("plain", "res", "res_b0"):
        cs = by.get(arm, {})
        if not cs:
            continue
        n = len(cs)
        m = lambda k: sum(c[k] for c in cs.values()) / n  # noqa: E731
        print(f"{arm:<10} {n:>3} {m('bpb_novel'):>9.4f} {m('bpb_common'):>9.4f} "
              f"{m('bpb_other'):>9.4f} {m('bpb_all'):>9.4f}")
    print("lower is better everywhere")

    for arm in ("res", "res_b0"):
        shared = sorted(set(by.get("plain", {})) & set(by.get(arm, {})))
        if len(shared) < 2:
            continue
        print(f"\npaired by seed, {arm} minus plain ({len(shared)} seeds; "
              f"negative favours {arm}):")
        for label, key in (("NOVEL", "bpb_novel"), ("common", "bpb_common"),
                           ("other", "bpb_other"), ("all", "bpb_all")):
            d = [by[arm][s][key] - by["plain"][s][key] for s in shared]
            wins = sum(1 for x in d if x < 0)
            print(f"  {label:<8} mean {sum(d) / len(d):+.4f}  "
                  f"{arm} better on {wins}/{len(d)}  "
                  f"p = {paired_p([-x for x in d]):.4f} "
                  f"(floor {1 / 2 ** len(d):.3f})")

    print("\nNOVEL-INDUCTION BPB BY RETRIEVAL DISTANCE")
    print("the prediction: the gap grows with distance if the write rule cannot")
    print("carry an exact fact across a gap")
    buckets = rows[0]["by_distance"]
    print(f"{'arm':<10} " + "".join(f"{b['lo']}-{b['hi']:<7}" for b in buckets))
    print("-" * (10 + 10 * len(buckets)))
    for arm in ("plain", "res", "res_b0"):
        cs = by.get(arm, {})
        if not cs:
            continue
        vals = []
        for i in range(len(buckets)):
            xs = [c["by_distance"][i]["bpb"] for c in cs.values()
                  if c["by_distance"][i]["n"] > 0]
            vals.append(sum(xs) / len(xs) if xs else float("nan"))
        print(f"{arm:<10} " + "".join(f"{v:>9.4f} " for v in vals))
    print(f"{'n/cell':<10} " + "".join(f"{b['n']:>9} " for b in buckets))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="lm_ind_runs")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.summary:
        summarise(out_dir)
        return 0

    torch.set_num_threads(args.threads)
    todo = cells()
    print(f"{len(todo)} cells -> {out_dir}", flush=True)
    for i, cell in enumerate(todo, 1):
        if not (out_dir / cell["name"] / "result.json").exists():
            print(f"[{i}/{len(todo)}] {cell['name']} (~45 min)", flush=True)
        r = run_cell(cell, out_dir)
        print(f"[{i}/{len(todo)}] {r['name']:<16} NOVEL={r['bpb_novel']:.4f} "
              f"other={r['bpb_other']:.4f} all={r['bpb_all']:.4f} "
              f"rho={r['rho']:.3f} [{r['secs'] / 60:.0f} min]", flush=True)
    summarise(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
