"""Does the write-rule failure appear in a language model?

Every result so far is on a synthetic retrieval probe. The paper being rebutted
is about a 435M-parameter LM, so the first question any reviewer asks is whether
any of this happens in language. This answers it on CPU, at the width ladder's
base geometry -- d=128, L=2, W=4, stacked local reach 6 -- so the LM result and
the synthetic result are directly comparable.

Corpus is text8 (cleaned Wikipedia, 27-character alphabet). One character is one
byte, so bits-per-byte is bits-per-character and comparable to the char-LM
literature.

WHAT IS AND IS NOT CLAIMED. A 6-character window is pathological for text, so
absolute BPB here is poor by construction and is not comparable to a real
char-LM. The claim is comparative: holding architecture, data, seed and step
budget fixed, does Maglev's write rule buy the model anything through its memory,
and does the identity path buy more?

Raw BPB is deliberately NOT the headline. Character prediction is dominated by
local statistics, so two models whose memory channels differ enormously can post
nearly the same aggregate BPB -- reporting it alone would hide the effect. The
quantity that isolates the channel is the difference between the deployed model
and the same weights with memory forced to zero (`mem_in_override`), which
leaves an exact window-W, L-layer local model. Whatever separates them is what
memory contributed and nothing else. See rwc/evaluate.py::lm_memory_benefit.

Resolved by position, that difference is the LM analogue of the synthetic
distance axis: a model whose memory is dead can improve only until its context
fills the local reach and must then flatten, while a live channel keeps
improving.

A null result here is a real outcome and is reported as found.

Usage:
    python scripts/lm_study.py --out-dir lm_runs
    python scripts/lm_study.py --out-dir lm_runs --summary
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

from rwc.analysis import diagnose_recurrence, memory_information  # noqa: E402
from rwc.config import load_config  # noqa: E402
from rwc.data.lm import CharShardLoader  # noqa: E402
from rwc.evaluate import lm_memory_benefit  # noqa: E402
from rwc.train import Trainer  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

BASE: Dict[str, Any] = {
    "data.kind": "lm",
    "data.shard_dir": "lm_data",
    "data.seq_len": 256,
    "data.val_tokens": 1_000_000,
    "model.max_seq_len": 256,
    "model.vocab_size": 27,          # text8's alphabet exactly; 64 would put
                                     # 37 impossible ids in every softmax
    "model.d_model": 128, "model.n_layers": 2, "model.window": 4,
    "model.n_heads": 4, "model.n_kv_heads": 2,
    "model.prefiller_pattern": "SL", "model.decoder_pattern": "SS",
    "optim.total_steps": 1500, "optim.warmup": 100, "optim.lr": 3e-3,
    "optim.train_mode": "bptt",      # train the path that is evaluated
    "optim.global_tokens_per_step": 16 * 256, "optim.micro_batch": 16,
    "run.ckpt_every": 100, "run.log_every": 500,
}
ARMS = {"plain": {}, "res": {"model.memory_residual": True}}
SEEDS = range(5)
MIN_FREE_GB = 0.5


def cells() -> List[Dict[str, Any]]:
    """Arms paired within a seed, so a truncated run still compares like to like."""
    out = []
    for seed in SEEDS:
        for arm, extra in ARMS.items():
            over = {"run.seed": seed}
            over.update(extra)
            out.append({"name": f"lm_{arm}_s{seed}", "arm": arm, "seed": seed,
                        "over": over})
    return out


def _free_gb(p: Path) -> float:
    return shutil.disk_usage(p).free / 1e9


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

    # Held-out split: a different file from the training text, never overlapping.
    val = CharShardLoader(cfg.data, seed=0, split="valid")
    benefit = lm_memory_benefit(trainer.model, val, n_batches=16, batch_size=8)

    tokens = val.batch(0, 2).tokens
    diag = diagnose_recurrence(trainer.model, tokens)
    info = memory_information(trainer.model, tokens)

    payload = {
        "name": cell["name"], "arm": cell["arm"], "seed": cell["seed"],
        "corpus": val.source, "vocab_size": val.vocab_size,
        "n_params": sum(p.numel() for n, p in trainer.model.named_parameters()
                        if "embed" not in n and "lm_head" not in n),
        "final_ce": trainer.history[-1].ce,
        "bpb_live": benefit["bpb_live"],
        "bpb_dead": benefit["bpb_dead"],
        "benefit": benefit["benefit"],
        "by_position": benefit["by_position"],
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


def paired_permutation(deltas: List[float], n_perm: int = 20000) -> float:
    """One-sided paired permutation test on per-seed differences.

    Seeds are the unit of independence. With five pairs the sign-flip null has
    only 2^5 = 32 arrangements, so the smallest attainable p is 1/32 = 0.031 --
    reported honestly rather than hidden behind a t-test's continuous p-value.
    """
    import itertools

    observed = sum(deltas) / len(deltas)
    signs = list(itertools.product([1, -1], repeat=len(deltas)))
    ge = sum(
        1 for s in signs
        if sum(x * y for x, y in zip(deltas, s)) / len(deltas) >= observed
    )
    return ge / len(signs)


def summarise(out_dir: Path) -> None:
    rows = [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(out_dir.glob("*/result.json"))]
    if not rows:
        print("no results yet")
        return

    print(f"\ncorpus {rows[0]['corpus']}, vocab {rows[0]['vocab_size']}, "
          f"{rows[0]['n_params'] / 1e6:.2f}M non-emb, local reach 6")
    print(f"uniform baseline = log2({rows[0]['vocab_size']}) = "
          f"{__import__('math').log2(rows[0]['vocab_size']):.3f} bpc\n")
    print(f"{'cell':<16} {'bpb live':>9} {'bpb dead':>9} {'memory buys':>12} "
          f"{'rho':>7} {'spread':>9}")
    print("-" * 68)
    for r in sorted(rows, key=lambda x: (x["arm"], x["seed"])):
        print(f"{r['name']:<16} {r['bpb_live']:>9.4f} {r['bpb_dead']:>9.4f} "
              f"{r['benefit']:>+12.4f} {r['rho']:>7.3f} {r['spread']:>9.5f}")

    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], {})[r["seed"]] = r
    print(f"\n{'arm':<16} {'n':>3} {'mean bits bought by memory':>28}")
    print("-" * 50)
    for arm in ("plain", "res"):
        cells_ = by_arm.get(arm, {})
        if cells_:
            vals = [c["benefit"] for c in cells_.values()]
            print(f"{arm:<16} {len(vals):>3} {sum(vals) / len(vals):>+28.4f}")

    shared = sorted(set(by_arm.get("plain", {})) & set(by_arm.get("res", {})))
    if len(shared) >= 2:
        deltas = [by_arm["res"][s]["benefit"] - by_arm["plain"][s]["benefit"]
                  for s in shared]
        print(f"\npaired by seed, identity path minus write rule ({len(shared)} seeds):")
        for s, d in zip(shared, deltas):
            print(f"  seed {s}: {d:+.4f} bits")
        print(f"  mean {sum(deltas) / len(deltas):+.4f} bits   "
              f"one-sided paired permutation p = {paired_permutation(deltas):.4f}")
        print(f"  (floor with {len(shared)} seeds is 1/{2 ** len(shared)} = "
              f"{1 / 2 ** len(shared):.3f})")

    print(f"\nbits bought by memory, BY POSITION -- reach is 6, so a dead")
    print(f"channel must flatten past bucket 0-8 and a live one keeps gaining")
    pos = rows[0]["by_position"]
    print(f"{'arm':<8} " + "".join(f"{b['lo']}-{b['hi']:<7}" for b in pos))
    print("-" * (8 + 10 * len(pos)))
    for arm in ("plain", "res"):
        cells_ = by_arm.get(arm, {})
        if not cells_:
            continue
        n = len(cells_)
        means = [
            sum(c["by_position"][i]["benefit"] for c in cells_.values()) / n
            for i in range(len(pos))
        ]
        print(f"{arm:<8} " + "".join(f"{m:>+9.4f} " for m in means))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="lm_runs")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.summary:
        summarise(out_dir)
        return 0

    torch.set_num_threads(8)
    todo = cells()
    print(f"{len(todo)} cells -> {out_dir}   free {_free_gb(out_dir):.1f} GB",
          flush=True)
    for i, cell in enumerate(todo, 1):
        cached = (out_dir / cell["name"] / "result.json").exists()
        if not cached:
            if _free_gb(out_dir) < MIN_FREE_GB + 0.1:
                print(f"[{i}] SKIP {cell['name']}: low disk", flush=True)
                continue
            print(f"[{i}/{len(todo)}] {cell['name']} (~35 min)", flush=True)
        r = run_cell(cell, out_dir)
        print(f"[{i}/{len(todo)}] {r['name']:<16} bpb={r['bpb_live']:.4f} "
              f"dead={r['bpb_dead']:.4f} memory_buys={r['benefit']:+.4f} "
              f"rho={r['rho']:.3f} [{r['secs'] / 60:.0f} min]", flush=True)
    summarise(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
