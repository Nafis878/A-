"""PHASE 0 GATE: can deployed retrieval work at all?

Across 26 runs the best deployed accuracy is 0.077 against a chance of 0.0625.
The structural reason: Maglev trains only the teacher-forced pass, so the
recurrent chain it is DEPLOYED on is never exercised. No configuration has ever
trained the path it is evaluated on.

This trains through the recurrence (BPTT) at lambda=0 -- no consistency term at
all -- purely to establish whether the chain CAN carry a retrieval binding. It
is a positive control, not a proposed method.

Geometry: seq 48, W=4, L=2 -> stacked local reach L*(W-1) = 6. Every distance
(8, 12, 16, 24) lies beyond it, so nothing is solvable by local attention and
the SWA control sits at chance.

Ladder runs easiest-first so a failure localises: if even 2 values with no
distractors fails under BPTT, the architecture cannot carry retrieval through
its recurrence here, and that is the finding.
"""
import sys, json, time
sys.path.insert(0, r"c:/Users/UseR/Downloads/A")
import torch
from pathlib import Path
from rwc.config import load_config
from rwc.train import Trainer
from rwc.analysis import wilson_interval

ROOT = Path(r"C:/Users/UseR/AppData/Local/Temp/claude/c--Users-UseR-Downloads-A/33c40683-2cc9-4597-b751-2ee8ef01368d/scratchpad/exp")
BASE = {
    "data.task": "delayed_recall", "model.window": 4,
    "model.d_model": 128, "model.n_layers": 2,
    "model.n_heads": 4, "model.n_kv_heads": 2,
    "model.prefiller_pattern": "SL", "model.decoder_pattern": "SS",
    "data.seq_len": 48, "model.max_seq_len": 48,
    "data.distances": [8, 12, 16, 24],
    "data.loss_on_answer_only": True,
    "optim.total_steps": 2000, "optim.warmup": 100, "optim.lr": 3e-3,
    "optim.global_tokens_per_step": 32 * 48, "optim.micro_batch": 32,
    "run.ckpt_every": 100, "run.log_every": 10**6,
    "run.out_dir": str(ROOT / "runs"),
}

def run(name, **over):
    out = ROOT / "runs" / name / "result.json"
    if out.exists():
        return json.loads(out.read_text())
    o = dict(BASE); o.update(over); o["run.name"] = name
    cfg = load_config(r"c:/Users/UseR/Downloads/A/configs/tiny_synth.yaml",
                      overrides=o, cuda_available=False)
    t0 = time.time()
    t = Trainer(cfg, device=torch.device("cpu"))
    if t.maybe_resume():
        print(f"  [{name}] resumed at {t.step}", flush=True)
    t.run(quiet=True)
    res = {"name": name, "final_ce": t.history[-1].ce,
           "deployed": t.evaluate(n_per_distance=128, mode="deployed"),
           "parallel": t.evaluate(n_per_distance=128, mode="parallel"),
           "secs": time.time() - t0, "config": cfg.to_dict()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    return res

def show(r, chance):
    d = r["deployed"]; p = r["parallel"]
    dm = sum(v["acc"] for v in d.values())/len(d)
    pm = sum(v["acc"] for v in p.values())/len(p)
    n = sum(v["n"] for v in d.values())
    hi = wilson_interval(round(chance*n), n)[1]   # upper bound for pure chance
    verdict = "ABOVE CHANCE" if dm > hi else "at chance"
    per = "  ".join(f"D{k}:{v['acc']:.2f}" for k, v in d.items())
    print(f"{r['name']:<22} ce={r['final_ce']:.3f} par={pm:.3f} dep={dm:.3f} "
          f"(chance {chance:.3f}, 95% hi {hi:.3f}) {verdict} | {per} "
          f"[{r['secs']:.0f}s]", flush=True)

LADDER = [
    ("v2_d0",  {"data.n_values": 2,  "data.n_keys": 4,  "data.n_filler": 8,  "data.n_distractors": 0}, 0.5),
    ("v8_d0",  {"data.n_values": 8,  "data.n_keys": 8,  "data.n_filler": 8,  "data.n_distractors": 0}, 0.125),
    ("v16_d2", {"data.n_values": 16, "data.n_keys": 16, "data.n_filler": 16, "data.n_distractors": 2}, 0.0625),
]
print("PHASE 0: does the memory chain work when it is actually trained?\n", flush=True)
for tag, over, chance in LADDER:
    show(run(f"p0_bptt_{tag}", **over, **{"optim.train_mode": "bptt"}), chance)
print("\n--- teacher-forced controls, same tasks ---", flush=True)
for tag, over, chance in LADDER:
    show(run(f"p0_par_{tag}", **over), chance)
