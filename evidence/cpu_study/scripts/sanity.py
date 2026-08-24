"""Is the Phase 0 task solvable AT ALL, by anything?

BPTT sits at ln(2) on the easiest rung (2 values, no distractors). Before
concluding the recurrence cannot carry retrieval, rule out a broken task: a
FULL-ATTENTION model sees every token and should solve this trivially. If it
cannot, the config is wrong, not the architecture.
"""
import sys, json, time
sys.path.insert(0, r"c:/Users/UseR/Downloads/A")
import torch
from pathlib import Path
from rwc.config import load_config
from rwc.train import Trainer

ROOT = Path(r"C:/Users/UseR/AppData/Local/Temp/claude/c--Users-UseR-Downloads-A/33c40683-2cc9-4597-b751-2ee8ef01368d/scratchpad/exp")
BASE = {
    # needle generates identically to delayed_recall; it simply carries no
    # reach constraint, so the wide-window control is expressible.
    "data.task": "needle", "model.window": 4,
    "model.d_model": 128, "model.n_layers": 2,
    "model.n_heads": 4, "model.n_kv_heads": 2,
    "model.prefiller_pattern": "SL", "model.decoder_pattern": "SS",
    "data.seq_len": 48, "model.max_seq_len": 48,
    "data.distances": [8, 12, 16, 24],
    "data.loss_on_answer_only": True,
    "data.n_values": 2, "data.n_keys": 4, "data.n_filler": 8, "data.n_distractors": 0,
    "optim.total_steps": 600, "optim.warmup": 50, "optim.lr": 3e-3,
    "optim.global_tokens_per_step": 32 * 48, "optim.micro_batch": 32,
    "run.ckpt_every": 200, "run.log_every": 10**6, "run.out_dir": str(ROOT / "runs"),
}
def go(name, **over):
    out = ROOT / "runs" / name / "result.json"
    if out.exists(): return json.loads(out.read_text())
    o = dict(BASE); o.update(over); o["run.name"] = name
    cfg = load_config(r"c:/Users/UseR/Downloads/A/configs/tiny_synth.yaml",
                      overrides=o, cuda_available=False)
    t0=time.time(); t = Trainer(cfg, device=torch.device("cpu")); t.maybe_resume(); t.run(quiet=True)
    res = {"name": name, "final_ce": t.history[-1].ce,
           "deployed": t.evaluate(n_per_distance=128, mode="deployed"),
           "parallel": t.evaluate(n_per_distance=128, mode="parallel"),
           "secs": time.time()-t0, "config": cfg.to_dict()}
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=2))
    return res

import math
print(f"chance = 0.5, chance CE = ln(2) = {math.log(2):.4f}\n", flush=True)
for name, over in [
    ("s_full",   {"model.kind": "full"}),                    # sees everything
    ("s_swa",    {"model.kind": "swa"}),                     # window 4 only
    ("s_fullwin",{"model.kind": "swa", "model.window": 48}), # window = whole seq
]:
    r = go(name, **over)
    p = r["parallel"]; pm = sum(v["acc"] for v in p.values())/len(p)
    per = "  ".join(f"D{k}:{v['acc']:.2f}" for k, v in p.items())
    print(f"{name:<10} ce={r['final_ce']:.4f} acc={pm:.3f} | {per} [{r['secs']:.0f}s]", flush=True)
