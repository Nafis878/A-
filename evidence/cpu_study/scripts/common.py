"""Shared config for the CPU experiment.

Geometry chosen so the distance axis is genuinely memory-dependent: W=8, L=2
gives stacked local reach L*(W-1) = 14, and every probe distance (16..96) lies
beyond it. In the A100 grid (W=64, L=4, reach 252) eight of ten buckets were
inside the reach and solved by local attention alone, which is why C3 had no
range to show a slope over.
"""
import sys, time, json
sys.path.insert(0, r"c:/Users/UseR/Downloads/A")
import torch
from pathlib import Path
from rwc.config import load_config
from rwc.train import Trainer

ROOT = Path(r"C:/Users/UseR/AppData/Local/Temp/claude/c--Users-UseR-Downloads-A/33c40683-2cc9-4597-b751-2ee8ef01368d/scratchpad/exp")
DISTANCES = [16, 24, 32, 48, 64, 96]
REACH = 2 * (8 - 1)
CHANCE = 1 / 16
STEPS = 2000

BASE = {
    "data.task": "delayed_recall",
    "model.window": 8,
    "model.d_model": 128, "model.n_layers": 2,
    "model.n_heads": 4, "model.n_kv_heads": 2,
    "model.prefiller_pattern": "SL", "model.decoder_pattern": "SS",
    "data.seq_len": 128, "model.max_seq_len": 128,
    "data.distances": DISTANCES,
    "data.loss_on_answer_only": True,
    "data.n_distractors": 2,
    "optim.total_steps": STEPS, "optim.warmup": 100, "optim.lr": 3e-3,
    "optim.global_tokens_per_step": 32 * 128, "optim.micro_batch": 32,
    # Checkpoint often enough that an interrupted cell resumes rather than
    # restarting. Cadence is excluded from the config hash, so turning this on
    # does not orphan checkpoints already written.
    "run.ckpt_every": 100, "run.log_every": 10**6,
    "run.out_dir": str(ROOT / "runs"),
}


def run(name, *, seed=0, n_per_distance=128, **over):
    out = ROOT / "runs" / name / "result.json"
    if out.exists():
        return json.loads(out.read_text())
    o = dict(BASE); o.update(over); o["run.name"] = name; o["run.seed"] = seed
    cfg = load_config(r"c:/Users/UseR/Downloads/A/configs/tiny_synth.yaml",
                      overrides=o, cuda_available=False)
    t0 = time.time()
    t = Trainer(cfg, device=torch.device("cpu"))
    resumed = t.maybe_resume()
    if resumed:
        print(f"  [{name}] resumed at step {t.step}/{cfg.optim.total_steps}", flush=True)
    t.run(quiet=True)
    res = {
        "name": name, "seed": seed,
        "final_ce": t.history[-1].ce,
        "final_consistency": t.history[-1].consistency,
        "deployed": t.evaluate(n_per_distance=n_per_distance, mode="deployed"),
        "parallel": t.evaluate(n_per_distance=n_per_distance, mode="parallel"),
        "secs": time.time() - t0,
        "config": cfg.to_dict(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    return res


def line(res):
    d = res["deployed"]
    per = "  ".join(f"D{k}:{v['acc']:.2f}" for k, v in d.items())
    mean = sum(v["acc"] for v in d.values()) / len(d)
    return f"{res['name']:<26} ce={res['final_ce']:.3f} mean={mean:.3f} ({mean/CHANCE:4.1f}x)  {per}  [{res['secs']:.0f}s]"
