"""The evidence a reviewer will ask for, one cell per question.

Established so far, single seed, easiest rung:
    no identity path   rho 0.004  deployed 0.500  (memory contributes +0.000)
    identity path      rho 1.001  deployed 1.000  (memory contributes +0.500)

Open questions this answers:
  Q1 does it replicate?          -> seeds 1, 2 for both arms
  Q2 does it generalise?         -> 8 and 16 values, with distractors
  Q3 does it hold further out?   -> distances to 32, well past reach 6
  Q4 does it fix the LOSS too?   -> residual + teacher forcing + consistency,
                                    which is the setting the collapse lives in
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
    "data.n_values": 2, "data.n_keys": 4, "data.n_filler": 8, "data.n_distractors": 0,
    "optim.total_steps": 1500, "optim.warmup": 100, "optim.lr": 3e-3,
    "optim.global_tokens_per_step": 32 * 48, "optim.micro_batch": 32,
    "run.ckpt_every": 100, "run.log_every": 10**6, "run.out_dir": str(ROOT / "runs"),
}
def go(name, **over):
    out = ROOT / "runs" / name / "result.json"
    if out.exists(): return json.loads(out.read_text())
    o = dict(BASE); o.update(over); o["run.name"] = name
    cfg = load_config(r"c:/Users/UseR/Downloads/A/configs/tiny_synth.yaml",
                      overrides=o, cuda_available=False)
    t0=time.time(); t=Trainer(cfg, device=torch.device("cpu")); t.maybe_resume(); t.run(quiet=True)
    res={"name":name,"final_ce":t.history[-1].ce,
         "deployed":t.evaluate(n_per_distance=128,mode="deployed"),
         "parallel":t.evaluate(n_per_distance=128,mode="parallel"),
         "secs":time.time()-t0,"config":cfg.to_dict()}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(res,indent=2))
    return res

def show(r):
    d=r["deployed"]; cfg=r["config"]
    dm=sum(v["acc"] for v in d.values())/len(d)
    n=sum(v["n"] for v in d.values()); ch=1/cfg["data"]["n_values"]
    hi=wilson_interval(round(ch*n),n)[1]
    tag="ABOVE CHANCE" if dm>hi else "at chance"
    per="  ".join(f"D{k}:{v['acc']:.2f}" for k,v in d.items())
    print(f"{r['name']:<24} ce={r['final_ce']:.4f} DEP={dm:.3f} "
          f"(ch {ch:.3f}, hi {hi:.3f}) {tag} | {per} [{r['secs']:.0f}s]", flush=True)

RES  = {"model.memory_residual": True}
BPTT = {"optim.train_mode": "bptt"}
HARD = {"data.n_values": 8, "data.n_keys": 8, "data.n_filler": 16, "data.n_distractors": 2}
HARDER = {"data.n_values": 16, "data.n_keys": 16, "data.n_filler": 16, "data.n_distractors": 2}
FAR  = {"data.distances": [8, 16, 24, 32]}

print("Q1 replication\n", flush=True)
for s in (1, 2):
    show(go(f"f_plain_s{s}", **BPTT, **{"run.seed": s}))
    show(go(f"f_res_s{s}",   **BPTT, **RES, **{"run.seed": s}))
print("\nQ2 harder tasks (with distractors)\n", flush=True)
show(go("f_plain_v8",  **BPTT, **HARD))
show(go("f_res_v8",    **BPTT, **RES, **HARD))
show(go("f_res_v16",   **BPTT, **RES, **HARDER))
print("\nQ3 longer distances\n", flush=True)
show(go("f_res_far",   **BPTT, **RES, **FAR))
print("\nQ4 does the identity path also fix the consistency loss?\n", flush=True)
show(go("f_par_lam01_plain", **{"loss.consistency":"uniform","loss.lam":0.1}))
show(go("f_par_lam01_res",   **RES, **{"loss.consistency":"uniform","loss.lam":0.1}))
