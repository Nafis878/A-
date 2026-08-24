"""Does an identity path make the memory chain learnable?

Established: the task is solvable (full attention 1.000, exact ce 0.0000), the
window-4 baseline is at chance, gradient reaches the planting site through the
closed loop (flat in D), and yet the memory chain never carries the binding --
neither teacher-forced nor under direct BPTT (ce 0.6932 vs ln(2)=0.6931).

Remaining candidate: Maglev recomputes m_t = decoder_norm(h_t) from scratch each
step, so MAINTAINING a memory means learning to reconstruct m_{t-1} exactly at
every position. memory_residual makes that free:
    m_t = f * m_{t-1} + (1 - f) * decoder_norm(h_t),  f = sigmoid(W_f h_t + b)
with b > 0 so f starts near 1 (identity), leaving only "when to write" to learn.

Run under BPTT so the chain being tested is the chain being trained.
"""
import sys, json, time, math
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
    d=r["deployed"]; p=r["parallel"]
    dm=sum(v["acc"] for v in d.values())/len(d); pm=sum(v["acc"] for v in p.values())/len(p)
    n=sum(v["n"] for v in d.values()); hi=wilson_interval(round(0.5*n),n)[1]
    verdict="ABOVE CHANCE" if dm>hi else "at chance"
    per="  ".join(f"D{k}:{v['acc']:.2f}" for k,v in d.items())
    print(f"{r['name']:<26} ce={r['final_ce']:.4f} par={pm:.3f} DEP={dm:.3f} "
          f"(hi {hi:.3f}) {verdict} | {per} [{r['secs']:.0f}s]", flush=True)

print(f"chance 0.5, ln(2)={math.log(2):.4f}.  BPTT without residual: ce 0.6932 = chance\n", flush=True)
# Control first: BPTT WITHOUT the identity path, re-run under the current code
# so the comparison does not straddle the cleanup. Previously measured at
# ce 0.694, deployed 0.494 vs chance 0.500 (Wilson hi 0.543) -- at chance.
show(go("r_bptt_plain", **{"optim.train_mode":"bptt"}))
show(go("r_bptt_res",   **{"optim.train_mode":"bptt","model.memory_residual":True}))
show(go("r_bptt_res_b4",**{"optim.train_mode":"bptt","model.memory_residual":True,
                           "model.memory_gate_bias":4.0}))
show(go("r_par_res",    **{"model.memory_residual":True}))
