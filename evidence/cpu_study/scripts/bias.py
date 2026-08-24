"""How sensitive is the fix to the identity-gate bias INITIALISATION?

Earlier the bias was added at every forward pass, pinning the gate where the
model could not move off it. b=2.0 gave deployed 1.000 and b=4.0 gave 0.494 --
chance -- because at f=0.98 only ~2% of the state can change per step and there
is no write bandwidth. That made it a hyperparameter we had introduced.

It is now an initialisation of a learnable bias (the standard LSTM forget-gate
trick). This sweep asks whether the fix still depends on the value: if training
can move off a bad init, the sensitivity is gone and there is no new
hyperparameter to defend.
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
    "data.task":"delayed_recall","model.window":4,"model.d_model":128,"model.n_layers":2,
    "model.n_heads":4,"model.n_kv_heads":2,"model.prefiller_pattern":"SL",
    "model.decoder_pattern":"SS","data.seq_len":48,"model.max_seq_len":48,
    "data.distances":[8,12,16,24],"data.loss_on_answer_only":True,
    "data.n_values":2,"data.n_keys":4,"data.n_filler":8,"data.n_distractors":0,
    "optim.total_steps":1500,"optim.warmup":100,"optim.lr":3e-3,
    "optim.global_tokens_per_step":32*48,"optim.micro_batch":32,
    "optim.train_mode":"bptt","model.memory_residual":True,
    "run.ckpt_every":100,"run.log_every":10**6,"run.out_dir":str(ROOT/"runs"),
}
def go(name, **over):
    out = ROOT/"runs"/name/"result.json"
    if out.exists(): return json.loads(out.read_text())
    o=dict(BASE); o.update(over); o["run.name"]=name
    cfg=load_config(r"c:/Users/UseR/Downloads/A/configs/tiny_synth.yaml",overrides=o,cuda_available=False)
    t0=time.time(); t=Trainer(cfg,device=torch.device("cpu")); t.maybe_resume(); t.run(quiet=True)
    res={"name":name,"final_ce":t.history[-1].ce,
         "deployed":t.evaluate(n_per_distance=128,mode="deployed"),
         "parallel":t.evaluate(n_per_distance=128,mode="parallel"),
         "secs":time.time()-t0,"config":cfg.to_dict()}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(res,indent=2))
    return res

print(f"{'init b':>6} {'f at init':>10} {'ce':>8} {'deployed':>9}  verdict", flush=True)
for bias in (0.0, 2.0, 4.0, 6.0):
    r = go(f"bl_bias{bias}", **{"model.memory_gate_bias": bias})
    d=r["deployed"]; dm=sum(v["acc"] for v in d.values())/len(d)
    n=sum(v["n"] for v in d.values()); hi=wilson_interval(round(0.5*n),n)[1]
    f0=1/(1+math.exp(-bias))
    print(f"{bias:>6} {f0:>10.3f} {r['final_ce']:>8.4f} {dm:>9.3f}  "
          f"{'WORKS' if dm>hi else 'at chance'} [{r['secs']:.0f}s]", flush=True)
