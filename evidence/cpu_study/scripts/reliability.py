"""Success RATE of the write rule, with and without the identity path.

A control run at seed 2 reached deployed 1.000 with residual=False, which the
earlier single-seed cells did not. So Maglev's write rule CAN learn this task --
it usually does not. The claim is therefore about optimisation reliability, not
architectural impossibility, and reliability is a rate that needs seeds.

Standing record before this sweep:
    no identity path   seeds 0,0,1,2 -> 0.506, 0.494, 0.482, 1.000   (1/4)
    identity path      seeds 0,1     -> 1.000, 1.000                 (2/2)

Outcome per run is categorical (deployed clears the Wilson upper bound for
chance, or does not), so this reports k/n with a Wilson interval on the rate.
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
    "data.task":"delayed_recall","model.window":4,"model.d_model":128,"model.n_layers":2,
    "model.n_heads":4,"model.n_kv_heads":2,"model.prefiller_pattern":"SL",
    "model.decoder_pattern":"SS","data.seq_len":48,"model.max_seq_len":48,
    "data.distances":[8,12,16,24],"data.loss_on_answer_only":True,
    "data.n_values":2,"data.n_keys":4,"data.n_filler":8,"data.n_distractors":0,
    "optim.total_steps":1500,"optim.warmup":100,"optim.lr":3e-3,
    "optim.global_tokens_per_step":32*48,"optim.micro_batch":32,
    "optim.train_mode":"bptt",
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

def works(r):
    d=r["deployed"]; dm=sum(v["acc"] for v in d.values())/len(d)
    n=sum(v["n"] for v in d.values())
    return dm > wilson_interval(round(0.5*n),n)[1], dm

for seed in range(3, 10):
    r = go(f"rel_plain_s{seed}", **{"run.seed": seed})
    ok, dm = works(r)
    print(f"rel_plain_s{seed:<2} deployed {dm:.3f}  {'WORKS' if ok else 'fails'} [{r['secs']:.0f}s]", flush=True)
for seed in range(2, 10):
    r = go(f"rel_res_s{seed}", **{"run.seed": seed, "model.memory_residual": True})
    ok, dm = works(r)
    print(f"rel_res_s{seed:<4} deployed {dm:.3f}  {'WORKS' if ok else 'fails'} [{r['secs']:.0f}s]", flush=True)
