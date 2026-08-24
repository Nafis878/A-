import sys; sys.path.insert(0, r"C:/Users/UseR/AppData/Local/Temp/claude/c--Users-UseR-Downloads-A/33c40683-2cc9-4597-b751-2ee8ef01368d/scratchpad/exp")
from common import run, line, CHANCE, REACH, DISTANCES
print(f"local reach = {REACH}; distances {DISTANCES} are ALL memory-only")
print(f"chance = {CHANCE:.4f}\n", flush=True)
for name, over in [
    ("pilot_swa",            {"model.kind": "swa"}),
    ("pilot_maglev_lam0",    {}),
    ("pilot_uniform_lam1",   {"loss.consistency": "uniform", "loss.lam": 1.0}),
    ("pilot_uniform_lam1_dt",{"loss.consistency": "uniform", "loss.lam": 1.0,
                              "loss.detach_target": True}),
]:
    print(line(run(name, **over)), flush=True)
