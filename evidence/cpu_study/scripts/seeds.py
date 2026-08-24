"""Seed replication for the headline claim.

lambda=0.1 + uniform L2 plateaus at ln(16) and is indistinguishable from the
memory-free baseline. Adding the propagation penalty took rho from 0.002 to
0.680 and turned the plateau into learning. Plateau-vs-learned is a categorical
state change rather than a shift inside the noise band, but it rests on one seed
so far, so this replicates both arms across seeds.
"""
import sys
sys.path.insert(0, r"C:/Users/UseR/AppData/Local/Temp/claude/c--Users-UseR-Downloads-A/33c40683-2cc9-4597-b751-2ee8ef01368d/scratchpad/exp")
from common import run

def show(r):
    p = r["parallel"]; d = r["deployed"]
    pm = sum(v["acc"] for v in p.values())/len(p)
    dm = sum(v["acc"] for v in d.values())/len(d)
    L = "LEARNED" if r["final_ce"] < 2.6 else "PLATEAU"
    print(f"{r['name']:<26} ce={r['final_ce']:.3f} {L} par={pm:.3f} dep={dm:.3f} "
          f"[{r['secs']:.0f}s]", flush=True)

# 3v3 hits the exact permutation floor of 2/20 = 0.100 -- the observed split is
# already maximally extreme and cannot reach 0.05. 5v5 floors at 2/252 = 0.008.
for seed in (1, 2, 3, 4):
    show(run(f"sd_ctl_lam0.1_s{seed}", seed=seed,
             **{"loss.consistency": "uniform", "loss.lam": 0.1}))
    show(run(f"sd_pp_lam0.1_s{seed}", seed=seed,
             **{"loss.consistency": "uniform", "loss.lam": 0.1,
                "loss.propagation_weight": 0.1}))
