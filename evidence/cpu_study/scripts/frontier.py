"""Low-lambda frontier for k=1 (Maglev) and k=4 (CL-RWC).

Fixed-lambda comparison is confounded: the k>=2 terms penalise the closed-loop
residual, measured at 1.2-4x the one-step residual, so identical lambda pulls
harder. Observed effective multiplier is roughly proportional to k --
k=4 @ lam .03 plateaus exactly like k=1 @ lam >= .1. So the comparison has to be
between the best achievable point on each frontier, which means covering the low
end for k=4.
"""
import sys
sys.path.insert(0, r"C:/Users/UseR/AppData/Local/Temp/claude/c--Users-UseR-Downloads-A/33c40683-2cc9-4597-b751-2ee8ef01368d/scratchpad/exp")
from common import run, CHANCE

def show(r):
    d = r["deployed"]; p = r["parallel"]
    dm = sum(v["acc"] for v in d.values())/len(d)
    pm = sum(v["acc"] for v in p.values())/len(p)
    L = "LEARNED" if r["final_ce"] < 2.6 else "PLATEAU"
    per = "  ".join(f"D{k}:{v['acc']:.2f}" for k, v in d.items())
    print(f"{r['name']:<20} ce={r['final_ce']:.3f} {L} par={pm:.3f} dep={dm:.3f} "
          f"| {per} [{r['secs']:.0f}s]", flush=True)

print(f"chance={CHANCE:.4f}", flush=True)
print("have: k=1 lam0 par=.840 dep=.066 | k=1 lam.03 par=.309 dep=.048", flush=True)
print("      k=2 lam.03 par=.141 dep=.057 | k=4 lam.03 PLATEAU\n", flush=True)
# k=4 at the lambda where its effective strength should match k=1 @ .03 or below
for lam in (0.003, 0.01):
    show(run(f"cl_k4_lam{lam}", **{"loss.consistency": "uniform", "loss.lam": lam,
                                   "loss.unroll_k": 4}))
# matched k=1 points so the frontier has partners at the same lambda
for lam in (0.003, 0.01):
    show(run(f"lam{lam}_uniform", **{"loss.consistency": "uniform", "loss.lam": lam}))
