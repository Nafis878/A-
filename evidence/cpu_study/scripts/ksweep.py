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
print("k=1 refs:  lam0 par=.840 dep=.066 | lam.03 par=.309 dep=.048 | lam>=.1 PLATEAU\n", flush=True)

# Does closed-loop unrolling lift deployed accuracy in the one viable lambda window?
for k in (2, 4, 8):
    show(run(f"cl_k{k}_lam0.03", **{"loss.consistency": "uniform", "loss.lam": 0.03,
                                    "loss.unroll_k": k}))
# Does it make a larger lambda survivable, where the one-step loss plateaus?
for lam in (0.1, 0.3):
    show(run(f"cl_k4_lam{lam}", **{"loss.consistency": "uniform", "loss.lam": lam,
                                   "loss.unroll_k": 4}))
