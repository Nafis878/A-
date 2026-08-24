import sys
sys.path.insert(0, r"C:/Users/UseR/AppData/Local/Temp/claude/c--Users-UseR-Downloads-A/33c40683-2cc9-4597-b751-2ee8ef01368d/scratchpad/exp")
from common import run, CHANCE

def show(r):
    d = r["deployed"]; p = r["parallel"]
    dm = sum(v["acc"] for v in d.values())/len(d)
    pm = sum(v["acc"] for v in p.values())/len(p)
    L = "LEARNED" if r["final_ce"] < 2.6 else "PLATEAU"
    per = "  ".join(f"D{k}:{v['acc']:.2f}" for k, v in d.items())
    print(f"{r['name']:<24} ce={r['final_ce']:.3f} {L} par={pm:.3f} dep={dm:.3f} "
          f"gap={pm-dm:+.3f} | {per} [{r['secs']:.0f}s]", flush=True)

print(f"chance={CHANCE:.4f}   baseline k=1: lam0 par=.840 dep=.066 | lam.03 par=.309 dep=.048\n", flush=True)
for lam in (0.03, 0.1, 0.3):
    show(run(f"cl_k4_lam{lam}", **{"loss.consistency": "uniform", "loss.lam": lam,
                                   "loss.unroll_k": 4}))
