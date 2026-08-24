import sys
sys.path.insert(0, r"C:/Users/UseR/AppData/Local/Temp/claude/c--Users-UseR-Downloads-A/33c40683-2cc9-4597-b751-2ee8ef01368d/scratchpad/exp")
from common import run, CHANCE
import json

def show(r):
    d = r["deployed"]; p = r["parallel"]
    dm = sum(v["acc"] for v in d.values())/len(d)
    pm = sum(v["acc"] for v in p.values())/len(p)
    learned = "LEARNED" if r["final_ce"] < 2.6 else "PLATEAU "
    print(f"{r['name']:<22} ce={r['final_ce']:.3f} {learned} par={pm:.3f} dep={dm:.3f} "
          f"gap={pm-dm:+.3f} cons={r['final_consistency']:.4f} [{r['secs']:.0f}s]", flush=True)

print(f"chance={CHANCE:.4f}   lam=0 gave par=0.840 dep=0.066; lam=1.0 never left the plateau\n", flush=True)
for lam in (0.03, 0.1, 0.3):
    show(run(f"lam{lam}_uniform", **{"loss.consistency": "uniform", "loss.lam": lam}))
for lam in (0.1, 0.3):
    show(run(f"lam{lam}_uniform_dt", **{"loss.consistency": "uniform", "loss.lam": lam,
                                        "loss.detach_target": True}))
