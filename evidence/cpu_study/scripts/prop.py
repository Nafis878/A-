"""Does explicitly regularising propagation rescue the consistency loss?

Baselines to beat, all measured:
  lam 0     rho .728  par .840  dep .066   (chain live, memory uncalibrated)
  lam .03   rho .015  par .309  dep .048   (WRITE SEVERED)
  lam .03 + stop-grad  rho .142  par .189  (partial rescue, ~10x on rho)
  lam .1    rho .002  par .064              (CHANNEL CLOSED, plateau)

lam=0.1 is the sharp test: it plateaus outright without help, so any recovery
there is unambiguous rather than a shift inside seed noise.
"""
import sys
sys.path.insert(0, r"C:/Users/UseR/AppData/Local/Temp/claude/c--Users-UseR-Downloads-A/33c40683-2cc9-4597-b751-2ee8ef01368d/scratchpad/exp")
from common import run, CHANCE

def show(r):
    d = r["deployed"]; p = r["parallel"]
    dm = sum(v["acc"] for v in d.values())/len(d)
    pm = sum(v["acc"] for v in p.values())/len(p)
    L = "LEARNED" if r["final_ce"] < 2.6 else "PLATEAU"
    print(f"{r['name']:<22} ce={r['final_ce']:.3f} {L} par={pm:.3f} dep={dm:.3f} "
          f"[{r['secs']:.0f}s]", flush=True)

print(f"chance={CHANCE:.4f}\n", flush=True)
for lam, w in ((0.1, 0.1), (0.03, 0.1), (0.1, 0.5)):
    show(run(f"pp_lam{lam}_w{w}", **{"loss.consistency": "uniform", "loss.lam": lam,
                                     "loss.propagation_weight": w}))
