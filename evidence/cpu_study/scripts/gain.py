"""Land rho inside (0, 1), where the theory says deployed error is bounded.

delta_t = e_t + J_t delta_{t-1}.  rho < 1 gives ||delta|| <= ||e||/(1-rho);
rho >= 1 leaves it unbounded. Measured so far, both failure modes give chance
deployed accuracy for opposite reasons:

  uniform L2 alone     rho 0.015   memory carries nothing
  + propagation (1.0)  rho 1.486   memory carries, errors compound

The penalty targets ||J v|| for random unit v -- an RMS gain, not the spectral
radius, and rho can exceed it substantially. So the target has to come down to
put rho below 1. If deployed accuracy rises here, the account is complete; if
rho lands in (0,1) and deployed still does not move, the bound is satisfied and
something else is binding, which is worth knowing.
"""
import sys
sys.path.insert(0, r"C:/Users/UseR/AppData/Local/Temp/claude/c--Users-UseR-Downloads-A/33c40683-2cc9-4597-b751-2ee8ef01368d/scratchpad/exp")
from common import run

def show(r):
    p = r["parallel"]; d = r["deployed"]
    pm = sum(v["acc"] for v in p.values())/len(p)
    dm = sum(v["acc"] for v in d.values())/len(d)
    L = "LEARNED" if r["final_ce"] < 2.6 else "PLATEAU"
    per = "  ".join(f"D{k}:{v['acc']:.2f}" for k, v in d.items())
    print(f"{r['name']:<24} ce={r['final_ce']:.3f} {L} par={pm:.3f} dep={dm:.3f} "
          f"| {per} [{r['secs']:.0f}s]", flush=True)

print("refs: lam.03 alone rho=.015 par=.309 dep=.048 | +pp(1.0) rho=1.486 "
      "par=.833 dep=.048\n", flush=True)
for tgt in (0.5, 0.25, 0.7):
    show(run(f"gn_lam0.03_t{tgt}", **{"loss.consistency": "uniform", "loss.lam": 0.03,
                                      "loss.propagation_weight": 0.1,
                                      "loss.propagation_target": tgt}))
