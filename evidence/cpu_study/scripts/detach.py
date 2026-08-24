"""Does stop-grad on the target prevent the propagation collapse?

Measured: at lambda <= 0.03 memory is still read (zeroing it moves logits by
6-13) but rho(J) = d m_t / d m_{t-1} is ~0.01-0.12 -- read, never written
forward. One candidate cause: the consistency pull is symmetric, so Q is free to
drift toward a memory trajectory P can reproduce *locally*, and then neither
needs to carry anything across steps.

The earlier stop-grad test ran at lambda=1.0, where the gate had already closed
(g_rec 0.32, memory ablation 1e-3) -- nothing left to rescue. This runs it in the
window where the model still learns.
"""
import sys
sys.path.insert(0, r"C:/Users/UseR/AppData/Local/Temp/claude/c--Users-UseR-Downloads-A/33c40683-2cc9-4597-b751-2ee8ef01368d/scratchpad/exp")
from common import run, CHANCE

def show(r):
    d = r["deployed"]; p = r["parallel"]
    dm = sum(v["acc"] for v in d.values())/len(d)
    pm = sum(v["acc"] for v in p.values())/len(p)
    L = "LEARNED" if r["final_ce"] < 2.6 else "PLATEAU"
    print(f"{r['name']:<24} ce={r['final_ce']:.3f} {L} par={pm:.3f} dep={dm:.3f} "
          f"[{r['secs']:.0f}s]", flush=True)

print(f"chance={CHANCE:.4f}", flush=True)
print("no-detach refs: lam.03 par=.309 dep=.048 rho=.015 | lam.1 PLATEAU\n", flush=True)
for lam in (0.03, 0.1, 0.3):
    show(run(f"dt_lam{lam}", **{"loss.consistency": "uniform", "loss.lam": lam,
                                "loss.detach_target": True}))
