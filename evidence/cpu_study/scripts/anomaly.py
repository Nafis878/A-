"""Reproduce Maglev's published lambda anomaly and test the mechanism.

Maglev (arXiv:2608.02870) reports: "A larger lambda improves the
separate-parameter model, but hurts the shared model on several downstream
tasks", attributed to capacity -- "when Q and P share most parameters, an
overly strong memory-consistency term can constrain the decoder representation".

The recurrence-collapse mechanism predicts something sharper and different:
SHARED should sever the memory write path at a LOWER lambda than SEPARATE.
When Q and P share weights, the degenerate solution -- make m' locally
computable, so P can reproduce it without carrying anything -- is reachable by
moving one set of weights. With separate stacks Q must be dragged there against
its own CE objective, so the collapse should be delayed.

Capacity predicts a magnitude difference. Collapse predicts a difference in
WHERE rho(dm_t/dm_{t-1}) falls off. Those are distinguishable.
"""
import sys
sys.path.insert(0, r"C:/Users/UseR/AppData/Local/Temp/claude/c--Users-UseR-Downloads-A/33c40683-2cc9-4597-b751-2ee8ef01368d/scratchpad/exp")
from common import run, CHANCE

def show(r):
    d = r["deployed"]; p = r["parallel"]
    dm = sum(v["acc"] for v in d.values())/len(d)
    pm = sum(v["acc"] for v in p.values())/len(p)
    L = "LEARNED" if r["final_ce"] < 2.6 else "PLATEAU"
    print(f"{r['name']:<26} ce={r['final_ce']:.3f} {L} par={pm:.3f} dep={dm:.3f} "
          f"[{r['secs']:.0f}s]", flush=True)

print(f"chance={CHANCE:.4f}   shared refs: lam0 par=.840 | lam.03 par=.309 | lam.1 PLATEAU\n",
      flush=True)
LAMS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
# The shared arm is NOT re-run here: with tiny_synth's default sharing="shared",
# an_shared_lam* produces a byte-identical config hash to the lam*_uniform runs
# already on disk (verified f9a06554... for lam=0.03). Re-training them would
# spend ~70 min re-deriving measured cells. They are read from those runs.
for sharing in ("separate",):
    for lam in LAMS:
        over = {"model.sharing": sharing}
        if lam > 0:
            over |= {"loss.consistency": "uniform", "loss.lam": lam}
        show(run(f"an_{sharing}_lam{lam}", **over))
