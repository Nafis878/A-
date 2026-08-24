"""The C3 experiment: 2x2 factorial {uniform, rwc} x {detach_target} at lam=1.0.

Eight seeds, because the sign-flip permutation test has a p floor of
2/2**n_seeds and five seeds cannot reach 0.05 at any effect size.
"""
import sys, json
sys.path.insert(0, r"C:/Users/UseR/AppData/Local/Temp/claude/c--Users-UseR-Downloads-A/33c40683-2cc9-4597-b751-2ee8ef01368d/scratchpad/exp")
from common import run, line, ROOT, DISTANCES, CHANCE

SEEDS = list(range(8))
ARMS = {
    "uniform":    {"loss.consistency": "uniform", "loss.lam": 1.0},
    "rwc":        {"loss.consistency": "rwc", "loss.lam": 1.0,
                   "loss.influence.estimator": "saliency",
                   "loss.influence.refresh_every": 50},
    "uniform_dt": {"loss.consistency": "uniform", "loss.lam": 1.0,
                   "loss.detach_target": True},
    "rwc_dt":     {"loss.consistency": "rwc", "loss.lam": 1.0,
                   "loss.influence.estimator": "saliency",
                   "loss.influence.refresh_every": 50,
                   "loss.detach_target": True},
}
only = sys.argv[1:] or list(ARMS)
for arm in only:
    for seed in SEEDS:
        print(line(run(f"g_{arm}_s{seed}", seed=seed, **ARMS[arm])), flush=True)
