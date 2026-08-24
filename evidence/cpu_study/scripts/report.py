"""Read the grid off disk and run the C3 significance test."""
import sys, json
sys.path.insert(0, r"C:/Users/UseR/AppData/Local/Temp/claude/c--Users-UseR-Downloads-A/33c40683-2cc9-4597-b751-2ee8ef01368d/scratchpad/exp")
sys.path.insert(0, r"c:/Users/UseR/Downloads/A")
from pathlib import Path
from common import ROOT, DISTANCES, CHANCE
from rwc.analysis import paired_slope_test, wilson_interval

def load(arm, seeds):
    out = {}
    for s in seeds:
        f = ROOT / "runs" / f"g_{arm}_s{s}" / "result.json"
        if f.exists():
            r = json.loads(f.read_text())
            out[s] = ([r["deployed"][str(d)]["acc"] for d in DISTANCES], r)
    return out

def show(arm, data):
    if not data: return
    n = len(data)
    print(f"\n--- {arm}  ({n} seeds) ---")
    print(f"{'D':>6} " + "".join(f"{('s'+str(s)):>7}" for s in sorted(data)) + f"{'mean':>8}{'95% CI':>18}")
    accs = {s: v[0] for s, v in data.items()}
    for i, d in enumerate(DISTANCES):
        col = [accs[s][i] for s in sorted(accs)]
        mean = sum(col)/len(col)
        n_eval = data[sorted(data)[0]][1]["deployed"][str(d)]["n"]
        lo, hi = wilson_interval(round(mean*n_eval*n), n_eval*n)
        print(f"{d:>6} " + "".join(f"{c:>7.3f}" for c in col) +
              f"{mean:>8.3f}   [{lo:.3f}, {hi:.3f}]")

def compare(name, ctl_arm, trt_arm, seeds):
    c, t = load(ctl_arm, seeds), load(trt_arm, seeds)
    common = sorted(set(c) & set(t))
    if len(common) < 3:
        print(f"\n[{name}] only {len(common)} paired seeds -- skipping"); return
    cc = {s: c[s][0] for s in common}; tt = {s: t[s][0] for s in common}
    r = paired_slope_test(DISTANCES, cc, tt, n_permutations=20000)
    print(f"\n### C3 TEST: {name}   ({len(common)} paired seeds)")
    print(f"{'D':>6} {'control':>9} {'treat':>9} {'delta':>9}")
    for i, d in enumerate(DISTANCES):
        cm = sum(cc[s][i] for s in common)/len(common)
        tm = sum(tt[s][i] for s in common)/len(common)
        print(f"{d:>6} {cm:>9.3f} {tm:>9.3f} {tm-cm:>+9.3f}")
    print(f"  mean advantage        {r.mean_delta:+.4f}")
    print(f"  slope vs log2(D)      {r.slope_mean:+.4f}   95% CI [{r.ci_low:+.4f}, {r.ci_high:+.4f}]")
    print(f"  permutation p         {r.p_value:.4f}   (floor {r.min_achievable_p:.4f})")
    print(f"  VERDICT: {r.verdict()}")

seeds = list(range(8))
for arm in ["uniform", "rwc", "uniform_dt", "rwc_dt"]:
    show(arm, load(arm, seeds))
compare("RWC vs uniform (symmetric pull, Maglev as written)", "uniform", "rwc", seeds)
compare("RWC vs uniform (stop-grad target)", "uniform_dt", "rwc_dt", seeds)
compare("stop-grad vs symmetric, uniform arm", "uniform", "uniform_dt", seeds)
