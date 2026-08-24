"""Recompute every headline number from the archived result.json files.

No claim in the write-up should depend on a machine, a checkpoint or a scratch
directory that no longer exists. Each `result.json` embeds its own full config,
so everything below is recomputed from the archive alone:

    python evidence/verify.py

Arms are selected by CONFIG, never by run name. The ten seeds per arm were not
produced by one script -- `reliability.py` starts at seed 3 for the write-rule
arm and seed 2 for the identity-path arm, because earlier scripts had already
covered the lower seeds under different names (`r_bptt_*`, `f_*`). Matching on
names would silently have dropped seeds 0-2, and did: the first version of this
file reported 2/7 and 8/8. Matching on the config makes arm membership a
property of what was actually run.

Two cells were run twice under different names (seed 2 of the identity path, and
seed 0 at gate bias 2.0). They are deduplicated by seed, and their agreement is
reported as an internal consistency check rather than being quietly discarded.

Directories:
    cpu_study/runs/     69 cells at d=128, L=2 -- the headline and its controls
    cpu_study/scripts/  the experiment definitions that produced them
    cpu_study/logs/     their console output
    ../ladder_runs/     the width ladder, 0.5M -> 27.4M (in progress)
    rwc_results/, headline_rwc_results/, calibration_results/   earlier A100 runs

Checkpoints and loss traces are deliberately not archived: ~51 MB per cell
against ~2.5 KB of result.json, and nothing here needs them.
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rwc.analysis import wilson_interval  # noqa: E402

HERE = Path(__file__).resolve().parent

# The headline task. A cell belongs to an arm iff its config matches this
# exactly; `memory_residual` then selects which arm.
HEADLINE_TASK = {
    "d_model": 128, "n_layers": 2, "seq_len": 48,
    "distances": [8, 12, 16, 24], "n_values": 2, "n_distractors": 0,
    "train_mode": "bptt", "lam": 0.0, "gate_bias": 2.0,
}


def load(pattern: str, root: Optional[Path] = None) -> Dict[str, dict]:
    out = {}
    for f in sorted((root or HERE).glob(pattern)):
        p = json.loads(f.read_text(encoding="utf-8"))
        out[p.get("name", f.parent.name)] = p
    return out


def deployed_mean(p: dict) -> float:
    b = p["deployed"]
    return sum(v["acc"] for v in b.values()) / len(b)


def learned(p: dict) -> bool:
    """Above the Wilson upper bound for chance -- not merely above chance."""
    b = p["deployed"]
    n = sum(v["n"] for v in b.values())
    chance = 1.0 / p["config"]["data"]["n_values"]
    return deployed_mean(p) > wilson_interval(round(chance * n), n)[1]


def matches_headline(p: dict) -> bool:
    c = p["config"]
    m, d, o, l = c["model"], c["data"], c["optim"], c["loss"]
    h = HEADLINE_TASK
    return (m["d_model"] == h["d_model"] and m["n_layers"] == h["n_layers"]
            and d["seq_len"] == h["seq_len"] and d["distances"] == h["distances"]
            and d["n_values"] == h["n_values"]
            and d.get("n_distractors", 0) == h["n_distractors"]
            and o.get("train_mode") == h["train_mode"]
            and l.get("lam", 0.0) == h["lam"]
            and m.get("memory_gate_bias") == h["gate_bias"])


def fisher_greater(k1: int, n1: int, k2: int, n2: int) -> float:
    """One-sided Fisher exact: is arm 1's success rate higher than arm 2's?

    Sums the hypergeometric tail over its true support,
    ``max(0, col1-(total-row1)) .. min(row1, col1)``. Using ``col1 - (n1-k1)``
    as the lower bound gives an EMPTY range whenever arm 1 is perfect, and so
    reports p = 0 for the very case the test exists to evaluate.
    """
    a, b, c, d = k1, n1 - k1, k2, n2 - k2
    total, row1, col1 = a + b + c + d, a + b, a + c
    lo = max(0, col1 - (total - row1))
    hi = min(row1, col1)
    return sum(
        comb(row1, x) * comb(total - row1, col1 - x) / comb(total, col1)
        for x in range(lo, hi + 1) if x >= a
    )


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def collect_arms(cpu: Dict[str, dict]) -> Tuple[Dict, Dict, List[str]]:
    """Group headline-matching cells into arms keyed by seed; note duplicates."""
    arms: Dict[bool, Dict[int, List[Tuple[str, dict]]]] = {False: {}, True: {}}
    for name, p in cpu.items():
        if not matches_headline(p):
            continue
        arm = bool(p["config"]["model"].get("memory_residual"))
        arms[arm].setdefault(p["config"]["run"]["seed"], []).append((name, p))

    notes = []
    for arm, by_seed in arms.items():
        label = "identity path" if arm else "Maglev write rule"
        for seed, cells in sorted(by_seed.items()):
            if len(cells) > 1:
                verdicts = {n: learned(q) for n, q in cells}
                agree = len(set(verdicts.values())) == 1
                notes.append(
                    f"  {label}, seed {seed}: run twice as "
                    f"{', '.join(verdicts)} -- verdicts "
                    f"{'AGREE' if agree else 'DISAGREE'} "
                    f"({', '.join(f'{n}={deployed_mean(q):.3f}' for n, q in cells)})"
                )
    return arms[False], arms[True], notes


def main() -> int:
    cpu = load("cpu_study/runs/*/result.json")
    print(f"loaded {len(cpu)} cells from cpu_study/runs")

    plain, res, notes = collect_arms(cpu)

    section("HEADLINE  delayed recall, every distance beyond stacked reach 6")
    rates = {}
    for label, by_seed in (("Maglev write rule", plain), ("+ identity path", res)):
        k = sum(any(learned(q) for _, q in cells) for cells in by_seed.values())
        n = len(by_seed)
        rates[label] = (k, n)
        lo, hi = wilson_interval(k, n)
        print(f"  {label:<20} {k}/{n}   95% CI [{lo:.2f}, {hi:.2f}]")
    (kp, np_), (kr, nr) = rates["Maglev write rule"], rates["+ identity path"]
    if np_ and nr:
        print(f"  Fisher exact, one-sided:  p = {fisher_greater(kr, nr, kp, np_):.2e}")

    if notes:
        section("duplicate cells (deduplicated by seed)")
        print("\n".join(notes))

    section("per-seed deployed accuracy")
    for label, by_seed in (("plain", plain), ("res  ", res)):
        for seed, cells in sorted(by_seed.items()):
            name, p = cells[0]
            flag = "LEARNED " if learned(p) else "at chance"
            per = " ".join(f"D{k}:{v['acc']:.3f}" for k, v in p["deployed"].items())
            print(f"  {label} s{seed:<2} {deployed_mean(p):.3f}  {flag}  "
                  f"{per}   [{name}]")

    section("ALL OTHER CELLS  (nothing dropped)")
    named = {n for by in (plain, res) for cs in by.values() for n, _ in cs}
    for name in sorted(set(cpu) - named):
        p = cpu[name]
        chance = 1.0 / p["config"]["data"]["n_values"]
        flag = "LEARNED " if learned(p) else "at chance"
        print(f"  {name:<26} deployed={deployed_mean(p):.3f} "
              f"(chance {chance:.3f}) ce={p['final_ce']:.4f}  {flag}")

    ladder = load("*/result.json", HERE.parent / "ladder_runs")
    if ladder:
        section(f"WIDTH LADDER  ({len(ladder)}/50 cells complete)")
        print(f"  {'rung':>6} {'non-emb':>9} {'Maglev':>9} {'identity':>10} "
              f"{'Fisher p':>10}")
        for d in sorted({p["d_model"] for p in ladder.values()}):
            at = [p for p in ladder.values() if p["d_model"] == d]
            kp2 = sum(learned(p) for p in at if p["arm"] == "plain")
            np2 = sum(1 for p in at if p["arm"] == "plain")
            kr2 = sum(learned(p) for p in at if p["arm"] == "res")
            nr2 = sum(1 for p in at if p["arm"] == "res")
            pv = f"{fisher_greater(kr2, nr2, kp2, np2):.2e}" if np2 and nr2 else "  --"
            print(f"  {d:>6} {at[0]['n_params'] / 1e6:>8.1f}M {kp2:>5}/{np2:<3} "
                  f"{kr2:>6}/{nr2:<3} {pv:>10}")

    for name in ("rwc_results", "headline_rwc_results", "calibration_results"):
        runs = load(f"{name}/*/result.json")
        if not runs:
            continue
        section(f"{name}  ({len(runs)} cells)")
        for cell in sorted(runs):
            p = runs[cell]
            if p.get("deployed"):
                print(f"  {cell:<30} deployed={deployed_mean(p):.3f} "
                      f"ce={p.get('final_ce', float('nan')):.4f}")
            else:
                print(f"  {cell:<30} (no deployed eval)")

    print("\nEvery number above is recomputed from archived JSON. No checkpoint,")
    print("scratch directory or GPU is required to reproduce it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
