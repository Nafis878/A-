# Evidence archive

Every number in the write-up, recomputable without a GPU, a checkpoint, or any
scratch directory:

```
python evidence/verify.py     # recomputes the headline and lists every cell
pytest tests/test_evidence.py # asserts it has not moved
```

## Headline

Delayed recall, `d=128, L=2, W=4` → stacked local reach `L·(W−1) = 6`, with
probe distances `[8, 12, 16, 24]` = 1.3×–4× the reach, so no distance is
solvable without memory. Trained through the recurrence (BPTT), no consistency
term, 1500 steps, 10 seeds per arm.

| arm | learns the task | 95% CI |
|---|---|---|
| Maglev write rule | **3/10** | [0.11, 0.60] |
| + identity path | **10/10** | [0.72, 1.00] |

One-sided Fisher exact, **p = 1.55e-03**.

"Learns" means deployed accuracy above the Wilson upper bound for chance, not
merely above chance. The outcome is bimodal: every successful cell sits at
exactly 1.000 at all four distances, every failure within noise of 0.5.

## Layout

| path | contents |
|---|---|
| `cpu_study/runs/` | 69 cells at `d=128, L=2` — the headline and all controls |
| `cpu_study/scripts/` | the experiment definitions that produced them |
| `cpu_study/logs/` | their console output |
| `rwc_results/` | early A100 runs — uniform vs read-weighted consistency (C1/C3) |
| `headline_rwc_results/` | the λ × sharing grid |
| `calibration_results/` | task-difficulty calibration |
| `../ladder_runs/` | the width ladder, 0.5M → 27.4M params (in progress) |

Checkpoints and loss traces are **not** archived: ~51 MB per cell against ~2.5 KB
of `result.json`, and nothing here needs them. Each `result.json` embeds its full
config, so a cell is self-describing.

## Two things worth knowing before reading the code

**Arms are selected by config, never by run name.** The ten seeds per arm were
not produced by one script — `reliability.py` starts at seed 3 for the write-rule
arm and seed 2 for the identity-path arm, because earlier scripts had already
covered the lower seeds under different names (`r_bptt_*`, `f_*`). Name matching
silently drops seeds 0–2, and did: the first version of `verify.py` reported
2/7 and 8/8.

**Two cells were run twice** under different names (identity path at seeds 0 and
2). They are deduplicated by seed, and `verify.py` reports whether their verdicts
agree rather than quietly keeping one. They agree, at 1.000 in both cases.
