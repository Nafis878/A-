# Reviewer-facing write-up

`findings.html` is the source of the published results page:
<https://claude.ai/code/artifact/422d67c9-7976-40d2-9a0e-8f6779e09d02>

Every number in it comes from `python evidence/verify.py` and the per-study
`--summary` commands, not from notes. To regenerate them before editing:

```
python evidence/verify.py
python scripts/cpu_ladder.py    --out-dir ladder_runs  --summary
python scripts/variant_study.py --study breadth --out-dir breadth_runs --summary
python scripts/lm_study.py      --out-dir lm_runs      --summary
python scripts/lm_induction.py  --out-dir lm_ind_runs  --summary
```

The page reports four findings with an explicit status each — three that hold
(scale, task breadth, mechanism) and one null (language modelling) — plus the
errors caught during validity checking and the limitations. The null is in the
opening paragraph and the top-line metrics by design: it is the result a
reviewer most needs to see, and burying it would be the kind of omission the
rest of this repository exists to prevent.

To republish after editing, pass the URL above as `url` so the same link updates
rather than creating a second artifact.
