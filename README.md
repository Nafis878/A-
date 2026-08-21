# Read-Weighted Consistency (RWC)

Research code for a single claim:

> The uniform L2 consistency loss used to distill recurrent memory in
> bounded-memory Transformers is misaligned with what the memory is actually
> **read for** downstream. The misalignment grows with retrieval distance.
> Weighting the consistency loss by per-dimension **read-influence** fixes it.

This is a rebuttal-style study against Maglev (arXiv:2608.02870), which trains a
sliding-window decoder to reproduce memory targets from a full-attention
prefiller under `λ‖m_t − m'_t‖₂/√d` and explains its λ anomaly as *capacity*.
The claim here is that it is *allocation*: uniform L2 spends limited memory
capacity on dimensions that are never read.

Full specification: [SPEC.md](SPEC.md).

## The three results

| | Claim | Measured by |
|---|---|---|
| **C1** | Rank correlation between per-dimension consistency error and per-dimension read-influence decays as retrieval distance `H` grows | `rwc/analysis.py` |
| **C2** | Masking consistency to the top-k% highest read-influence dims removes the high-λ pathology; random-k and bottom-k do not | `rwc/losses.py` ablations |
| **C3** | RWC's advantage over uniform L2 is near zero at short range and **grows** with retrieval distance | probe accuracy per distance bucket |

C3's **slope** is the finding, not its mean. Probe results are always reported
per distance bucket and never averaged into one number.

## Status

| Milestone | State |
|---|---|
| M1 — skeleton, configs, validated dataclass config | **done** |
| M2 — attention, blocks, Maglev prefiller/decoder, baselines | **done** |
| M3 — synthetic data, resumable trainer, run queue | **done** |
| M4 — read-influence estimators + ground-truth validity gate | **done** |
| M5 — losses and the C1 decorrelation curve | pending |
| M6 — full grid and figures | pending |

`rwc/losses.py::consistency_loss`, `rwc/evaluate.py::bits_per_byte`,
`rwc/analysis.py`, `scripts/make_figures.py` and the FineWeb-Edu pipeline in
`rwc/data/lm.py` raise `NotImplementedError` naming their milestone. Everything
else is built and tested: `pytest tests/ -q` is 133 passed, 0 skipped.

### Verification numbers as built

| Gate | Result |
|---|---|
| Parallel vs recurrent decoder, fp64 | max abs deviation **1.1e-16 … 3.3e-16**; `m` identical to 0.0 |
| Parallel vs recurrent decoder, fp32 | max abs deviation **1.5e-07 … 2.4e-07** |
| SDPA vs math attention backends | **0.0** exactly |
| Resume vs uninterrupted, 20 steps | max abs Δloss **0.000e+00** (in-process and across a fresh process) |
| Influence A vs known read structure | Spearman ρ **+0.9987** (threshold 0.7); live carriers only +0.9529; precision@k **1.000** |
| Influence B vs A | Spearman ρ **+0.9926** (threshold 0.6); live carriers only +0.7235 |
| Parameter estimate vs built model | exact for all three configs and both sharing modes |

## Layout

```
configs/          tiny_synth, small_lm, medium_lm
rwc/config.py     dataclass configs, YAML load, validation
rwc/data/         synthetic probes; FineWeb-Edu shards
rwc/model/        attention, blocks, Maglev Q/P, baselines
rwc/influence.py  read-influence estimators (gradient + saliency)
rwc/losses.py     CE, uniform consistency, RWC, masked variants
rwc/train.py      resumable single-run training loop
scripts/          data prep, Colab run queue, figures
notebooks/        colab_driver.ipynb — the only thing opened in Colab
tests/
```

## Configs as shipped

| Config | Geometry | Non-emb params | Window | Seq len | Target |
|---|---|---|---|---|---|
| `tiny_synth` | d=256, L=4, 4 heads | 4.26M (4.28M total) | 64 | 512 | T4, synthetic, primary domain |
| `small_lm` | d=512, L=8, 8 heads / 4 kv | 27.8M | 256 | 1024 | T4, LM confirmation |
| `medium_lm` | d=768, L=8, 12 heads / 6 kv | 62.5M | 512 | 1024 | **A100 only** |

## Running

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

Training runs in Google Colab via `notebooks/colab_driver.ipynb` (M3). Nothing
here assumes multi-GPU; there is no DDP/FSDP code and there will not be.

---

## Assumptions log

SPEC.md section 10 requires every genuinely underspecified choice to be recorded
here. Each entry states what was chosen and why.

### M1

1. **Repo root.** The SPEC's outer `rwc/` is the repository directory itself, so
   files live at `./README.md`, `./configs/`, `./rwc/config.py`. The inner `rwc/`
   is the Python package.

2. **`tests/conftest.py` exists** although it is not in the SPEC's file tree. It
   does one thing: put the repo root on `sys.path` so `import rwc` works without
   an install step. The alternative — a `pyproject.toml` and `pip install -e .` —
   is more machinery and one more file, and adds an install step to every Colab
   session.

3. **Layer patterns tile.** `prefiller_pattern: SLSL` with `n_layers: 8` expands
   to `SLSLSLSL`. The pattern length must divide `n_layers` or loading fails,
   rather than silently truncating or padding.

4. **A full-causal layer in `decoder_pattern` is a hard error.** The decoder is
   the deployed, bounded-memory model; an `L` layer there would leak unbounded
   context and invalidate every result, so it is rejected at load time rather
   than left as a footgun.

5. **GQA in the LM configs** (`n_kv_heads = n_heads / 2`). The recurrent
   injection adds four `d × d_kv` matrices per layer, which at full MHA pushes
   `small_lm` to ~34M non-embedding, well over the SPEC's ~25M. Halving the KV
   heads brings it to 27.8M. `tiny_synth` stays full MHA — it is small enough
   not to need the saving, and the synthetic domain is the headline.

6. **Parameter budgets are approximate.** SPEC targets are "~5M / ~25M / ~60M";
   as shipped the configs are 4.28M total / 27.8M non-emb / 62.5M non-emb. Tests
   assert generous bands, whose purpose is to catch a geometry change that blows
   the Colab budget, not to pin an exact count.

7. **SwiGLU hidden width is `ceil(8/3 · d / 64) · 64`.** The 8/3 multiplier makes
   the FFN parameter-matched to a conventional 4× GELU FFN; the rounding rule
   lives in `config.py` so every config gets the same one.

8. **`config_hash` includes `run.seed`** but excludes `run.out_dir`,
   `run.drive_dir`, `run.name` and `run.wandb`. Different seeds are genuinely
   different runs in the manifest (SPEC section 8, priority 7); different output
   paths are not.

9. **`optim.micro_batch: null` means the trainer resolves it** from available
   VRAM at startup, then derives gradient accumulation to hit
   `global_tokens_per_step` exactly. If `micro_batch` *is* pinned, a
   `global_tokens_per_step` that is not an exact multiple of
   `micro_batch × seq_len` is rejected — silently missing the token budget would
   make runs on different GPUs incomparable.

10. **`compile: false` in every shipped config**, per SPEC section 7: compile
    time is punishing on Colab.

### M2–M4

11. **Estimator A runs as a truncated recurrent rollout, not in parallel mode.**
    In parallel teacher-forced mode the memory chain is broken (memory input is
    the prefiller's output), so `∂L/∂m_j` would capture only within-window and
    cross-layer propagation and would understate influence for `H > W`.

12. **"Detach the local path" is implemented as detaching the residual `a_t`
    that feeds the gates `G_loc`/`G_rec`, during the influence pass only.**
    Gradient w.r.t. `m_j` is already confined to the recurrent channel by
    construction; the gates are the one place a local-path dependence leaks in.

13. **Gradient flows from the decoder into the prefiller through `mem_in`** (no
    detach). Detaching would leave `Q` trained only by the consistency term,
    which pulls it toward `P` and is degenerate. Configurable, default off.

14. **Estimator B needs materialised attention probabilities**, so attention has
    two backends selected by `model.attn_impl`: `sdpa` for training, `math` for
    saliency collection.

15. **`sharing: shared`** means `Q` and `P` share transformer block weights, run
    under different attention masks (`SLSL` vs `SSSS`), with the recurrent
    injection modules and per-layer residual-scaling parameters exclusive to the
    decoder path.

16. **Delayed-recall's distance bound is `D > L·(W-1)`, not `D > W`.** The SPEC
    describes the fact as "provably unreachable through local sliding-window
    attention" at `D > W`, which is the single-layer statement. With `L` stacked
    sliding layers information propagates `L·(W-1)` positions, and a test in
    `tests/test_recurrence_equivalence.py` pins that bound empirically. The
    stricter condition is enforced at config load (`ModelConfig.local_reach`);
    using the looser one would have let local attention solve part of the task.
    For `tiny_synth` (L=4, W=64) the bound is 252, not 64, so the delayed-recall
    runs shrink the window to 16 to keep a usable distance axis.

17. **The gates read the *normalised* sublayer input as `a_t`,** not the raw
    residual stream. Eq. 7 says "incoming residual"; with a pre-norm block the
    raw residual grows with depth and `2·sigmoid(G a_t)` saturates to 0 or 2,
    killing the gate's gradient. The normalised input is what every other
    projection in the sublayer consumes.

18. **Eq. 6's RMSNorm on `k_rec` is computed per head over `d_head`,** with a
    learnable gain of size `d_kv` (one per key dimension). The equation does not
    name an axis; per-head is standard QK-norm practice and keeps one head's
    scale from being set by another's.

19. **The prefiller `Q` gets no auxiliary CE of its own.** It is trained purely
    by gradient flowing back through `mem_in` from the decoder's CE, plus the
    consistency term. Adding an auxiliary head would be a hyperparameter the
    SPEC does not mention.

20. **The KV cache appends out-of-place (`torch.cat`).** In-place writes into a
    preallocated buffer would be faster, but Estimator A rolls the decoder
    forward with the autograd graph retained and an in-place write into a tensor
    the graph still needs raises at backward.

21. **`normalise` clamps the denominator rather than adding epsilon.** Gradient
    magnitudes in Estimator A are often ~1e-7, comparable to eps itself, and
    `w / (sum + eps)` then shrinks every weight by several percent in a way that
    depends on gradient scale — an error that rides along in every RWC weight
    without ever failing a test.

22. **Influence anchors are chosen from scoring positions, not by position.**
    Under an answer-only loss almost every horizon window is entirely ignored,
    `L_CE` over it is undefined, and the estimator would return zeros — which
    reads exactly like "this memory is never used". Anchors with nothing to
    score are dropped, and if none remain the call raises rather than returning
    an all-zero vector.

23. **`data.loss_on_answer_only` (default `False`).** Full-sequence CE matches
    "CE" in SPEC section 5 and is what the grid runs use. Filler positions are
    unpredictable noise, so answer-only converges far faster and is what the
    short CPU gate tests use.

24. **Priority 9's scale trend uses tiny/medium LM widths.** The SPEC's table
    lists `tiny/small/medium × {Maglev, RWC}` as 4 runs, but `small` is already
    covered by priority 8 — and a test confirmed the cells collided by config
    hash exactly. Priority 9 now supplies the other two widths, so the trend has
    three real points and no manifest cell is wasted.
