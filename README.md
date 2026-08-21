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
| M2 — attention, blocks, Maglev prefiller/decoder, baselines | pending |
| M3 — synthetic data, resumable trainer, run queue | pending |
| M4 — read-influence estimators + ground-truth validity gate | pending |
| M5 — losses and the C1 decorrelation curve | pending |
| M6 — full grid and figures | pending |

Test files for unbuilt milestones exist and `pytest.skip` with the milestone
named, so the suite is green *and* honest about what is not yet covered.

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

### M2–M6 (decided, not yet implemented)

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
