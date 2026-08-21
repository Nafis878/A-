# Claude Code Build Prompt — Read-Weighted Consistency (RWC)

> Paste everything below the line into Claude Code as a single message.
> Recommended: run it in a fresh repo directory. Expect it to work for several hours.
> After it scaffolds, save the prompt as `SPEC.md` in the repo so later sessions can re-read it.

---

You are building a complete, runnable research codebase for a machine learning paper. Read this entire spec before writing any code. Then build it in the milestone order given, running the stated verification at the end of each milestone before moving on. Do not skip verification steps. Do not build ahead of the current milestone.

## 0. What this project is

I am testing a single scientific claim:

> The uniform L2 consistency loss used to distill recurrent memory in bounded-memory Transformers is misaligned with what the memory is actually **read for** downstream. The misalignment grows with retrieval distance. Weighting the consistency loss by per-dimension **read-influence** fixes it.

This is a rebuttal-style paper against **Maglev** (arXiv:2608.02870), which trains a sliding-window decoder to reproduce memory targets from a full-attention prefiller under the loss `λ‖m_t − m'_t‖₂/√d`. Maglev reports that raising λ helps one variant and hurts another, and explains it only as "capacity." My claim is that it is **allocation**: uniform L2 spends limited memory capacity on dimensions that are never read.

The codebase must produce three results:

- **C1 (diagnostic)** — rank correlation between per-dimension consistency error and per-dimension read-influence **decays as retrieval distance H grows**.
- **C2 (mechanism)** — masking the consistency loss to the top-k% highest read-influence dimensions removes the high-λ pathology; random-k and bottom-k do not.
- **C3 (method)** — RWC's advantage over uniform L2 is near zero at short range and **grows with retrieval distance**.

C3's *slope* is the result, not its mean. A flat improvement would indicate RWC is acting as a generic regulariser and would weaken the paper. Build the evaluation so that slope is directly measurable.

## 1. Hard constraints — read carefully, these are unusual

**All training happens in Google Colab.** This is the dominant design constraint and it drives everything:

- A single GPU, most likely T4 (16GB) or A100 (40GB). Never assume multi-GPU. Never write DDP/FSDP code.
- Sessions disconnect without warning. **Every run must be resumable from a checkpoint with zero manual intervention.** A run that cannot survive a disconnect is a bug.
- Persistent storage is Google Drive, mounted at `/content/drive/MyDrive`. It is slow and has quota limits. Write checkpoints and logs there; keep hot data on local `/content` disk.
- Target **under 2 hours of wall-clock per run** so a disconnect rarely destroys a full run.

**Consequences for the science, which you must respect in the design:**

At the model scales that fit this budget (5–30M parameters), standard downstream benchmarks — PIQA, HellaSwag, ARC, WinoGrande — are at chance and carry no signal. **Do not build an evaluation harness around them.** Maglev observed its λ anomaly in downstream accuracy at 435M; I cannot measure that. The global metric here is **long-range probe accuracy** instead, which is what the mechanism actually predicts and is measurable at small scale.

So the primary experimental domain is **synthetic tasks with known ground-truth read structure**, where I can verify the read-influence estimator against truth rather than trusting it. Natural-language LM training is a secondary confirmation, not the headline.

**Build fresh.** No nanochat dependency, no forked training stack. Use PyTorch + Hugging Face `datasets`/`tokenizers` only.

## 2. Repository layout

Create exactly this. No extra files, no `utils.py` grab-bag.

```
rwc/
  README.md
  SPEC.md                 # this prompt, verbatim
  requirements.txt
  configs/
    tiny_synth.yaml       # synthetic, ~5M params, T4-friendly
    small_lm.yaml         # LM, ~25M non-emb params, T4 ok
    medium_lm.yaml        # LM, ~60M non-emb params, A100 only
  rwc/
    __init__.py
    config.py             # dataclass configs, YAML load, validation
    data/
      synthetic.py        # the three probe tasks + on-the-fly generation
      lm.py               # FineWeb-Edu tokenization -> uint16 shards, loader
    model/
      attention.py        # sliding-window + full causal attention
      blocks.py           # transformer block, RMSNorm, RoPE, SwiGLU
      maglev.py           # prefiller Q, decoder P, recurrent K/V injection
      baselines.py        # plain SWA transformer, full-attention transformer
    influence.py          # read-influence estimators (grad + saliency)
    losses.py             # CE, uniform consistency, RWC, masked variants
    train.py              # single-run training loop, resumable
    evaluate.py           # BPB, probe accuracy vs distance, C1 correlation
    analysis.py           # effective rank, decorrelation curves, figure data
  scripts/
    prepare_lm_data.py    # one-time tokenization to Drive
    run_queue.py          # Colab-resilient experiment orchestrator
    make_figures.py       # matplotlib figures from results/
  notebooks/
    colab_driver.ipynb    # the ONLY thing I open in Colab
  tests/
    test_shapes.py
    test_recurrence_equivalence.py   # the critical correctness test
    test_influence_groundtruth.py    # the critical validity test
    test_resume.py
  results/                # gitignored; JSON per run
```

## 3. The model — implement this precisely

This is where errors will silently produce a wrong paper, so the equations are given explicitly. Follow them exactly and add a comment citing the equation number next to each implementation.

**Two coupled models.** A prefiller `Q` (more context, training only) and a decoder `P` (sliding-window only, deployed).

Training does two sequence-parallel passes (Maglev Eq. 3):

```
m'_{1:T} = Q_φ(x_{1:T})                    # prefiller pass, m'_0 = 0
m_{1:T}  = P_θ(x_{1:T}, m'_{0:T-1})        # decoder pass, consumes SHIFTED targets
```

The shift is essential. At position `t` the decoder consumes `m'_{t-1}`, not `m'_t`. Getting this wrong makes the task trivial and the results meaningless. Assert it in a test.

**Recurrent K/V injection** (Maglev Eq. 6–9). At decoder layer ℓ, with incoming residual `a_t` and local `q_t, k_t, v_t`:

```
k_rec_t = RMSNorm(W_k_rec @ m'_{t-1})              # Eq. 6
v_rec_t =          W_v_rec @ m'_{t-1}
g_loc_t = 2 * sigmoid(G_loc @ a_t)                 # Eq. 7  (factor 2 -> neutral at 0)
g_rec_t = 2 * sigmoid(G_rec @ a_t)
k_bar_t = g_loc_t * k_t + g_rec_t * k_rec_t        # Eq. 8  (elementwise)
v_bar_t = g_loc_t * v_t + g_rec_t * v_rec_t
c_t = Attention(q_t, k_bar[max(1,t-W+1):t], v_bar[max(1,t-W+1):t])   # Eq. 9
```

`W_k_rec` and `W_v_rec` are shared across the sequence but **layer-specific**. Apply RoPE to both local and recurrent keys before mixing, when RoPE is enabled.

The decoder's post-final-norm state **is** `m_t`, and it is also the LM head input:

```
o_t = U @ m_t
p(x_{t+1} | x_<=t) = softmax(15 * tanh(o_t / 15))   # Eq. 4, logit soft-cap
```

**Layer patterns.** Prefiller `Q` uses `SLSL` (alternating sliding/full-causal). Decoder `P` uses `SSSS` (sliding only). Make the pattern a config string so other patterns are one-line changes.

**Parameter sharing.** Support both `shared` (Q and P share transformer blocks, with separate residual-scaling parameters on the decoder path) and `separate` (independent stacks). Both variants are needed — the λ anomaly appears in one and not the other.

**Inference** (Eq. 10). `Q` is discarded; `P` runs recurrently consuming its own `m_{t-1}`. Bounded KV cache of size `W`.

### The single most important test

`tests/test_recurrence_equivalence.py`: feed a model the ground-truth memory trajectory in parallel-training mode, and separately run it in recurrent inference mode fed its own memories, with the memories forced identical. **Outputs must match to within float tolerance.** If parallel and recurrent paths disagree, every downstream number is garbage. Write this test before writing the training loop.

## 4. Read-influence — define it exactly, then prove it works

Read-influence `w_j ∈ R^d` measures how much each dimension of memory `m_j` affects predictions at positions `j+1 … j+H` through the recurrent K/V channel.

**Estimator A — gradient (reference, exact, slow).**

```
w_j[i] = E_batch[ | ∂ L_CE(positions j+1 .. j+H) / ∂ m_j[i] | ]
```

Compute with one truncated backward pass over a window of size `H`. Gradient must flow **only** through the recurrent channel — detach the local path — so this measures memory read specifically, not general position importance.

**Estimator B — saliency (cheap, used in training).**

Gradient×input style, from cached attention statistics:

```
w_j ≈ | m_j * sum_over_layers( W_k_rec^T @ (attn_mass_j * g_rec_j)
                             + W_v_rec^T @ (attn_mass_j * g_rec_j) ) |
```

where `attn_mass_j` is the total attention weight future queries within the window place on cache entry `j`. This is what RWC uses at training time, because it costs a cheap extra reduction rather than a backward pass.

**Both estimators must be normalised per position** (sum to 1 over `d`) and computed under `torch.no_grad()` / `.detach()` when used as loss weights.

### The validity test that gates everything

`tests/test_influence_groundtruth.py`. Construct a synthetic task where the true read structure is **known by construction** — for example, a copy task where only a designated subset of memory dimensions can possibly carry the retrieved token. Train briefly, then require:

- Estimator A recovers the true important dimensions with Spearman ρ > 0.7.
- Estimator B agrees with Estimator A with Spearman ρ > 0.6.

If these fail, RWC is weighting noise and the whole paper collapses. **Do not proceed past this test.** Report the actual correlations, do not just assert pass/fail.

## 5. Losses

```python
# uniform (Maglev, Eq. 5)
L = CE + lam * ||m_t - m_prime_t||_2 / sqrt(d)

# RWC (ours)
L = CE + lam * ||w_t * (m_t - m_prime_t)||_2 / sqrt(d)      # w_t detached, EMA-smoothed

# masked ablations (C2)
#   mask = top-k% / bottom-k% / random-k% of dims by read-influence
L = CE + lam * ||mask_t * (m_t - m_prime_t)||_2 / sqrt(d)
```

`w_t` is refreshed on a slow EMA (default: recompute every 50 steps, EMA decay 0.9) so the per-step cost is negligible. Make refresh interval and decay configurable. Log the mean/std of `w` so I can see if it collapses to uniform or to a spike — both are failure modes worth catching early.

## 6. Data

**Synthetic probes** (`rwc/data/synthetic.py`) — the primary domain. Three tasks, each with an explicit **retrieval distance** parameter that is the x-axis of every headline figure:

1. **Needle-at-depth** — a key–value pair is planted at distance `D` before the query. Must be recalled.
2. **Two-hop retrieval** — `A→B` planted at distance `D₁`, `B→C` at `D₂`; query `A`, answer `C`. Forces composition across two memory reads.
3. **Delayed recall past the window** — the fact sits at distance `D > W`, so it is provably unreachable through local sliding-window attention and **can only** survive via the recurrent memory. This is the cleanest test of the mechanism.

Generate on the fly from a seeded RNG. Sweep `D` across a grid spanning well below `W` to several multiples of `W`. Every probe result must be reported **per distance bucket**, never averaged into a single number — the slope is the finding.

**Language modelling** (`rwc/data/lm.py`) — secondary confirmation. Stream `HuggingFaceFW/fineweb-edu` (`sample-10BT`), tokenize with a 16k-vocab BPE tokenizer trained on a subset (keeps the embedding table small at these model sizes), write `uint16` shards to Drive. Target ~1B training tokens and a held-out validation split. `scripts/prepare_lm_data.py` runs once and is checkpointed so an interrupted tokenization resumes.

Report **bits-per-byte**, not just loss, so numbers are comparable to Maglev's table.

## 7. Colab orchestration — treat disconnection as the normal case

`scripts/run_queue.py` and `notebooks/colab_driver.ipynb` together must give me this behaviour: I open the notebook, run one cell, and it picks up whatever is unfinished and keeps going. Specifically:

- A run manifest (JSON on Drive) listing every experiment with status `pending | running | done | failed`, its config hash, and its checkpoint path.
- On start: mount Drive, detect GPU type, select batch size and gradient accumulation to hit a **fixed global token count per step regardless of GPU**, then claim the next `pending` run.
- A `running` run whose heartbeat is stale (>20 min) is reclaimed as `pending` — this is how disconnects self-heal.
- Checkpoint every `N` steps and on `SIGTERM`. Resume restores model, optimizer, LR schedule, dataloader position, and RNG state. **Resumed runs must be bit-comparable to uninterrupted ones** — `tests/test_resume.py` verifies this by checkpointing mid-run, killing, resuming, and comparing to a clean run.
- Log per-step metrics to a CSV on Drive (not just stdout — stdout dies with the session). Optional W&B behind a flag, off by default.
- Print an ETA and a clear "SAFE TO DISCONNECT" banner after each checkpoint.

Precision: bf16 on A100, fp16 with a grad scaler on T4, chosen automatically. `torch.compile` behind a config flag, **default off** — its compile time is punishing on Colab.

## 8. Experiment grid

Encode this as the default manifest. Sorted by scientific priority — if I run out of time, the top of the list must already be done.

| Priority | Group | Configs | Runs | Serves |
|---|---|---|---|---|
| 1 | Influence validation | ground-truth synthetic | 3 | gates everything |
| 2 | Synthetic λ sweep, uniform | λ ∈ {0.1, 0.3, 1.0} × {shared, separate} | 6 | reproduce anomaly |
| 3 | Synthetic λ sweep, RWC | λ ∈ {0.1, 0.3, 1.0} × {shared, separate} | 6 | C3 headline |
| 4 | Top-k masking | k ∈ {10, 25, 50, 100} at λ=1.0 shared | 4 | C2 mechanism |
| 5 | Mask controls | random-k, bottom-k | 2 | C2 falsification |
| 6 | Baselines | SWA-only, full-attention, Maglev | 3 | reference points |
| 7 | Seeds | 3 seeds × 4 headline configs | 8 | error bars |
| 8 | LM confirmation | small_lm × {Maglev, RWC} | 2 | natural-language check |
| 9 | Scale trend | tiny/small/medium × {Maglev, RWC} | 4 | scaling claim |

Every result JSON records: full config, git commit, GPU type, wall-clock, seed, final metrics, **and probe accuracy broken out per distance bucket**.

## 9. Build order and gates

Build in this order. Run the verification at the end of each milestone and **report the actual numbers to me** before continuing.

- **M1 — Skeleton.** Repo layout, configs, `requirements.txt`, dataclass config with validation. Verify: `pytest tests/test_shapes.py` passes; configs load and reject invalid combinations.
- **M2 — Model.** Attention, blocks, baselines, Maglev prefiller/decoder with recurrent K/V injection. Verify: **`test_recurrence_equivalence.py` passes.** Report the max absolute deviation between parallel and recurrent paths.
- **M3 — Data + training loop.** Synthetic tasks, resumable trainer, run queue. Verify: `test_resume.py` passes; a 200-step tiny run completes, is killed, resumes, and matches an uninterrupted run.
- **M4 — Influence.** Both estimators. Verify: **`test_influence_groundtruth.py` passes.** Report the actual Spearman correlations. If below threshold, stop and tell me — do not tune the test to pass.
- **M5 — Losses + C1.** RWC, masked variants. Produce the C1 decorrelation curve. Report whether correlation actually decays with H — **if it does not, say so plainly.** A null result here is information, not a bug to hide.
- **M6 — Full grid + figures.** Run the manifest by priority; `make_figures.py` produces the C1, C2 and C3 plots.

## 10. Non-goals — do not do these

- No multi-GPU, no DDP, no FSDP, no DeepSpeed.
- No PIQA/HellaSwag/ARC/WinoGrande harness. At these scales they are noise.
- No Hydra, no PyTorch Lightning, no experiment-tracking framework beyond optional W&B. Plain PyTorch and dataclasses.
- No web UI, no dashboard, no Gradio.
- No premature optimization: correctness and resumability first, speed only where profiling shows a real bottleneck.
- No inventing hyperparameters that contradict this spec. If something here is genuinely underspecified, pick a sensible default, **log it as an explicit assumption in `README.md`**, and tell me.

## 11. How to work

- Write the tests in each milestone **before or alongside** the implementation, not after.
- Prefer clear code over clever code — this becomes a public artifact attached to a paper.
- Type-hint public functions. Docstring anything implementing a numbered equation with its equation number.
- Where you implement something that could plausibly be done another way, add a one-line comment saying why you chose this way.
- When you finish a milestone, stop, report the verification numbers, and wait for my go-ahead. Do not chain milestones silently.

Start with M1. Before you write code, restate in your own words: the shift relationship between `m'` and `m`, and why `test_recurrence_equivalence` is the test that matters most. If either is unclear to you, ask me before proceeding.

Use this repo of mine to store everything: https://github.com/Nafis878/A-.git
