"""Why does a model with one-step gain 0.09 solve a 24-step retrieval?

The ladder produced a contradiction. Cell `d128_plain_s2` reaches deployed
accuracy 1.000 at distance 24 while recording rho = 0.093. Compounded over the
24 steps between needle and query that is 1.7e-20, and sigma_max does not rescue
it (3.5e-17). Yet the answer arrives.

Three hypotheses were measured. The first two are WRONG and are kept because the
third is only credible against them.

1. A GATED WRITER holding gain ~1 only where it has something to carry, so that
   three arbitrary sample positions miss it. Refuted: the position-resolved gain
   is 0.112 along the carry path against 0.150 at the sampled positions -- if
   anything lower, and flat.

2. A HIGHER-ORDER recurrence. Real, but insufficient. The deployed loop keeps a
   sliding KV cache: the injection at step s writes K/V derived from ``m_{s-1}``
   and step t attends over the last W entries, so ``m_t`` depends on
   ``m_{t-1} .. m_{t-W}`` and the order is ``L*(W-1)``, not 1. Measured: lag 6 is
   1.4e9x larger than (lag 1)^6 for plain_s2, so compounding rho is the wrong
   sum. But the closed-loop sensitivity is still only 1.99e-07 -- far too small
   to drive a confident readout.

3. A LATCH. Every measure above is a linearisation, and a linearisation is blind
   to a saturated channel: if the value is stored as WHICH BASIN the memory
   settles into, perturbations decay back while the discrete state copies forward
   intact. Confirmed by dropping infinitesimals and using the task's own signal
   -- two sequences identical but for the needle value:

       cell       before needle   at needle   at query   margin   correct
       plain_s2        0.00e+00      1.2957     1.4338    13.71   yes
       plain_s0        0.00e+00      0.0000     0.0000    -0.01   NO
       res_s0          0.00e+00      0.9498     0.7706    17.99   yes

   plain_s2 separates the two memory trajectories at the needle and the
   separation GROWS to the query, with a 13.7 logit margin -- while its Jacobian
   says 2e-7. The channel is saturated, not absent.

What this changes. The arms differ in KIND, not degree. Maglev's write rule can
only succeed by discovering a latch -- a discrete attractor, a brittle thing for
gradient descent to find -- hence 3/10 seeds. The identity path supplies a LINEAR
maintenance channel by construction: rho ~ 0.99, a lag profile flat at ~0.6, and
rho^D predicting the closed-loop sensitivity within 27% -- hence 10/10. That is a
better account of the headline than "memory is not carried", and it is the one
the measurements support.

It does not weaken the failed cells' reading: plain_s0 shows exactly zero
separation and answers wrongly. A dead channel reads dead by every measure.

Runs at 3 threads so the main ladder keeps the machine.

Usage:
    python scripts/carry_profile.py --out-dir carry_runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from rwc.analysis import recurrence_gain  # noqa: E402
from rwc.config import load_config  # noqa: E402
from rwc.data.synthetic import SyntheticGenerator, SyntheticVocab  # noqa: E402
from rwc.model.maglev import MaglevModel  # noqa: E402
from rwc.train import Trainer  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DISTANCE = 24  # the longest probe; the widest carry window

# d=128 cells from the ladder, chosen to span the three observed behaviours.
BASE: Dict[str, Any] = {
    "model.d_model": 128, "model.n_layers": 2, "model.window": 4,
    "model.n_heads": 4, "model.n_kv_heads": 2,
    "model.prefiller_pattern": "SL", "model.decoder_pattern": "SS",
    "data.task": "delayed_recall",
    "data.seq_len": 48, "model.max_seq_len": 48,
    "data.distances": [8, 12, 16, 24],
    "data.loss_on_answer_only": True,
    "data.n_values": 2, "data.n_keys": 4, "data.n_filler": 8,
    "data.n_distractors": 0,
    "optim.total_steps": 1500, "optim.warmup": 100, "optim.lr": 3e-3,
    "optim.global_tokens_per_step": 32 * 48, "optim.micro_batch": 32,
    "optim.train_mode": "bptt",
    "run.ckpt_every": 500, "run.log_every": 1500,
}
CELLS = [
    ("plain_s2", {"run.seed": 2}),                                    # learns, rho 0.093
    ("plain_s0", {"run.seed": 0}),                                    # fails,  rho 0.005
    ("res_s0", {"run.seed": 0, "model.memory_residual": True}),       # learns, rho 0.991
]


def train_or_load(name: str, over: Dict[str, Any], out_dir: Path) -> tuple:
    cfg_over = dict(BASE)
    cfg_over.update(over)
    cfg_over["run.name"] = name
    cfg_over["run.out_dir"] = str(out_dir)
    cfg = load_config(CONFIGS / "tiny_synth.yaml", overrides=cfg_over,
                      cuda_available=False)
    trainer = Trainer(cfg, device=torch.device("cpu"))
    trainer.maybe_resume()
    if trainer.step < cfg.optim.total_steps:
        print(f"  training {name} from step {trainer.step}...", flush=True)
        trainer.run(quiet=True)
    return trainer, cfg


def fixed_distance_batch(cfg, n: int = 2) -> tuple:
    """A batch whose needle sits at a known, identical position in every row."""
    data = cfg.data.__class__(**{**cfg.data.__dict__, "distances": [DISTANCE]})
    gen = SyntheticGenerator(data, cfg.model, seed=7)
    batch = gen.batch(0, n)
    return batch.tokens, gen.query_pos - DISTANCE, gen.query_pos


def profile(model: MaglevModel, tokens: torch.Tensor) -> List[float]:
    """Exact one-step spectral radius at every position."""
    with torch.no_grad():
        mem_in = MaglevModel.shift_memory(model.forward_prefiller(tokens))
    gains = []
    for t in range(tokens.shape[1]):
        rho, _ = recurrence_gain(model, tokens, mem_in, positions=(t,))
        gains.append(rho)
    return gains


def rollout_gain(model: MaglevModel, tokens: torch.Tensor, p: int, q: int,
                 n_probe: int = 12) -> float:
    """End-to-end sensitivity ||d m_q / d m_p|| through the ACTUAL deployed loop.

    This is the quantity the puzzle needs, and it is not what `recurrence_gain`
    reports. In `forward_recurrent` the injection at step s writes K/V derived
    from ``m_{s-1}`` into a sliding cache, and step t attends over the last W
    cache entries. So ``m_t`` depends on ``m_{t-1} ... m_{t-W}``: the deployed
    recurrence has order ``L*(W-1)``, not one. `recurrence_gain` perturbs
    ``mem_in[t]`` and reads ``m[t]``, which is the FIRST-LAG BLOCK alone. For a
    higher-order system the decay is governed by the companion matrix's spectral
    radius, which can sit near 1 while every individual lag block is ~0.1 --
    so raising the first-lag block to the 24th power is simply the wrong sum.

    Measured by making ``m_p`` a leaf, running the closed loop forward to q, and
    back-propagating random unit probes. Returns the largest ``||J v||`` found,
    a lower bound on the operator norm -- enough to separate 1e-23 from ~1.
    """
    d = model.cfg.d_model
    caches = model.make_caches(1, device=tokens.device, dtype=model.embed.weight.dtype)
    m_prev = torch.zeros(1, 1, d, dtype=model.embed.weight.dtype)
    leaf = None
    for step in range(q + 1):
        mem_t = m_prev
        if step == p + 1:
            # m_p enters the loop here; make it the independent variable.
            leaf = m_prev.detach().clone().requires_grad_(True)
            mem_t = leaf
        _, m_t, _ = model.forward_decoder(
            tokens[:1, step:step + 1], mem_t,
            positions=torch.arange(step, step + 1), caches=caches,
        )
        m_prev = m_t
    if leaf is None:
        return float("nan")

    g = torch.Generator().manual_seed(0)
    best = 0.0
    for _ in range(n_probe):
        u = torch.randn(d, generator=g)
        u = u / u.norm()
        (grad,) = torch.autograd.grad(
            (m_prev[0, 0] * u).sum(), leaf, retain_graph=True
        )
        best = max(best, float(grad.norm()))
    return best


def lag_profile(model: MaglevModel, tokens: torch.Tensor, p: int,
                max_lag: int = 10) -> List[float]:
    """||d m_{p+k} / d m_p|| for k = 1..max_lag: the ORDER of the recurrence.

    A first-order chain would show one nonzero lag and then pure decay by the
    per-step gain. If the sliding cache really does make ``m_t`` depend on
    ``m_{t-1} .. m_{t-W}`` directly, the first W lags fall off far more slowly
    than the first-lag block compounded, and that gap IS the effect that lets a
    model with first-lag gain 0.1 carry information 24 steps.

    This tests the claim instead of asserting it: `stacked reach = L*(W-1) = 6`
    predicts where the slow region ends.
    """
    return [rollout_gain(model, tokens, p, p + k) for k in range(1, max_lag + 1)]


def value_signal(model: MaglevModel, cfg, p: int, q: int) -> Dict[str, Any]:
    """Does the ANSWER survive in memory? Measured with the task's own signal.

    Every gradient measure above is a linearisation, and a linearisation is blind
    to a saturated channel. If a model stores the value as WHICH BASIN its memory
    settles into, small perturbations decay back -- Jacobian ~ 0 -- while the
    discrete state is copied forward perfectly. `plain_s2` looks exactly like
    that: closed-loop sensitivity 2e-7, yet deployed accuracy 1.000 at this very
    distance. 2e-7 cannot drive a confident readout; a latch can.

    So this stops perturbing infinitesimally and asks the question the task asks.
    Two sequences identical in every token except the needle VALUE at p+1 are run
    through the deployed loop, and ``||m_t^A - m_t^B||`` is tracked, normalised by
    the memory's own scale. If that separation is created at p+1 and still stands
    at q, memory carries the answer -- whatever the Jacobian says.
    """
    v = SyntheticVocab.from_config(cfg.data)
    T = cfg.data.seq_len
    g = torch.Generator().manual_seed(11)
    base = torch.randint(v.filler_lo, v.size, (1, T), generator=g)
    base[0, q] = v.QUERY
    base[0, p] = v.key_lo + 0          # needle key
    base[0, q + 1] = v.key_lo + 0      # query repeats the key
    a, b = base.clone(), base.clone()
    a[0, p + 1] = v.value_lo + 0       # the only difference between the two
    b[0, p + 1] = v.value_lo + 1
    a[0, q + 2], b[0, q + 2] = v.value_lo + 0, v.value_lo + 1

    with torch.no_grad():
        ma = model.forward_recurrent(a).m[0]
        mb = model.forward_recurrent(b).m[0]
        sep = (ma - mb).norm(dim=-1)
        scale = 0.5 * (ma.norm(dim=-1) + mb.norm(dim=-1))
        rel = (sep / scale.clamp_min(1e-12)).tolist()

        # And does it reach the output? The answer is scored at q+1.
        la = model.forward_recurrent(a).logits[0, q + 1]
        lb = model.forward_recurrent(b).logits[0, q + 1]
        va, vb = v.value_lo + 0, v.value_lo + 1
        margin_a = float(la[va] - la[vb])
        margin_b = float(lb[vb] - lb[va])
    return {"rel": rel, "at_needle": rel[p + 1], "at_query": rel[q],
            "before_needle": rel[max(0, p - 2)],
            "margin_a": margin_a, "margin_b": margin_b,
            "correct": margin_a > 0 and margin_b > 0}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="carry_runs")
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, over in CELLS:
        trainer, cfg = train_or_load(name, over, out_dir)
        tokens, p, q = fixed_distance_batch(cfg)
        model = trainer.model.eval()
        gains = profile(model, tokens)
        end_to_end = rollout_gain(model, tokens, p, q)
        lags = lag_profile(model, tokens, p)
        signal = value_signal(model, cfg, p, q)

        before = gains[:p] or [float("nan")]
        carry = gains[p:q]
        after = gains[q:] or [float("nan")]
        rec = {
            "name": name, "needle": p, "query": q,
            "gains": gains,
            "mean_before": sum(before) / len(before),
            "mean_carry": sum(carry) / len(carry),
            "mean_after": sum(after) / len(after),
            "max_carry": max(carry),
            "min_carry": min(carry),
            # What the recorded diagnostic samples: t/4, t/2, 3t/4.
            "three_position_rho": sum(
                gains[i] for i in (len(gains) // 4, len(gains) // 2,
                                   3 * len(gains) // 4)) / 3,
            # What the first-lag block PREDICTS over this distance, vs what the
            # closed loop actually transmits.
            "predicted_from_rho": None,
            "end_to_end": end_to_end,
            "lags": lags,
            "signal": signal,
        }
        rec["predicted_from_rho"] = rec["three_position_rho"] ** (q - p)
        results[name] = rec
        (out_dir / f"{name}_profile.json").write_text(json.dumps(rec, indent=2),
                                                      encoding="utf-8")
        print(f"  {name}: carry {rec['mean_carry']:.3f} "
              f"vs three-position {rec['three_position_rho']:.3f}", flush=True)

    print(f"\nneedle at {results[CELLS[0][0]]['needle']}, "
          f"query at {results[CELLS[0][0]]['query']}, distance {DISTANCE}\n")
    print(f"{'cell':<12} {'before':>9} {'CARRY':>9} {'after':>9} "
          f"{'3-pos rho':>10} {'carry/3pos':>11}")
    print("-" * 64)
    for name, _ in CELLS:
        r = results[name]
        ratio = r["mean_carry"] / max(r["three_position_rho"], 1e-12)
        print(f"{name:<12} {r['mean_before']:>9.3f} {r['mean_carry']:>9.3f} "
              f"{r['mean_after']:>9.3f} {r['three_position_rho']:>10.3f} "
              f"{ratio:>11.1f}x")

    print(f"\nfirst-lag block vs the closed loop over {DISTANCE} steps")
    print(f"{'cell':<12} {'rho^D (predicted)':>20} {'measured |dm_q/dm_p|':>22} "
          f"{'understated by':>16}")
    print("-" * 74)
    for name, _ in CELLS:
        r = results[name]
        pred, meas = r["predicted_from_rho"], r["end_to_end"]
        factor = meas / pred if pred > 0 else float("inf")
        print(f"{name:<12} {pred:>20.2e} {meas:>22.2e} {factor:>15.1e}x")

    print(f"\n{'cell':<12} per-position gain, needle marked ^")
    for name, _ in CELLS:
        r = results[name]
        bar = "".join(
            "#" if g > 0.8 else "+" if g > 0.4 else "-" if g > 0.1 else "."
            for g in r["gains"]
        )
        mark = " " * r["needle"] + "^" + " " * (r["query"] - r["needle"] - 1) + "|"
        print(f"{name:<12} {bar}")
        if name == CELLS[0][0]:
            print(f"{'':<12} {mark}")
    print("\nlegend  # >0.8   + >0.4   - >0.1   . <=0.1"
          "     ^ needle   | query")

    print("\n||d m_(p+k) / d m_p|| by lag k -- stacked reach L*(W-1) = 6")
    print(f"{'cell':<10}" + "".join(f"k={k:<9}" for k in range(1, 11)))
    print("-" * 112)
    for name, _ in CELLS:
        print(f"{name:<10}" + "".join(f"{g:<11.2e}" for g in results[name]["lags"]))
    print("\nfirst order would give lag k = (lag 1)^k:")
    for name, _ in CELLS:
        L = results[name]["lags"]
        print(f"  {name:<10} measured lag6 {L[5]:.2e}   (lag1)^6 {L[0] ** 6:.2e}   "
              f"ratio {L[5] / max(L[0] ** 6, 1e-300):.1e}")

    print("\nTASK SIGNAL: two sequences differing only in the needle value")
    print(f"{'cell':<10} {'before needle':>14} {'at needle':>11} {'at query':>10} "
          f"{'margin':>9} {'correct':>8}")
    print("-" * 66)
    for name, _ in CELLS:
        s = results[name]["signal"]
        print(f"{name:<10} {s['before_needle']:>14.2e} {s['at_needle']:>11.4f} "
              f"{s['at_query']:>10.4f} {min(s['margin_a'], s['margin_b']):>9.2f} "
              f"{'yes' if s['correct'] else 'NO':>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
