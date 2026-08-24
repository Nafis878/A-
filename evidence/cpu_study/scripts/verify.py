"""Is deployed=1.000 actually carried by the memory chain?

A perfect score is exactly when to check for a shortcut. Every probe distance
(8..24) lies beyond the stacked local reach L*(W-1) = 6, so local attention
cannot reach the planted pair -- but that must be demonstrated on THIS
checkpoint, not assumed:

  1. diagnostics: is the chain live (rho, spread, ablation)?
  2. memory ablation on the DEPLOYED path: zeroing the carried state must
     destroy accuracy. If it does not, something other than memory is solving it.
"""
import sys, json
sys.path.insert(0, r"c:/Users/UseR/Downloads/A")
import torch
from pathlib import Path
from rwc.config import config_from_dict
from rwc.model.maglev import MaglevModel
from rwc.analysis import diagnose_recurrence, memory_information
from rwc.evaluate import probe_accuracy
from rwc.data.synthetic import SyntheticGenerator

R = Path(sys.argv[1])
for name in sys.argv[2:]:
    f = R / name / "result.json"
    if not f.exists():
        print(f"{name}: pending"); continue
    r = json.loads(f.read_text()); cfg = config_from_dict(r["config"], cuda_available=False)
    ck = torch.load(R / name / "checkpoint.pt", map_location="cpu", weights_only=False)
    m = MaglevModel(cfg.model).double().eval()
    m.load_state_dict({k: v.double() for k, v in ck["model"].items()})

    T = min(48, cfg.model.max_seq_len)
    tok = torch.randint(0, cfg.model.vocab_size, (2, T),
                        generator=torch.Generator().manual_seed(0))
    d = diagnose_recurrence(m, tok)
    info = memory_information(m, tok)

    # n_distractors is a CONSTRUCTOR argument, not read from cfg.data -- omit it
    # and the eval silently runs a harder distribution (default 4) than the model
    # trained on. Trainer._build_data passes it; this must too.
    gen = SyntheticGenerator(cfg.data, cfg.model, seed=999,
                             n_distractors=cfg.data.n_distractors)
    batches = gen.eval_batches(batch_size=64, n_per_distance=64)
    with torch.no_grad():
        live = probe_accuracy(lambda x: m.forward_recurrent(x).logits, batches)
        # Same deployed rollout, but the carried state is forced to zero at
        # every step: the local pathway alone, with the memory removed.
        dead = probe_accuracy(
            lambda x: m.forward_recurrent(
                x, mem_in_override=torch.zeros(x.shape[0], x.shape[1],
                                               cfg.model.d_model, dtype=torch.float64)
            ).logits, batches)
    lm = sum(v["acc"] for v in live.values())/len(live)
    dm = sum(v["acc"] for v in dead.values())/len(dead)
    print(f"\n=== {name} ===")
    print(f"  rho {d.rho:.3f}   g_rec {d.g_rec:.3f}   ablate {d.ablation_delta:.2e}   "
          f"m' spread {info['spread']:.5f}   [{d.verdict()}]")
    print(f"  deployed, memory LIVE   {lm:.3f}   " +
          "  ".join(f"D{k}:{v['acc']:.2f}" for k, v in live.items()))
    print(f"  deployed, memory ZEROED {dm:.3f}   " +
          "  ".join(f"D{k}:{v['acc']:.2f}" for k, v in dead.items()))
    print(f"  -> memory accounts for {lm-dm:+.3f} of accuracy (chance {1/cfg.data.n_values:.3f})")
