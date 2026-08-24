"""Assemble the reviewer-facing table: accuracy beside the mechanism."""
import sys, json
sys.path.insert(0, r"c:/Users/UseR/Downloads/A")
import torch
from pathlib import Path
from rwc.config import config_from_dict
from rwc.model.maglev import MaglevModel
from rwc.analysis import diagnose_recurrence, memory_information, wilson_interval
from rwc.evaluate import probe_accuracy
from rwc.data.synthetic import SyntheticGenerator

R = Path(r"C:/Users/UseR/AppData/Local/Temp/claude/c--Users-UseR-Downloads-A/33c40683-2cc9-4597-b751-2ee8ef01368d/scratchpad/exp/runs")

def row(name, label):
    f = R / name / "result.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text()); cfg = config_from_dict(r["config"], cuda_available=False)
    ck = torch.load(R / name / "checkpoint.pt", map_location="cpu", weights_only=False)
    m = MaglevModel(cfg.model).double().eval()
    m.load_state_dict({k: v.double() for k, v in ck["model"].items()})
    T = min(48, cfg.model.max_seq_len)
    tok = torch.randint(0, cfg.model.vocab_size, (2, T),
                        generator=torch.Generator().manual_seed(0))
    d = diagnose_recurrence(m, tok); info = memory_information(m, tok)

    gen = SyntheticGenerator(cfg.data, cfg.model, seed=999)
    batches = gen.eval_batches(batch_size=64, n_per_distance=64)
    with torch.no_grad():
        live = probe_accuracy(lambda x: m.forward_recurrent(x).logits, batches)
        dead = probe_accuracy(lambda x: m.forward_recurrent(
            x, mem_in_override=torch.zeros(x.shape[0], x.shape[1],
                                           cfg.model.d_model, dtype=torch.float64)).logits,
            batches)
    lm = sum(v["acc"] for v in live.values())/len(live)
    dm = sum(v["acc"] for v in dead.values())/len(dead)
    par = sum(v["acc"] for v in r["parallel"].values())/len(r["parallel"])
    n = sum(v["n"] for v in live.values()); ch = 1/cfg.data.n_values
    hi = wilson_interval(round(ch*n), n)[1]
    return dict(label=label, ce=r["final_ce"], par=par, dep=lm, dep_ablated=dm,
                contrib=lm-dm, rho=d.rho, spread=info["spread"],
                ablate=d.ablation_delta, chance=ch, hi=hi,
                mode=cfg.optim.train_mode, res=cfg.model.memory_residual)

ROWS = [
    ("r_bptt_plain", "Maglev write rule, BPTT"),
    ("r_bptt_res",   "+ identity path, BPTT"),
    ("r_par_res",    "+ identity path, teacher-forced"),
    ("r_bptt_res_b4","+ identity path (bias 4.0), BPTT"),
]
print(f"{'configuration':<36} {'mode':>6} {'ce':>7} {'par':>6} {'dep':>6} "
      f"{'dep-abl':>8} {'memory':>7} | {'rho':>6} {'spread':>8} {'ablate':>9}")
print("-" * 112)
for name, label in ROWS:
    r = row(name, label)
    if r is None:
        print(f"{label:<36} (pending)"); continue
    print(f"{r['label']:<36} {r['mode']:>6} {r['ce']:>7.4f} {r['par']:>6.3f} "
          f"{r['dep']:>6.3f} {r['dep_ablated']:>8.3f} {r['contrib']:>+7.3f} | "
          f"{r['rho']:>6.3f} {r['spread']:>8.5f} {r['ablate']:>9.2e}")
print(f"\nchance {ROWS and row(ROWS[0][0],'')['chance']:.3f};  "
      f"'memory' = deployed minus deployed-with-state-zeroed")
