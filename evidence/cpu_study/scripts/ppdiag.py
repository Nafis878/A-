import sys, json
sys.path.insert(0, r"c:/Users/UseR/Downloads/A")
import torch
from pathlib import Path
from rwc.config import config_from_dict
from rwc.model.maglev import MaglevModel
from rwc.analysis import diagnose_recurrence, memory_information

R = Path(sys.argv[1])
print(f"{'configuration':<22} {'lam':>5} {'pw':>5} {'ce':>7} {'par':>6} | "
      f"{'rho':>7} {'g_rec':>7} {'ablate':>9} {'spread':>9}  state")
print("-" * 92)
g = torch.Generator().manual_seed(0)
for name in sys.argv[2:]:
    f = R / name / "result.json"
    if not f.exists():
        print(f"{name:<22} (pending)"); continue
    r = json.loads(f.read_text())
    cfg = config_from_dict(r["config"], cuda_available=False)
    ck = torch.load(R / name / "checkpoint.pt", map_location="cpu", weights_only=False)
    m = MaglevModel(cfg.model).double().eval()
    m.load_state_dict({k: v.double() for k, v in ck["model"].items()})
    # Probe length must respect the model's RoPE cache, which is sized to
    # max_seq_len; a longer probe indexes past it.
    T = min(96, cfg.model.max_seq_len)
    tok = torch.randint(0, cfg.model.vocab_size, (4, T),
                        generator=torch.Generator().manual_seed(0))
    d = diagnose_recurrence(m, tok[:2])
    info = memory_information(m, tok)
    par = sum(v["acc"] for v in r["parallel"].values())/len(r["parallel"])
    state = ("collapsed" if info["spread"] < 1e-3 else
             "write severed" if d.rho < 0.05 else "chain live")
    print(f"{name:<22} {cfg.loss.lam:>5} {cfg.loss.propagation_weight:>5} "
          f"{r['final_ce']:>7.3f} {par:>6.3f} | {d.rho:>7.3f} {d.g_rec:>7.3f} "
          f"{d.ablation_delta:>9.2e} {info['spread']:>9.5f}  {state}")
