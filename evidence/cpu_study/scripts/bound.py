"""Is the contraction bound satisfied but VACUOUS?

delta_t = e_t + J_t delta_{t-1}  =>  ||delta|| <= ||e|| / (1 - rho)  for rho < 1.

rho < 1 has now been achieved (0.760 at target gain 0.5) with deployed accuracy
still at chance, so rho alone is not sufficient. The bound has a second factor:
the one-step residual e must also be SMALL relative to the memory scale, or the
guarantee permits a deviation larger than the memory itself.

Reports e, the predicted bound e/(1-rho), and both against ||m'||. If the
predicted bound exceeds the memory scale, the guarantee is vacuous -- satisfied
and useless -- which would explain why deployed accuracy never moves.
"""
import sys, json
sys.path.insert(0, r"c:/Users/UseR/Downloads/A")
import torch
from pathlib import Path
from rwc.config import config_from_dict
from rwc.model.maglev import MaglevModel
from rwc.analysis import recurrence_gain

R = Path(sys.argv[1])
print(f"{'configuration':<22} {'lam':>5} {'rho':>6} {'|e|':>7} {'|m*|':>7} "
      f"{'e/|m*|':>7} {'bound':>8} {'bound/|m*|':>11}  guarantee")
print("-" * 100)
for name in sys.argv[2:]:
    f = R / name / "result.json"
    if not f.exists():
        continue
    r = json.loads(f.read_text()); cfg = config_from_dict(r["config"], cuda_available=False)
    ck = torch.load(R / name / "checkpoint.pt", map_location="cpu", weights_only=False)
    m = MaglevModel(cfg.model).double().eval()
    m.load_state_dict({k: v.double() for k, v in ck["model"].items()})
    g = torch.Generator().manual_seed(0)
    tok = torch.randint(0, cfg.model.vocab_size, (8, 48), generator=g)
    with torch.no_grad():
        out = m.forward_train(tok)
        e = float((out.m - out.m_prime).norm(dim=-1).mean())
        scale = float(out.m_prime.norm(dim=-1).mean())
        mem_in = MaglevModel.shift_memory(out.m_prime)
    rho, _ = recurrence_gain(m, tok, mem_in)
    if rho < 1:
        bound = e / (1 - rho)
        verdict = "USEFUL" if bound < scale else "VACUOUS (bound > memory scale)"
        bs = f"{bound:>8.2f}"; br = f"{bound/scale:>11.2f}"
    else:
        bound = float("inf"); verdict = "NONE (rho >= 1, error compounds)"
        bs = f"{'inf':>8}"; br = f"{'inf':>11}"
    print(f"{name:<22} {cfg.loss.lam:>5} {rho:>6.3f} {e:>7.2f} {scale:>7.2f} "
          f"{e/scale:>7.2f} {bs} {br}  {verdict}")
