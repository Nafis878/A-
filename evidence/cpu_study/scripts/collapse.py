"""Is the constant memory a collapse, or did it simply never start?

If a randomly initialised prefiller already produced constant memory, the
lambda>=0.3 measurement would say nothing. It does not: the control below fixes
the baseline. Also reports the information actually carried by the memory --
the spread of m' across positions, which is what a constant solution destroys.
"""
import sys, json
sys.path.insert(0, r"c:/Users/UseR/Downloads/A")
import torch
from pathlib import Path
from rwc.config import config_from_dict
from rwc.model.maglev import MaglevModel
from rwc.analysis import prefiller_locality

R = Path(sys.argv[1])

def stats(m, tok, label, lam):
    with torch.no_grad():
        mp = m.forward_prefiller(tok)
        # Spread of m' ACROSS positions, relative to its own scale. A constant
        # trajectory carries no information no matter how large its norm.
        spread = float(mp.std(dim=1).mean() / mp.norm(dim=-1).mean().clamp_min(1e-12))
        rank = float(torch.linalg.matrix_rank(
            (mp[0] - mp[0].mean(0)).double(), rtol=1e-3))
    loc = prefiller_locality(m, tok)
    print(f"{label:<26} {lam:>6} {spread:>10.5f} {rank:>6.0f} "
          f"{loc['far_delta']:>9.4f} {loc['near_delta']:>9.4f}")

print(f"{'configuration':<26} {'lam':>6} {'m-spread':>10} {'rank':>6} "
      f"{'far':>9} {'near':>9}")
print("-" * 72)

cfg0 = config_from_dict(
    json.loads((R / "pilot_maglev_lam0" / "result.json").read_text())["config"],
    cuda_available=False)
g = torch.Generator().manual_seed(0)
tok = torch.randint(0, cfg0.model.vocab_size, (4, 96), generator=g)

torch.manual_seed(0)
stats(MaglevModel(cfg0.model).double().eval(), tok, "RANDOM INIT (control)", "-")

for name in sys.argv[2:]:
    f = R / name / "result.json"
    if not f.exists():
        continue
    cfg = config_from_dict(json.loads(f.read_text())["config"], cuda_available=False)
    ck = torch.load(R / name / "checkpoint.pt", map_location="cpu", weights_only=False)
    m = MaglevModel(cfg.model).double().eval()
    m.load_state_dict({k: v.double() for k, v in ck["model"].items()})
    stats(m, tok, name, cfg.loss.lam)
