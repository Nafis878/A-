import sys, json
sys.path.insert(0, r"c:/Users/UseR/Downloads/A")
import torch
from pathlib import Path
from rwc.config import config_from_dict
from rwc.model.maglev import MaglevModel
from rwc.analysis import prefiller_locality, diagnose_recurrence

R = Path(sys.argv[1])
print(f"{'configuration':<22} {'lam':>6} {'far':>8} {'near':>8} {'far/near':>9} "
      f"{'rho':>7} {'g_rec':>7}")
print("-" * 74)
for name in sys.argv[2:]:
    f = R / name / "result.json"
    if not f.exists():
        continue
    cfg = config_from_dict(json.loads(f.read_text())["config"], cuda_available=False)
    ck = torch.load(R / name / "checkpoint.pt", map_location="cpu", weights_only=False)
    m = MaglevModel(cfg.model).double().eval()
    m.load_state_dict({k: v.double() for k, v in ck["model"].items()})
    g = torch.Generator().manual_seed(0)
    tok = torch.randint(0, cfg.model.vocab_size, (4, 96), generator=g)
    loc = prefiller_locality(m, tok)
    d = diagnose_recurrence(m, tok[:2, :48])
    print(f"{name:<22} {cfg.loss.lam:>6} {loc['far_delta']:>8.3f} "
          f"{loc['near_delta']:>8.3f} {loc['far_over_near']:>9.4f} "
          f"{d.rho:>7.3f} {d.g_rec:>7.3f}")
