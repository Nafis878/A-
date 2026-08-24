"""Does CL-RWC shrink the closed-loop deviation it targets?

Separates two questions that are easy to conflate:
  (a) does the method do what it claims mechanically -- shrink
      delta_t = m_free_t - m'_t, the deviation Maglev never trains against;
  (b) does that translate into deployed probe accuracy.

(a) can hold while (b) fails. Reporting only (b) would hide that.
"""
import sys, json
sys.path.insert(0, r"c:/Users/UseR/Downloads/A")
import torch
from pathlib import Path
from rwc.config import config_from_dict
from rwc.train import Trainer

R = Path(sys.argv[1])
print(f"{'run':<20} {'lam':>5} {'k':>3} {'||e||':>8} {'||delta||':>10} "
      f"{'delta/e':>8} {'delta/||m*||':>12} {'dep':>6}")
print("-" * 82)
for name in sys.argv[2:]:
    f = R / name / "result.json"
    if not f.exists():
        print(f"{name:<20} (not finished)"); continue
    r = json.loads(f.read_text())
    cfg = config_from_dict(r["config"], cuda_available=False)
    t = Trainer(cfg, device=torch.device("cpu"), run_dir=R / name)
    assert t.maybe_resume(), name
    t.model.eval()
    b = t.data.batch(step=10**6, batch_size=32)
    with torch.no_grad():
        out = t.model.forward_train(b.tokens)
        free = t.model.forward_recurrent(b.tokens)
        e = (out.m - out.m_prime).norm(dim=-1).mean()
        dl = (free.m - out.m_prime).norm(dim=-1).mean()
        sc = out.m_prime.norm(dim=-1).mean()
    dep = sum(v["acc"] for v in r["deployed"].values()) / len(r["deployed"])
    print(f"{name:<20} {cfg.loss.lam:>5} {cfg.loss.unroll_k:>3} {e:>8.3f} {dl:>10.3f} "
          f"{dl/e.clamp_min(1e-9):>8.2f} {dl/sc:>12.3f} {dep:>6.3f}")
