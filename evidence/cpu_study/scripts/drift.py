"""Does the closed-loop memory deviation compound?

delta_t = m_t - m'_t obeys, to first order,  delta_t = e_t + J_t delta_{t-1}
with e_t the one-step residual Maglev penalises and J_t = dP/dmem the memory
Jacobian. If ||delta|| grows while ||e|| stays flat, the failure is amplification
through the recurrence, not the size of the one-step residual -- and a loss that
only controls e cannot fix it.
"""
import sys, json
sys.path.insert(0, r"c:/Users/UseR/Downloads/A")
import torch
from pathlib import Path
from rwc.config import config_from_dict
from rwc.train import Trainer
from rwc.model.maglev import MaglevModel

R = Path(sys.argv[1])
for name in sys.argv[2:]:
    r = json.loads((R / name / "result.json").read_text())
    cfg = config_from_dict(r["config"], cuda_available=False)
    t = Trainer(cfg, device=torch.device("cpu"), run_dir=R / name)
    assert t.maybe_resume(), name
    t.model.eval()
    b = t.data.batch(step=10**6, batch_size=32)
    with torch.no_grad():
        out = t.model.forward_train(b.tokens)          # teacher-forced
        free = t.model.forward_recurrent(b.tokens)     # closed loop
        mp, mhat, mfree = out.m_prime, out.m, free.m
        e = (mhat - mp).norm(dim=-1)                   # one-step residual
        delta = (mfree - mp).norm(dim=-1)              # closed-loop deviation
        scale = mp.norm(dim=-1)
    print(f"\n=== {name} (lam={cfg.loss.lam}, {cfg.loss.consistency}) ===")
    print(f"{'t':>5} {'||e_t||':>9} {'||delta_t||':>12} {'delta/e':>9} {'delta/||m*||':>13}")
    T = mp.shape[1]
    for pos in [1, 2, 4, 8, 16, 32, 64, T - 1]:
        if pos >= T: continue
        ee, dd, ss = e[:, pos].mean(), delta[:, pos].mean(), scale[:, pos].mean()
        print(f"{pos:>5} {ee:>9.4f} {dd:>12.4f} {dd/ee.clamp_min(1e-9):>9.2f} {dd/ss:>13.3f}")
