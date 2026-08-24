"""Spectral radius of the memory recurrence.

The deployed deviation obeys  delta_t = e_t + J_t delta_{t-1},  J_t = dm_t/dm_{t-1}.
If the recurrence contracts with gain gamma < 1, then ||delta|| <= ||e||/(1-gamma):
one-step consistency BOUNDS the deployed error and Maglev's loss is sufficient.
If gamma >= 1, errors compound and no amount of one-step consistency controls the
closed loop -- the bound is vacuous.

d is small enough to form J exactly, so this is the true spectrum, not an estimate.
"""
import sys, json
sys.path.insert(0, r"c:/Users/UseR/Downloads/A")
import torch
from pathlib import Path
from rwc.config import config_from_dict
from rwc.model.maglev import MaglevModel

R = Path(sys.argv[1])
print(f"{'configuration':<22} {'rho(J)':>8} {'sigma_max':>10} {'1/(1-rho)':>10} {'contracts':>10}")
print("-" * 66)
for name in sys.argv[2:]:
    f = R / name / "result.json"
    if not f.exists():
        continue
    cfg = config_from_dict(json.loads(f.read_text())["config"], cuda_available=False)
    ck = torch.load(R / name / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = MaglevModel(cfg.model).double().eval()
    model.load_state_dict({k: v.double() for k, v in ck["model"].items()})

    gen = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, cfg.model.vocab_size, (1, 48), generator=gen)
    with torch.no_grad():
        mem = MaglevModel.shift_memory(model.forward_prefiller(tokens))

    rhos, sigmas = [], []
    for t in (12, 24, 36):
        base = mem.clone()

        def step(v):
            mi = base.clone()
            mi[:, t] = v
            _, m, _ = model.forward_decoder(tokens, mi)
            return m[:, t].squeeze(0)

        J = torch.autograd.functional.jacobian(step, base[:, t].squeeze(0).clone())
        ev = torch.linalg.eigvals(J)
        rhos.append(float(ev.abs().max()))
        sigmas.append(float(torch.linalg.svdvals(J).max()))

    rho = sum(rhos) / len(rhos); sig = sum(sigmas) / len(sigmas)
    amp = f"{1/(1-rho):.2f}" if rho < 1 else "unbounded"
    print(f"{name:<22} {rho:>8.3f} {sig:>10.3f} {amp:>10} "
          f"{('YES' if rho < 1 else 'NO'):>10}")
