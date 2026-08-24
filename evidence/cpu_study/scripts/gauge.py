"""Pre-registration gate for the gauge argument.

Memory enters the decoder only through the layer-specific read maps W_k_rec and
W_v_rec. So the behavioural size of a memory perturbation eps is ||R eps||, with
R the stacked read operator -- not ||eps||. Uniform L2 penalises eps^T I eps;
the behaviourally correct metric is eps^T G eps with G = R^T R.

If G were near-isotropic the two metrics would agree and the whole argument
would be empty. This measures how far from isotropic G actually is, and how much
of it a per-dimension (diagonal) weighting can capture -- which is all RWC can
ever represent.
"""
import sys, json
sys.path.insert(0, r"c:/Users/UseR/Downloads/A")
import torch
from pathlib import Path
from rwc.config import config_from_dict
from rwc.model.maglev import MaglevModel

R = Path(sys.argv[1])
print(f"{'run':<22} {'d':>4} {'cond(G)':>10} {'eff.rank':>9} {'top1%':>7} {'diag frac':>10}")
print("-" * 68)
for name in sys.argv[2:]:
    f = R / name / "result.json"
    if not f.exists():
        continue
    cfg = config_from_dict(json.loads(f.read_text())["config"], cuda_available=False)
    # Load weights directly: these checkpoints predate two LossConfig fields, so
    # the trainer's config-hash guard fires. Only the read maps are needed here.
    ck = torch.load(R / name / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = MaglevModel(cfg.model)
    model.load_state_dict(ck["model"])
    rows = []
    for inj in model.injections:
        rows += [inj.W_k_rec.weight.detach(), inj.W_v_rec.weight.detach()]
    Rop = torch.cat(rows, dim=0).double()          # (2L*d_kv, d)
    G = Rop.T @ Rop
    ev = torch.linalg.eigvalsh(G).clamp_min(0)
    p = ev / ev.sum()
    eff = float(torch.exp(-(p * p.clamp_min(1e-300).log()).sum()))
    d = G.shape[0]
    # How much of G's action a diagonal metric can reproduce: ||diag(G)||/||G||.
    diag_frac = float(G.diag().norm() / G.norm())
    print(f"{name:<22} {d:>4} {ev.max()/ev.min().clamp_min(1e-12):>10.1f} "
          f"{eff:>9.1f} {p.max()*100:>6.1f}% {diag_frac:>9.3f}")
