import sys, json
sys.path.insert(0, r"c:/Users/UseR/Downloads/A")
import torch
from pathlib import Path
from rwc.config import config_from_dict
from rwc.model.maglev import MaglevModel
from rwc.analysis import diagnose_recurrence

R = Path(sys.argv[1])
print(f"{'configuration':<24} {'lam':>6} {'sg':>3} {'rho':>7} {'ablate':>9} "
      f"{'g_rec':>7} {'par':>6} {'dep':>6}  verdict")
print("-" * 100)
for name in sys.argv[2:]:
    f = R / name / "result.json"
    if not f.exists():
        print(f"{name:<24} (pending)"); continue
    r = json.loads(f.read_text()); cfg = config_from_dict(r["config"], cuda_available=False)
    ck = torch.load(R / name / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = MaglevModel(cfg.model).double().eval()
    model.load_state_dict({k: v.double() for k, v in ck["model"].items()})
    g = torch.Generator().manual_seed(0)
    tok = torch.randint(0, cfg.model.vocab_size, (2, 48), generator=g)
    d = diagnose_recurrence(model, tok)
    par = sum(v["acc"] for v in r["parallel"].values())/len(r["parallel"])
    dep = sum(v["acc"] for v in r["deployed"].values())/len(r["deployed"])
    sg = "Y" if cfg.loss.detach_target else "n"
    print(f"{name:<24} {cfg.loss.lam:>6} {sg:>3} {d.rho:>7.3f} {d.ablation_delta:>9.2e} "
          f"{d.g_rec:>7.3f} {par:>6.3f} {dep:>6.3f}  {d.verdict()}")
