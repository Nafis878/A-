"""Main table: does SHARED sever the memory write path at a lower lambda than SEPARATE?

Maglev attributes its lambda split to capacity -- "when Q and P share most
parameters, an overly strong memory-consistency term can constrain the decoder
representation". The collapse account predicts something checkable instead: the
degenerate solution (m' constant, decoder ignoring memory, consistency exactly
zero) is reachable by moving ONE weight set when Q and P are shared, so shared
should lose rho(dm_t/dm_{t-1}) at a lower lambda than separate.

Capacity predicts a magnitude difference. Collapse predicts a difference in the
lambda at which propagation dies. Those are distinguishable, and rho is
deterministic given a checkpoint -- unlike accuracy, which is at chance here.
"""
import sys, json
sys.path.insert(0, r"c:/Users/UseR/Downloads/A")
import torch
from pathlib import Path
from rwc.config import config_from_dict
from rwc.model.maglev import MaglevModel
from rwc.analysis import diagnose_recurrence, memory_information, prefiller_locality

R = Path(r"C:/Users/UseR/AppData/Local/Temp/claude/c--Users-UseR-Downloads-A/33c40683-2cc9-4597-b751-2ee8ef01368d/scratchpad/exp/runs")
LAMS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)

def load(name):
    f = R / name / "result.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    cfg = config_from_dict(r["config"], cuda_available=False)
    ck = torch.load(R / name / "checkpoint.pt", map_location="cpu", weights_only=False)
    m = MaglevModel(cfg.model).double().eval()
    m.load_state_dict({k: v.double() for k, v in ck["model"].items()})
    return r, cfg, m

g = torch.Generator().manual_seed(0)
tok96 = torch.randint(0, 64, (4, 96), generator=g)

for sharing in ("shared", "separate"):
    print(f"\n=== {sharing.upper()} ===")
    print(f"{'lam':>6} {'ce':>7} {'par':>7} {'dep':>7} | {'rho':>7} {'g_rec':>7} "
          f"{'ablate':>9} {'spread':>9} {'far/near':>9}  state")
    print("-" * 96)
    for lam in LAMS:
        got = load(f"an_{sharing}_lam{lam}")
        if got is None:
            print(f"{lam:>6}   (pending)"); continue
        r, cfg, m = got
        par = sum(v["acc"] for v in r["parallel"].values())/len(r["parallel"])
        dep = sum(v["acc"] for v in r["deployed"].values())/len(r["deployed"])
        d = diagnose_recurrence(m, tok96[:2, :48])
        info = memory_information(m, tok96)
        loc = prefiller_locality(m, tok96, n_probes=8)
        state = ("collapsed" if info["spread"] < 1e-3 else
                 "write severed" if d.rho < 0.05 else "chain live")
        print(f"{lam:>6} {r['final_ce']:>7.3f} {par:>7.3f} {dep:>7.3f} | {d.rho:>7.3f} "
              f"{d.g_rec:>7.3f} {d.ablation_delta:>9.2e} {info['spread']:>9.5f} "
              f"{loc['far_over_near']:>9.4f}  {state}")
