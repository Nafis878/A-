"""Does gradient from the query reach the planting site THROUGH THE CHAIN?

Must be measured in the closed loop. In the parallel graph the memory chain does
not exist -- mem_in is an independent input per position -- so the gradient there
just recovers the stacked local reach L*(W-1) and is exactly zero beyond it.
grad_influence rolls the decoder forward with the graph retained, which is the
recurrence the deployed model actually runs.

Maglev injects memory into the attention K/V (Eq. 8) and recomputes
m_t = decoder_norm(h_t) from scratch each step: there is no identity path from
m_{t-1} to m_t. That is a vanilla RNN, so gradient should decay like rho^D.
"""
import sys, json
sys.path.insert(0, r"c:/Users/UseR/Downloads/A")
import torch
from pathlib import Path
from rwc.config import config_from_dict
from rwc.model.maglev import MaglevModel
from rwc.analysis import recurrence_gain
from rwc.influence import grad_influence

R = Path(sys.argv[1])
for name in sys.argv[2:]:
    f = R / name / "result.json"
    if not f.exists():
        continue
    r = json.loads(f.read_text()); cfg = config_from_dict(r["config"], cuda_available=False)
    ck = torch.load(R / name / "checkpoint.pt", map_location="cpu", weights_only=False)
    m = MaglevModel(cfg.model).double().eval()
    m.load_state_dict({k: v.double() for k, v in ck["model"].items()})

    T = min(cfg.data.seq_len, 48)
    gen = torch.Generator().manual_seed(0)
    tok = torch.randint(0, cfg.model.vocab_size, (2, T), generator=gen)
    tgt = torch.full_like(tok, -100); tgt[:, :-1] = tok[:, 1:]
    with torch.no_grad():
        mem0 = MaglevModel.shift_memory(m.forward_prefiller(tok))
    rho, _ = recurrence_gain(m, tok, mem0)

    print(f"\n=== {name}  rho={rho:.4f}  reach L*(W-1)={cfg.model.local_reach} ===")
    print(f"{'D':>4} {'||dL/dm_j|| (closed loop)':>26} {'ratio':>10} {'rho^D':>11}")
    print("-" * 56)
    base = None
    for D in (1, 2, 4, 8, 16, 24, 32):
        j = T - 2 - D
        if j < 1:
            continue
        w = grad_influence(m, tok, tgt, horizon_H=D, anchors=[j],
                           normalise_out=False)
        gn = float(w.sum())
        if base is None:
            base = gn
        print(f"{D:>4} {gn:>26.3e} {gn/max(base,1e-30):>10.3e} {rho**D:>11.3e}")
