"""Is the recurrence switched off at the gate, and does memory still matter?

Eq. 8 mixes local and recurrent K/V as  k_bar = g_loc*k + g_rec*k_rec.
If the consistency loss drives g_rec -> 0, the memory channel is closed and the
model degenerates to sliding-window attention -- which satisfies the consistency
term trivially, because m_t then becomes a deterministic function of local
context that m'_t can also be matched to, with nothing carried across steps.

Ablation check: zero the memory input entirely and see whether outputs move.
If they do not, the trained model is not using memory at all.
"""
import sys, json
sys.path.insert(0, r"c:/Users/UseR/Downloads/A")
import torch
from pathlib import Path
from rwc.config import config_from_dict
from rwc.model.maglev import MaglevModel

R = Path(sys.argv[1])
print(f"{'configuration':<22} {'g_rec':>8} {'g_loc':>8} {'g_rec/g_loc':>12} "
      f"{'|d logits| if mem=0':>20} {'parallel':>9}")
print("-" * 86)
for name in sys.argv[2:]:
    f = R / name / "result.json"
    if not f.exists():
        continue
    r = json.loads(f.read_text())
    cfg = config_from_dict(r["config"], cuda_available=False)
    ck = torch.load(R / name / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = MaglevModel(cfg.model).double().eval()
    model.load_state_dict({k: v.double() for k, v in ck["model"].items()})

    gen = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, cfg.model.vocab_size, (4, 96), generator=gen)
    with torch.no_grad():
        out = model.forward_train(tokens, need_mass=True)
        gr = torch.stack([a.g_rec.mean() for a in out.aux]).mean()
        # g_loc is not returned; recompute it from the same normalised residual.
        mem_in = MaglevModel.shift_memory(out.m_prime)
        zero, _, _ = model.forward_decoder(tokens, torch.zeros_like(mem_in))
        delta = (out.logits - zero).abs().max()
        gl = torch.stack([
            (2 * torch.sigmoid(model.injections[i].G_loc(
                model.decoder_blocks[i].attn_norm(model.embed(tokens))))).mean()
            for i in range(cfg.model.n_layers)
        ]).mean()
    par = sum(v["acc"] for v in r["parallel"].values()) / len(r["parallel"])
    print(f"{name:<22} {gr:>8.4f} {gl:>8.4f} {gr/gl:>12.4f} {delta:>20.3e} {par:>9.3f}")
