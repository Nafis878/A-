"""Is the k-th Jacobi iterate exactly the closed-loop memory for t <= k?

Maglev's training pass feeds mem_in_t = m'_{t-1} (teacher) at every position, in
one parallel pass. Inference feeds m_{t-1} (the decoder's own). The claim tested
here: iterating the parallel pass, feeding back the PREVIOUS ITERATE's memory,

    m^(1)_t = P(x_t, m'_{t-1})            <- exactly Maglev's pass
    m^(j)_t = P(x_t, m^(j-1)_{t-1})

reproduces the true closed loop exactly for t <= j, in j PARALLEL passes rather
than T sequential steps. If so, closed-loop error can be trained against without
giving up the parallel training that is Maglev's whole point.
"""
import sys, json
sys.path.insert(0, r"c:/Users/UseR/Downloads/A")
import torch
from pathlib import Path
from rwc.config import config_from_dict
from rwc.train import Trainer
from rwc.model.maglev import MaglevModel

R = Path(sys.argv[1]); name = sys.argv[2]
r = json.loads((R / name / "result.json").read_text())
cfg = config_from_dict(r["config"], cuda_available=False)
t = Trainer(cfg, device=torch.device("cpu"), run_dir=R / name)
assert t.maybe_resume()
model = t.model.double().eval()
tokens = t.data.batch(step=7, batch_size=4).tokens

with torch.no_grad():
    truth = model.forward_recurrent(tokens).m           # sequential ground truth
    m_prime = model.forward_prefiller(tokens)
    prev = m_prime
    print(f"{'k':>3} {'max|m^(k)-m_closed| for t<k':>28} {'over all t':>13}")
    for k in range(1, 9):
        _, cur, _ = model.forward_decoder(tokens, MaglevModel.shift_memory(prev))
        head = (cur[:, :k] - truth[:, :k]).abs().max().item()
        allt = (cur - truth).abs().max().item()
        print(f"{k:>3} {head:>28.2e} {allt:>13.2e}")
        prev = cur
