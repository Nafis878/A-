SP="C:/Users/UseR/AppData/Local/Temp/claude/c--Users-UseR-Downloads-A/33c40683-2cc9-4597-b751-2ee8ef01368d/scratchpad"
until [ -f "$SP/exp/runs/cl_k4_lam0.3/result.json" ]; do sleep 30; done
python "$SP/exp/frontier.py"
