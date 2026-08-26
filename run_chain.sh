#!/usr/bin/env bash
# Runs the remaining studies back to back so the CPU never idles.
# Waits on lm_runs' completion marker (10 result.json) rather than a PID, so it
# is safe to start while lm_study.py is already running and safe to re-run: every
# study caches completed cells and resumes mid-cell.
cd "$(dirname "$0")"

echo "[chain] waiting for the LM study to finish..."
while [ "$(ls lm_runs/*/result.json 2>/dev/null | wc -l)" -lt 10 ]; do
  sleep 120
done
echo "[chain] LM done at $(date '+%H:%M')"

echo "[chain] breadth starting $(date '+%H:%M')"
python -u scripts/variant_study.py --study breadth --out-dir breadth_runs >> breadth.log 2>&1
echo "[chain] breadth done $(date '+%H:%M')"

echo "[chain] depth starting $(date '+%H:%M')"
python -u scripts/variant_study.py --study depth --out-dir depth_runs >> depth.log 2>&1
echo "[chain] depth done $(date '+%H:%M')"
