#!/usr/bin/env bash
# Remaining studies, in value order, each retried on death.
#
# Background jobs on this machine get killed periodically -- the ladder died once
# mid-run, and the breadth study and autosave died together later, all without
# error output. Every study caches completed cells by result.json and resumes
# mid-cell, so a death costs at most 100 steps and re-running the chain from the
# top simply skips finished work. This wraps each study in a retry loop so a
# dead python is picked up without waiting for a human.
#
# lm_induction runs BEFORE depth: the aggregate-BPB null made natural-language
# retrieval the question that decides how this work is positioned, while depth
# closes an objection the width ladder has already largely answered.
cd "$(dirname "$0")"

run_until_done () {           # name, expected_results, dir, command...
  local name="$1" want="$2" dir="$3"; shift 3
  local tries=0
  while [ "$(ls "$dir"/*/result.json 2>/dev/null | wc -l)" -lt "$want" ]; do
    tries=$((tries + 1))
    if [ "$tries" -gt 40 ]; then
      echo "[chain] $name: giving up after $tries attempts"; return 1
    fi
    echo "[chain] $name attempt $tries $(date '+%H:%M')"
    "$@"
    sleep 5
  done
  echo "[chain] $name complete $(date '+%H:%M')"
}

run_until_done breadth 30 breadth_runs \
  python -u scripts/variant_study.py --study breadth --out-dir breadth_runs
run_until_done lm_induction 15 lm_ind_runs \
  python -u scripts/lm_induction.py --out-dir lm_ind_runs
run_until_done depth 30 depth_runs \
  python -u scripts/variant_study.py --study depth --out-dir depth_runs
echo "[chain] ALL DONE $(date '+%H:%M')"
