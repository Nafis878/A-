#!/usr/bin/env bash
# Commit and push result.json files as the studies produce them.
#
# The three remaining studies run for ~35 hours and write ~70 result files. Those
# are the entire product of that compute: checkpoints are deleted per cell by
# design, so a result.json that is only on this disk is one power cut away from
# being 35 hours of nothing. This pushes them as they land instead of at the end.
#
# Stages ONLY result.json paths under the run directories -- never `git add -A` --
# so it cannot sweep up work in progress, a half-written script, or a corpus.
# Everything else in those directories is already gitignored (checkpoints,
# metrics traces, heartbeats).
#
# Safe to run alongside manual commits: if git is busy the add or commit fails,
# nothing is staged, and the next cycle picks the files up.

cd "$(dirname "$0")" || exit 1
INTERVAL=${1:-900}          # seconds between checks
EXPECTED=$((70 + 10 + 30 + 30))   # ladder + lm + breadth + depth

while true; do
  sleep "$INTERVAL"

  # One directory at a time, each guarded. `git add a/* b/*` exits 128 and
  # stages NOTHING if any single pathspec matches no files -- so while
  # breadth_runs is still empty it would silently discard the LM results it was
  # meant to be saving. Adding a directory (not a glob) also lets .gitignore do
  # the filtering, so only result.json is picked up.
  for d in ladder_runs lm_runs breadth_runs depth_runs carry_runs; do
    [ -d "$d" ] && git add "$d" 2>/dev/null
  done

  if ! git diff --cached --quiet 2>/dev/null; then
    n=$(git diff --cached --name-only | wc -l)
    total=$(git ls-files 'ladder_runs/*/result.json' 'lm_runs/*/result.json' \
            'breadth_runs/*/result.json' 'depth_runs/*/result.json' | wc -l)
    git commit -q -m "Autosave $n new result files ($total total)

Written by save_results.sh while the LM, breadth and depth studies run.
Checkpoints are deleted per cell, so these JSON files are the whole product
of the compute that produced them." && git push -q origin main
    echo "[$(date '+%H:%M')] saved $n new (total $total)"
  fi

  have=$(ls ladder_runs/*/result.json lm_runs/*/result.json \
            breadth_runs/*/result.json depth_runs/*/result.json 2>/dev/null | wc -l)
  if [ "$have" -ge "$EXPECTED" ]; then
    echo "[$(date '+%H:%M')] all $EXPECTED results saved; autosave exiting"
    exit 0
  fi
done
