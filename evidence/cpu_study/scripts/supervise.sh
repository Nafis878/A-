#!/bin/bash
# Keep a cell-cached experiment script running until it completes.
# Completed cells are skipped by their result.json; an interrupted cell resumes
# from its checkpoint. So a restart never redoes finished work.
script="$1"; log="$2"; max="${3:-40}"
for i in $(seq 1 "$max"); do
  echo "=== attempt $i  $(date +%H:%M:%S) ===" >> "$log"
  python "$script" >> "$log" 2>&1 && { echo "=== COMPLETE ===" >> "$log"; exit 0; }
  echo "=== exited $?; restarting in 10s ===" >> "$log"
  sleep 10
done
echo "=== GAVE UP after $max attempts ===" >> "$log"
