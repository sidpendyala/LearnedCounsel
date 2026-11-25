#!/usr/bin/env bash
# v2: the single test on 90 fresh posts. Safe to stop and restart (finished runs are skipped, calls are cached).
cd "$(dirname "$0")/.." || exit 1
exec > >(tee -a logs/run_v2.log) 2>&1
run() {  # run <run_name> <arguments for src.experiments>
  local name=$1; shift
  for attempt in 1 2 3; do
    if [ -f "results/$name/metrics.json" ]; then echo "[$(date +%H:%M)] done: $name"; return; fi
    echo "[$(date +%H:%M)] start: $name (attempt $attempt)"
    .venv/Scripts/python -m src.experiments --run-name "$name" --set v2_test "$@"
  done
  [ -f "results/$name/metrics.json" ] || echo "[$(date +%H:%M)] FAILED: $name"
}
run learnedcounsel_v2_test --system learnedcounsel --variant v2_both
run b3_v2_test --system b3
run b4_v2_test --system b4
run learnedcounsel_v1_test --system learnedcounsel
echo "[$(date +%H:%M)] V2 TEST FINISHED"
