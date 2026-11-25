#!/usr/bin/env bash
# Runs every M6 experiment in order of importance, one after the other.
# Safe to stop and restart: finished runs are skipped and LLM calls are cached.
#   bash scripts/run_all.sh
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
exec > >(tee -a logs/run_all.log) 2>&1

run() {  # run <run_name> <arguments for src.experiments>
  local name=$1; shift
  for attempt in 1 2 3; do
    if [ -f "results/$name/metrics.json" ]; then echo "[$(date +%H:%M)] done: $name"; return; fi
    echo "[$(date +%H:%M)] start: $name (attempt $attempt)"
    .venv/Scripts/python -m src.experiments --run-name "$name" "$@"
  done
  [ -f "results/$name/metrics.json" ] || echo "[$(date +%H:%M)] FAILED: $name"
}

# Main comparison
run learnedcounsel_main --system learnedcounsel --set main
run b3_main --system b3 --set main
run b2_A --system b2 --set A
run b4_main --system b4 --set main
run b5_B --system b5 --set B

# Ablations (a)-(e) and tau sweep (f); they reuse the main run's cached calls
run learnedcounsel_ablation --system learnedcounsel --set ablation
for v in no_skeptic no_router no_consults single_specialist rounds0 rounds1 tau0.5 tau0.7; do
  run "learnedcounsel_${v}_ablation" --system learnedcounsel --variant $v --set ablation
done

# True per-post latency: one request at a time, no cache (dev posts)
for s in b3 b4 b5 learnedcounsel; do
  run "latency_$s" --system $s --set dev --n 5 --concurrency 1 --no-cache
done

# (g) 4B specialists
run learnedcounsel_mixed_4b_ablation --system learnedcounsel --variant mixed_4b --set ablation
# (h) few-shot examples for the specialists
run learnedcounsel_few_shot_ablation --system learnedcounsel --variant few_shot --set ablation
echo "[$(date +%H:%M)] ALL RUNS FINISHED"
