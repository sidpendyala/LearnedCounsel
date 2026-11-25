#!/usr/bin/env bash
# v2: choose between the candidates on the 24 new dev posts.
cd "$(dirname "$0")/.." || exit 1
exec > >(tee -a logs/run_v2.log) 2>&1
for v in v2_add v2_both v2_no_scout; do
  [ -f "results/learnedcounsel_${v}_v2_dev/metrics.json" ] && continue
  echo "[$(date +%H:%M)] start: $v"
  .venv/Scripts/python -m src.experiments --system learnedcounsel --variant $v --set v2_dev
done
.venv/Scripts/python -m src.experiments --system b3 --set v2_dev
echo "[$(date +%H:%M)] V2 DEV FINISHED"
