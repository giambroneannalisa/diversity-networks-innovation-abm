#!/usr/bin/env bash
# Run the NSGA-II experiment across 10 seeds with run_trajectory_v2.py.
#
# Seeds: the 3 thesis seeds (42, 123, 456) re-run under deterministic
# replicate seeding, plus 7 extension seeds continuing the same arbitrary
# "sequential round numbers" convention documented in Appendix D.3.
#
# Usage:
#   ./run_all_seeds.sh              # sequential (safest; ~45-55h total)
#   PARALLEL=3 ./run_all_seeds.sh   # run 3 seeds concurrently
#
# Each seed writes trajectory_seed<N>.csv, pareto_seed<N>.csv,
# replicates_seed<N>.csv and a log in logs/.

set -u
SEEDS=(42 123 456 789 1234 2345 3456 4567 5678 6789)
CONFIG="${CONFIG:-nsga2_config_v5_1_1.json}"
PARALLEL="${PARALLEL:-1}"

mkdir -p logs
running=0
for seed in "${SEEDS[@]}"; do
  if [ -f "pareto_seed${seed}.csv" ]; then
    echo "[skip] seed ${seed}: pareto_seed${seed}.csv already exists"
    continue
  fi
  echo "[start] seed ${seed} → logs/seed${seed}.log"
  python3 run_trajectory_v2.py "${CONFIG}" --seed "${seed}" > "logs/seed${seed}.log" 2>&1 &
  running=$((running + 1))
  if [ "${running}" -ge "${PARALLEL}" ]; then
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  fi
done
wait
echo "[done] all seeds finished. Run: python3 analyze_fronts.py ."
