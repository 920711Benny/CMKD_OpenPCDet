#!/usr/bin/env bash
# End-to-end lifecycle driver for the Sub-1B Dual-Head Diffusion VLA.
#
#   ./run_pipeline.sh budget            parameter-budget gate (<1B)
#   ./run_pipeline.sh test              unit tests
#   ./run_pipeline.sh prepare <dataset> CARLA logs -> training manifests
#   ./run_pipeline.sh train             single-GPU training
#   ./run_pipeline.sh verify <ckpt>     atomic gates (terminal gate)
#   ./run_pipeline.sh align <ckpt>      Action-CoT alignment score
#   ./run_pipeline.sh launch <ckpt>     print the CARLA leaderboard command
#   ./run_pipeline.sh report            benchmark table (terminal output only)
set -euo pipefail

CONFIG="${SUB1B_CONFIG:-sub1b_vla/configs/default.yaml}"
RUN_DIR="${SUB1B_RUNS:-runs}"
export SUB1B_SCRATCH="${SUB1B_SCRATCH:-$PWD/scratch}"
export HF_HOME="$SUB1B_SCRATCH/hf"
export TORCH_HOME="$SUB1B_SCRATCH/torch"
mkdir -p "$SUB1B_SCRATCH" "$RUN_DIR"

cmd="${1:-help}"; shift || true

case "$cmd" in
  budget)  python3 -m sub1b_vla.tools.param_budget --config "$CONFIG" ;;
  test)    python3 -m pytest sub1b_vla/tests -q ;;
  prepare) python3 -m sub1b_vla.data.prepare_carla_data --root "$1" --out "$1" ;;
  train)   python3 -m sub1b_vla.train.train --config "$CONFIG" "$@" ;;
  verify)
    python3 -m sub1b_vla.verify.atomic_checks --config "$CONFIG" \
      --checkpoint "$1" --json-out "$RUN_DIR/atomic_gates.json" ;;
  align)
    python3 -m sub1b_vla.bench.alignment_eval --config "$CONFIG" \
      --checkpoint "$1" --out "$RUN_DIR/alignment.json" --intent-source parsed ;;
  launch)
    for suite in town05_long town05_hard; do
      for weather in ClearNoon HardRainNoon WetCloudySunset MidRainyNight; do
        python3 -m sub1b_vla.bench.run_benchmark --config "$CONFIG" \
          --checkpoint "$1" --suite "$suite" --weather "$weather" --print-launch
        echo
      done
    done ;;
  report)
    python3 -m sub1b_vla.bench.run_benchmark --config "$CONFIG" \
      --results "$RUN_DIR"/results_*.json \
      --baseline baselines/simlingo.json \
      --runtime "$RUN_DIR/latency_report.json" \
      --alignment "$RUN_DIR/alignment.json" \
      --json-out "$RUN_DIR/benchmark_summary.json" ;;
  *) sed -n '2,12p' "$0" ;;
esac
