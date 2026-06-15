#!/usr/bin/env bash
# One-command run: experiment -> figures. Uses real data if
# data/transaction_dataset.csv exists, else a clearly-labeled synthetic set.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore
echo "== experiment =="; python3 src/run_experiment.py
echo "== figures ==";    python3 src/plots.py
echo "Done. See results/ (CSVs + figures/). Data source is in results/run_meta.json."
