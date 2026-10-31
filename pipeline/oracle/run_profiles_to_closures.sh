#!/bin/bash
#SBATCH --job-name=prof2clos
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-dev

# Convert an oracle run's 1D profiles into closure files, one chunk of CHUNK time units per array task.
#   sbatch --output=<run>/../closures/logs/p2c_%A_%a.out run_profiles_to_closures.sh <run_dir> [t0] [chunk]
# Task k writes <run_dir>/../closures/closures_part_<k>.h5 for t in (t0 + k*chunk, t0 + (k+1)*chunk].

set -euo pipefail
ROOT="${THINFILM_ROOT:-/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/code/thinfilm-closure-ml}"
source "${ROOT}/env.sh"
RUN="${1:?run directory}"
T0="${2:-1800}"
CHUNK="${3:-10}"
k="${SLURM_ARRAY_TASK_ID:-0}"
OUT="$(dirname "${RUN}")/closures"
mkdir -p "${OUT}"

ts=$(python3 -c "print(${T0} + ${k}*${CHUNK} + 0.005)")
te=$(python3 -c "print(${T0} + (${k}+1)*${CHUNK} + 0.005)")
echo "task ${k}: t in [${ts}, ${te}) -> ${OUT}/closures_part_${k}.h5"
cd "${ROOT}/pipeline/extract_closures"
"${THINFILM_ENV}/bin/python" -u profiles_to_closures.py "${RUN}/profiles" "${OUT}/closures_part_${k}.h5" \
    --t-start "${ts}" --t-end "${te}"
echo "done $(date)"
