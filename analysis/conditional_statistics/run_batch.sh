#!/bin/bash
#SBATCH --job-name=closure_statistics
#SBATCH --output=logfile_%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-dev
set -euo pipefail
# Edit resource settings to match your allocation; load your working Python modules if needed.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi

cd -- "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/software/envs/thinfilm-ml-py311/bin/python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/analyze_closures.py" \
  --config "$SCRIPT_DIR/config.json" --mode scan "$@"
