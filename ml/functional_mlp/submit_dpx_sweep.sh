#!/bin/bash
#SBATCH --job-name=dpx_sweep_submit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=hpg-dev
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
# Generate the ΔP_x v2 sweep (after the cache exists) and submit its array.
#   sbatch --dependency=afterok:<prepare job> submit_dpx_sweep.sh
set -euo pipefail
export PATH="/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/software/envs/thinfilm-ml-py311/bin:$PATH"
unset PYTHONHOME
cd "${SLURM_SUBMIT_DIR:-$PWD}"
OUT="${SWEEP_DIR:?set SWEEP_DIR}"
python -u generate_sweep.py --config sweep_config_dpx.json --output "$OUT"
for f in "$OUT"/*/submit_array.sh; do echo "submitting $f"; sbatch "$f"; done
