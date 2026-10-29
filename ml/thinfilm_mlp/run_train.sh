#!/bin/bash
#SBATCH --job-name=thinfilm_mlp
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=train_%j.log
#SBATCH --account=bala1s
#SBATCH --qos=bala1s

set -euo pipefail
export PATH="/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/software/envs/thinfilm-ml-py311/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
CACHE="${CACHE:-cache/default}"
RUN_DIR="${RUN_DIR:-runs/corrections_j4_s1}"
HALF_WINDOW="${HALF_WINDOW:-4}"
STRIDE="${STRIDE:-1}"
python -u train.py --cache "$CACHE" --run-dir "$RUN_DIR" \
    --half-window "$HALF_WINDOW" --stride "$STRIDE" "$@"
