#!/bin/bash
#SBATCH --job-name=mlp_eval
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --output=eval_single_%j.log
#SBATCH --error=eval_single_%j.err

set -euo pipefail

export PATH="/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/software/envs/thinfilm-ml-py311/bin:$PATH"

cd "${SLURM_SUBMIT_DIR:?}"

: "${CACHE:?CACHE must be exported}"
: "${RUN_DIR:?RUN_DIR must be exported}"
SPLIT="${SPLIT:-val}"

python -u evaluate.py \
    --cache "$CACHE" \
    --checkpoint "$RUN_DIR/best.pt" \
    --split "$SPLIT" \
    --output "$RUN_DIR/eval_$SPLIT" \
    --save-predictions
