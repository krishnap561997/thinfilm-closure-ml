#!/bin/bash
#SBATCH --job-name=mlp_eval
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=eval_%A_%a.log
#SBATCH --account=bala1s
#SBATCH --qos=bala1s

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export PATH="/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/software/envs/thinfilm-ml-py311/bin:$PATH"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

CACHE="/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/ml-models/thinfilm_mlp/cache_dir"
SPLIT="${SPLIT:-val}"

checkpoint=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" evaluation_checkpoints.txt)
test -f "$checkpoint"

run_dir=$(dirname "$checkpoint")

python -u evaluate.py \
    --cache "$CACHE" \
    --checkpoint "$checkpoint" \
    --split "$SPLIT" \
    --output "$run_dir/eval_$SPLIT" \
    --save-predictions
