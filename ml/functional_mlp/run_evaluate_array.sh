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
#SBATCH --output=eval_array_%A_%a.log
#SBATCH --error=eval_array_%A_%a.err

# Array version of run_evaluate_single.sh. TASKS is a file with one "RUN_DIR SPLIT" per line;
# array task i evaluates line i+1. At most 2 GPU jobs at a time, so always submit with %2:
#   n=$(wc -l < tasks.txt)
#   TASKS=tasks.txt CACHE=... sbatch --export=ALL --array=0-$((n-1))%2 run_evaluate_array.sh

set -euo pipefail

export PATH="/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/software/envs/thinfilm-ml-py311/bin:$PATH"

cd "${SLURM_SUBMIT_DIR:?}"

: "${CACHE:?CACHE must be exported}"
: "${TASKS:?TASKS must be exported}"
read -r RUN_DIR SPLIT < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")
: "${RUN_DIR:?no line $((SLURM_ARRAY_TASK_ID + 1)) in $TASKS}"
echo "task ${SLURM_ARRAY_TASK_ID}: ${RUN_DIR} ${SPLIT}"

python -u evaluate.py \
    --cache "$CACHE" \
    --checkpoint "$RUN_DIR/best.pt" \
    --split "$SPLIT" \
    --output "$RUN_DIR/eval_$SPLIT" \
    --save-predictions
