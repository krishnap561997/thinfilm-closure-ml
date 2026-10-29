#!/bin/bash
#SBATCH --job-name=thinfilm_sweep
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --array=0-29%4
#SBATCH --output="/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/ml-models/thinfilm_mlp/runs/sweep_01/logs/%A_%a.log"
set -euo pipefail
export PATH="/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/software/envs/thinfilm-ml-py311/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -u /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/ml-models/thinfilm_mlp/run_sweep_task.py --sweep /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/ml-models/thinfilm_mlp/runs/sweep_01/sweep.json --task-id "$SLURM_ARRAY_TASK_ID"
