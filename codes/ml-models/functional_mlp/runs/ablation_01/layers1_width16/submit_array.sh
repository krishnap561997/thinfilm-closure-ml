#!/bin/bash
#SBATCH --job-name=func_L1W16
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --array=0-15%2
#SBATCH --output="/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/ml-models/functional_mlp/runs/ablation_01/layers1_width16/logs/%A_%a.log"
set -euo pipefail
export PATH="/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/software/envs/thinfilm-ml-py311/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -u /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/ml-models/functional_mlp/run_sweep_task.py --sweep /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/ml-models/functional_mlp/runs/ablation_01/layers1_width16/sweep.json --task-id "$SLURM_ARRAY_TASK_ID"
