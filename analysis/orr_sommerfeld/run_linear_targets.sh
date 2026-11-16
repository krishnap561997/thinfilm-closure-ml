#!/bin/bash
#SBATCH --job-name=os_linear_targets
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-dev
#SBATCH --cpus-per-task=8
#SBATCH --mem=8gb
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
# B.7/B.8 over all cases. Submit from $THINFILM_STORE/analysis/orr_sommerfeld/logs with --export=ALL,THINFILM_ROOT=...
source "$THINFILM_ROOT/env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -I "$THINFILM_ROOT/analysis/orr_sommerfeld/linear_targets.py" --ncpu "$SLURM_CPUS_PER_TASK"
