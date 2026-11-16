#!/bin/bash
#SBATCH --job-name=os_case_table
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-dev
#SBATCH --cpus-per-task=8
#SBATCH --mem=8gb
#SBATCH --time=00:45:00
#SBATCH --output=%x_%j.out
# B.4: OS case table over datasets/MANIFEST_dns.csv. Submit from $THINFILM_STORE/analysis/orr_sommerfeld/logs:
#   cd $THINFILM_STORE/analysis/orr_sommerfeld/logs && sbatch $THINFILM_ROOT/analysis/orr_sommerfeld/run_case_table.sh
source "$THINFILM_ROOT/env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -I "$THINFILM_ROOT/analysis/orr_sommerfeld/case_table.py" --ncpu "$SLURM_CPUS_PER_TASK"
