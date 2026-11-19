#!/bin/bash
# Broadband held-out ROM runs written by broadband.py make (one directory per line of rom/broadband/runs.txt):
#   sbatch --array=0-<n-1> ml/structured_closure/run_broadband.sh
#   LIST=runs_x.txt sbatch --export=ALL --array=... ml/structured_closure/run_broadband.sh   # another list
# 2000 time units: Lavalle ~20 min, ml_sum ~1 h, flat MLP a few hours with 4 OpenMP threads.
#SBATCH --job-name=sc_bb
#SBATCH --account=bala1s
#SBATCH --qos=bala1s-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=4gb
#SBATCH --time=12:00:00
#SBATCH --output=/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/ml-models/structured_closure/logs/bb_%A_%a.out
module load gcc/14.2.0 hdf5
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
BIN=/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/code/thinfilm-closure-ml/rom/lavalle-cpp/thinfilm
D=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/ml-models/structured_closure/rom/broadband/${LIST:-runs.txt})
cd "$D" || exit 1
if [ -f closure.txt ]; then "$BIN" input.txt none closure.txt > run.log 2>&1; else "$BIN" input.txt > run.log 2>&1; fi
echo "exit $?" >> run.log
tail -n 3 run.log
