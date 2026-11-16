#!/bin/bash
#SBATCH --job-name=b6_rom
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-dev
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=01:00:00
#SBATCH --output=%x_%A_%a.out
# B.6 linear ROM runs. Submit from $THINFILM_STORE/analysis/orr_sommerfeld/b6_rom:
#   sbatch --array=0-8 $THINFILM_ROOT/analysis/orr_sommerfeld/b6_rom/run_array.sh
module load gcc/14.2.0 hdf5
BIN=/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/code/thinfilm-closure-ml/rom/lavalle-cpp/thinfilm
D=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" runs.txt)
cd "$D" || exit 1
if [ -f ml.txt ]; then "$BIN" input.txt none ml.txt > run.log 2>&1; else "$BIN" input.txt > run.log 2>&1; fi
echo "exit $?" >> run.log
