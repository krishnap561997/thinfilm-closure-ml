#!/bin/bash
# ROM runs written by rom_tests.py make (one directory per line of rom/runs.txt):
#   sbatch --array=0-<n-1> ml/structured_closure/run_rom_tests.sh
#SBATCH --job-name=sc_rom
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-dev
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=04:00:00   # MLP closures: ~1 h (oracle) to ~2.5 h (comb Re200) on one core
#SBATCH --output=/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/ml-models/structured_closure/logs/rom_%A_%a.out
module load gcc/14.2.0 hdf5
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
BIN=/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/code/thinfilm-closure-ml/rom/lavalle-cpp/thinfilm
D=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/ml-models/structured_closure/rom/runs.txt)
cd "$D" || exit 1
ORA=none; [ -f oracle.txt ] && ORA=oracle.txt
"$BIN" input.txt $ORA sc.txt > run.log 2>&1
echo "exit $?" >> run.log
tail -n 3 run.log
