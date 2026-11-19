#!/bin/bash
# After a training array finishes: export the named runs and submit their ROM tests (uniform, comb, oracle)
# and broadband held-out runs. Usage:
#   RUNS="mlp_s0.25_L10_seed1 mlp_s0.5_L10_seed1" TAG=seed1 sbatch --export=ALL --dependency=afterok:<train job> ml/structured_closure/chain_seed_rom.sh
#SBATCH --job-name=sc_chain
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-dev
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=00:30:00
#SBATCH --output=/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/ml-models/structured_closure/logs/chain_%j.out
source /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/code/thinfilm-closure-ml/env.sh
export LD_LIBRARY_PATH=$THINFILM_ENV/lib
SC=$THINFILM_ROOT/ml/structured_closure; OUT=$THINFILM_STORE/codes/ml-models/structured_closure/rom
cd $THINFILM_ROOT
$THINFILM_ENV/bin/python $SC/rom_tests.py make $RUNS || exit 1
n=$(wc -l < $OUT/runs.txt); sbatch --array=0-$((n-1)) $SC/run_rom_tests.sh
$THINFILM_ENV/bin/python $SC/broadband.py make --closure $RUNS --list runs_${TAG}.txt || exit 1
n=$(wc -l < $OUT/broadband/runs_${TAG}.txt); LIST=runs_${TAG}.txt sbatch --export=ALL --array=0-$((n-1)) $SC/run_broadband.sh
