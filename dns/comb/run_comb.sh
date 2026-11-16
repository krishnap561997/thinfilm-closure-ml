#!/bin/bash
#SBATCH --job-name=comb
#SBATCH -o comb_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH -t 1-00:00:00
#SBATCH --account=bala1s
#SBATCH --qos=bala1s-b
# One comb case: spin-up (filmflow0) then profile restart. Submit with
#   sbatch --chdir=$THINFILM_STORE/datasets/BasC/DNS/comb/<case> $THINFILM_ROOT/dns/comb/run_comb.sh
# SKIP_RESTART=1 runs the spin-up only (timing test).
set -euo pipefail
[[ -f COMB_INFO.txt ]] || { echo "not a comb case dir: $(pwd)" >&2; exit 1; }
source /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/code/thinfilm-closure-ml/env.sh
BUILD=$THINFILM_STORE/datasets/BasC/DNS/comb/build
module purge; module load gcc/14.2.0 imagemagick/7.0.8-20 hdf5
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_PROC_BIND=spread OMP_PLACES=cores KMP_STACKSIZE=1g
export LD_LIBRARY_PATH=$HPC_HDF5_LIB:$LD_LIBRARY_PATH
date; hostname; cat COMB_INFO.txt
RUN=$(ls -d run_t*)
cd spinup
cp $BUILD/filmflow . ; cp $BUILD/filmflow0.c $BUILD/Makefile .
/usr/bin/time -f "spinup wall %e s" ./filmflow params.in 2> log.err || { tail -5 log.err; exit 1; }
tail -1 log.err
[[ "${SKIP_RESTART:-0}" == 1 ]] && exit 0
TS=$(sed -n 's/^T_SPIN //p' ../COMB_INFO.txt)
DUMP=$(printf 'dump-%06d' "${TS%.*}")
[[ -f $DUMP ]] || { echo "missing $DUMP" >&2; ls; exit 1; }
cd ../$RUN
ln -sf ../spinup/$DUMP restart.dump
cp $BUILD/filmflow_restart_profiles . ; cp $BUILD/filmflow_restart_profiles.c .
/usr/bin/time -f "restart wall %e s" ./filmflow_restart_profiles params.in 2> log.err || { tail -5 log.err; exit 1; }
tail -1 log.err
echo "done $(date); $(ls profiles | wc -l) profiles"
