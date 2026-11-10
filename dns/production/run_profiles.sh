#!/bin/bash
#SBATCH --job-name=prof
#SBATCH -o run_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH -t 06:00:00
#SBATCH --account=bala1s
#SBATCH --qos=bala1s

# Run filmflow_restart_profiles in a directory made by setup_profile_run.sh, then a
# post-processing step:
#   POST=check  SOURCE=<case dir>  -> pipeline/extract_closures/check_profile_restart.py
#   POST=budget                    -> pipeline/extract_closures/profile_budget.py (budget.csv)
#   sbatch --chdir=RUN_DIR --export=ALL,POST=budget run_profiles.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
date; hostname
ROOT="${THINFILM_ROOT:-/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/code/thinfilm-closure-ml}"
source "${ROOT}/env.sh"
module purge
module load gcc/14.2.0 imagemagick/7.0.8-20 hdf5
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_PROC_BIND=spread OMP_PLACES=cores KMP_STACKSIZE=1g
export LD_LIBRARY_PATH=$HPC_HDF5_LIB:$LD_LIBRARY_PATH
cat RUN_INFO.txt
./filmflow_restart_profiles params.in 2> log.err
echo "solver finished: $(date); $(ls profiles | wc -l) profiles"
PY="env -u PYTHONHOME ${THINFILM_ENV}/bin/python -u"
case "${POST:-}" in
  check)  ${PY} "${ROOT}/pipeline/extract_closures/check_profile_restart.py" . "${SOURCE}" ;;
  budget) ${PY} "${ROOT}/pipeline/extract_closures/profile_budget.py" profiles --csv budget.csv ;;
  *) echo "no post-processing" ;;
esac
date
