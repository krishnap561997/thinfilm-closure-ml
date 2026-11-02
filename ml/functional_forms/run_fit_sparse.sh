#!/bin/bash
#SBATCH --job-name=fit_sparse
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --output=fit_sparse_%j.log

set -euo pipefail
S=/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml
export OMP_NUM_THREADS=8
cd "${SLURM_SUBMIT_DIR:?}"
$S/software/envs/thinfilm-ml-py311/bin/python -I fit_sparse.py \
    --cache $S/codes/ml-models/functional_mlp/cache_dir \
    --out $S/codes/ml-models/functional_forms/sparse_01 "$@"
