#!/bin/bash
#SBATCH --job-name=fit_form_p0
#SBATCH --partition=hpg-dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --output=fit_form_%j.log

# Phase 0 Tier-1 fits (plan task 0.3) and talk figures (task 0.4)
set -euo pipefail
S=/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml
PY=$S/software/envs/thinfilm-ml-py311/bin/python
OUT=$S/codes/ml-models/functional_forms/phase0_tier1
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
cd "${SLURM_SUBMIT_DIR:?}"
$PY fit_form.py --cache $S/codes/ml-models/functional_mlp/cache_dir --out $OUT "$@"
$PY -I ../../scripts/figures/functional-form-v2/phase0_figures.py --phase0 $OUT/phase0.json \
    --sparse $S/codes/ml-models/functional_forms/sparse_01/sparse_path.json --out $OUT/figures
