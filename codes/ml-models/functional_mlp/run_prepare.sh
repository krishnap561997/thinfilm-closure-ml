#!/bin/bash
#SBATCH --job-name=thinfilm_prepare
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/log_prepare_%j.log
#SBATCH --partition=hpg-dev
#SBATCH --account=bala1s
#SBATCH --qos=bala1s

set -euo pipefail
export PATH="/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/software/envs/thinfilm-ml-py311/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
python -u prepare_data.py --config "${CONFIG:-data_config.json}"
