#!/usr/bin/env bash
#SBATCH --job-name=extract
#SBATCH --output=/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/datasets/BasC/param_space/ClosureData/logs/%x_%j.out
#SBATCH --error=/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/datasets/BasC/param_space/ClosureData/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-dev

set -euo pipefail

# Mirror the original job environment while allowing submission from anywhere.
cd "${SLURM_SUBMIT_DIR}"
export PATH=/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/software/envs/thinfilm-ml-py311/bin:"$PATH"
PYTHON_SCRIPT=/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/data-preprocessing/extract-closure-terms/extract_closure_terms.py
CONFIG_FILE="${1:-}"

echo "============================================================"
echo "Thin-film closure extraction"
echo "============================================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Node          : $(hostname)"
echo "Working dir   : $(pwd)"
echo "Python        : $(command -v python3)"
echo "Config        : ${CONFIG_FILE}"
echo "Start time    : $(date)"
echo "============================================================"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "ERROR: Cannot find $PYTHON_SCRIPT" >&2
    exit 1
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Cannot find $CONFIG_FILE" >&2
    exit 1
fi

echo "Config file: $(realpath "$CONFIG_FILE")"
echo "Config contents:"
cat "$CONFIG_FILE"

python3 -u "$PYTHON_SCRIPT" "$CONFIG_FILE"
echo "Finished successfully"
echo "End time: $(date)"
