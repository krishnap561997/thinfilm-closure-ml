#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --output=logfile_%j.out
#SBATCH --error=error_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00

#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-dev

set -euo pipefail


# ============================================================
# Move to the directory from which this job was submitted
# ============================================================

cd "${SLURM_SUBMIT_DIR}"

echo "============================================================"
echo "Thin-film closure extraction"
echo "============================================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Node          : $(hostname)"
echo "Working dir   : $(pwd)"
echo "Start time    : $(date)"
echo "============================================================"

export PATH=/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/software/envs/thinfilm-ml-py311/bin:$PATH

PYTHON_SCRIPT="extract_closure_data_v2.py"
CONFIG_FILE="${1:-params.json}"

echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "Working dir : $(pwd)"
echo "Python      : $(which python3)"
echo "Config      : ${CONFIG_FILE}"
echo "Start time  : $(date)"
echo "Config file: $(realpath "${CONFIG_FILE}")"
echo "Config contents:"
cat "${CONFIG_FILE}"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Cannot find ${PYTHON_SCRIPT}"
    exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: Cannot find ${CONFIG_FILE}"
    exit 1
fi

python3 -u "${PYTHON_SCRIPT}" "${CONFIG_FILE}"

echo "Finished successfully"
echo "End time: $(date)"
