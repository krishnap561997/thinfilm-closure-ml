#!/bin/bash

set -euo pipefail

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT="/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml"

CASE_DIR="${PROJECT_ROOT}/codes/ml-models/case01"
DATA_ROOT="${PROJECT_ROOT}/datasets/ROM_Ca_Re_sweep"
OUTPUT_DIR="${DATA_ROOT}/dataset_index"

PYTHON_SCRIPT="${CASE_DIR}/build_thinfilm_dataset.py"

# Change this if your environment has a different name/path.
CONDA_ENV="${PROJECT_ROOT}/software/envs/thinfilm-ml-py311"

# ============================================================
# Environment
# ============================================================

module load conda

conda activate "${CONDA_ENV}"

echo "============================================================"
echo "Thin-film dataset indexing"
echo "============================================================"
echo "Host:              $(hostname)"
echo "Working directory: ${CASE_DIR}"
echo "Python script:     ${PYTHON_SCRIPT}"
echo "Data root:         ${DATA_ROOT}"
echo "Output directory:  ${OUTPUT_DIR}"
echo "Python executable: $(which python)"
echo "============================================================"

python --version

python -c "
import numpy
import h5py

print('NumPy version:', numpy.__version__)
print('h5py version: ', h5py.__version__)
"

# ============================================================
# Input checks
# ============================================================

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Python script not found:"
    echo "  ${PYTHON_SCRIPT}"
    exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "ERROR: Dataset directory not found:"
    echo "  ${DATA_ROOT}"
    exit 1
fi

# ============================================================
# Build metadata and sampled indices
# ============================================================

cd "${CASE_DIR}"

python "${PYTHON_SCRIPT}" \
    --root "${DATA_ROOT}" \
    --output "${OUTPUT_DIR}" \
    --hx-key /h_x \
    --time-key /time \
    --x-key /x \
    --x-stride 4 \
    --time-stride 5 \
    --high-hx-percentile 95 \
    --high-hx-extra-per-time 200 \
    --train-count 20 \
    --val-count 5 \
    --test-count 5 \
    --seed 20260721

echo "============================================================"
echo "Dataset indexing completed successfully."
echo "Output written to:"
echo "  ${OUTPUT_DIR}"
echo "============================================================"
