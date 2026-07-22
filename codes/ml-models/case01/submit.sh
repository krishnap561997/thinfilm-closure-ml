#!/bin/bash
#SBATCH --job-name=thinfilm_index
#SBATCH --output=thinfilm_index_%j.out
#SBATCH --error=thinfilm_index_%j.err
#SBATCH --partition=hpg-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_ROOT="/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/ml-models/case01"

cd "${PROJECT_ROOT}"

./build_dataset_index.sh
