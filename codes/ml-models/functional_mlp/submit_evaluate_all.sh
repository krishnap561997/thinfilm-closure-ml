#!/bin/bash
set -euo pipefail

export CACHE="/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/ml-models/thinfilm_mlp/cache_dir"
export SPLIT=val

for checkpoint in runs/sweep_curvature_01/*/best.pt; do
    [ -f "$checkpoint" ] || continue
    export RUN_DIR="${checkpoint%/best.pt}"
    sbatch run_evaluate.sh
done
