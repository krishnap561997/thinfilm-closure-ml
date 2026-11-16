#!/bin/bash
#SBATCH --job-name=cache_v2_full
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-dev
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH -t 01:30:00
# Run after all closure extractions of 16 Nov have finished (sbatch --dependency=afterok:...).
# 1. For each redone case, check ClosureData/thinfilm_closure_data_<case>.h5.new (readable, 200 snapshots,
#    t = 10..2000, targets present); if OK rename the old file to .stale_20261116 and the .new into place.
#    A case that fails the check is left untouched and reported.
# 2. Build cache v2 for every closure file (pipeline/cache_v2/build_cache_v2.py).
set -uo pipefail
ROOT=/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/code/thinfilm-closure-ml
source $ROOT/env.sh
export OMP_NUM_THREADS=1
CD=$THINFILM_STORE/datasets/BasC/DNS/param_space/ClosureData
for c in Re010_Ca0_010_theta45 Re010_Ca0_100_theta45 Re040_Ca0_100_theta45 Re010_Ca0_040_theta90; do
  NEW=$CD/thinfilm_closure_data_$c.h5.new; OLD=$CD/thinfilm_closure_data_$c.h5
  if python -I - "$NEW" <<'PY'
import sys, h5py, numpy as np
with h5py.File(sys.argv[1], "r") as f:
    t = f["t"][:]
    need = ["h", "q", "int_u2_dy", "delta_Mx_ML", "delta_Tw_ML", "delta_Px_ML_v2"]
    miss = [k for k in need if k not in f]
    ok = len(t) == 200 and abs(t[0] - 10) < 1e-6 and abs(t[-1] - 2000) < 1e-6 and not miss and f["h"].shape[1] == 200
    print(sys.argv[1], "nt", len(t), "t", t[0], t[-1], "missing", miss, "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
  then
    mv "$OLD" "$OLD.stale_20261116" && mv "$NEW" "$OLD" && echo "swapped $c"
  else
    echo "NOT swapped $c (check failed)"
  fi
done
python -I $ROOT/pipeline/cache_v2/build_cache_v2.py --ncpu 8
