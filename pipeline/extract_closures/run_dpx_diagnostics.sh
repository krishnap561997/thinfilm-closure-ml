#!/bin/bash
#SBATCH --job-name=dpx_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-dev

# ΔP_x extraction diagnostics (plan Step 1) on 2D snapshots: one array task per run.
#   sbatch --output=<out_dir>/logs/dpx_%A_%a.out run_dpx_diagnostics.sh
# Results: <out_dir>/<label>_metrics.csv, <label>_fields.h5; summary afterwards with
#   $THINFILM_ENV/bin/python dpx_diagnostics.py summary <out_dir>

set -euo pipefail
ROOT="${THINFILM_ROOT:-/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/code/thinfilm-closure-ml}"
source "${ROOT}/env.sh"
PS="${THINFILM_DATA}/BasC/DNS/param_space"
LC="${THINFILM_STORE}/codes/basilisk/physics-tests/bc-blowup-fix/length-check"
OUT_DIR="${OUT_DIR:-${PS}/ClosureData/dpx_diagnostics}"
SCRIPT="${ROOT}/pipeline/extract_closures/dpx_diagnostics.py"
mkdir -p "${OUT_DIR}/logs"

# label | snapshot dir | times | closure file (or -)
RUNS=(
  "Re010_Ca0_010_theta45|${PS}/Re010_Ca0_010_theta45/output|100:1900:200|${PS}/ClosureData/thinfilm_closure_data_Re010_Ca0_010_theta45.h5"
  "Re080_Ca0_020_theta45|${PS}/Re080_Ca0_020_theta45/output|100:1900:200|${PS}/ClosureData/thinfilm_closure_data_Re080_Ca0_020_theta45.h5"
  "Re010_Ca0_100_theta45|${PS}/Re010_Ca0_100_theta45/output|100:1500:200|${PS}/ClosureData/thinfilm_closure_data_Re010_Ca0_100_theta45.h5"
  "Re040_Ca0_002_theta45|${PS}/Re040_Ca0_002_theta45/output|100:1100:100|-"
  "Re010_Ca0_005_theta45|${PS}/Re010_Ca0_005_theta45/output|100:700:100|-"
  "Re120_Ca0_010_theta45|${PS}/Re120_Ca0_010_theta45/output|100:1900:200|${PS}/ClosureData/thinfilm_closure_data_Re120_Ca0_010_theta45.h5"
  "mono_L14|${LC}/L800/output|550:650:10|-"
  "mono_L15|${LC}/L800_Level15/output|550:650:10|-"
)

IFS='|' read -r LABEL SNAP TIMES CLOS <<< "${RUNS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "job ${SLURM_JOB_ID:-interactive} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname): ${LABEL}"
echo "git commit: $(git -C "${ROOT}" rev-parse HEAD)$(git -C "${ROOT}" diff --quiet -- "${SCRIPT}" || echo ' (script modified)')"
EXTRA=()
[[ "${CLOS}" != "-" ]] && EXTRA=(--closure "${CLOS}")
env -u PYTHONHOME "${THINFILM_ENV}/bin/python" -u "${SCRIPT}" run --snapdir "${SNAP}" --times "${TIMES}" \
    --label "${LABEL}" --out "${OUT_DIR}" "${EXTRA[@]}"
echo "Finished: $(date)"
