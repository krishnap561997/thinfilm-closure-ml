#!/bin/bash
#SBATCH --job-name=budget_check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-dev

# Momentum-budget check over every extracted closure file.
#   sbatch --output=<out_dir>/logs/budget_%j.out run_budget_check.sh [times] [smooth]
# Defaults: times 1000:2000:100, smoothing widths 0,1 (h_N).
# Results: <out_dir>/per_case/<case>.csv, <out_dir>/summary.csv, <out_dir>/run_info.txt

set -euo pipefail

ROOT="${THINFILM_ROOT:-/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/code/thinfilm-closure-ml}"
source "${ROOT}/env.sh"
CLOSURE_DIR="${THINFILM_DATA}/BasC/DNS/param_space/ClosureData"
OUT_DIR="${OUT_DIR:-${CLOSURE_DIR}/budget_check}"
TIMES="${1:-1000:2000:100}"
SMOOTH="${2:-0,1}"
SCRIPT="${ROOT}/pipeline/extract_closures/check_momentum_budget.py"

mkdir -p "${OUT_DIR}/per_case" "${OUT_DIR}/logs"
rm -f "${OUT_DIR}/summary.csv"

{
    echo "date        : $(date)"
    echo "job         : ${SLURM_JOB_ID:-interactive} on $(hostname)"
    echo "script      : ${SCRIPT}"
    echo "git commit  : $(git -C "${ROOT}" rev-parse HEAD)$(git -C "${ROOT}" diff --quiet -- "${SCRIPT}" || echo ' (script modified)')"
    echo "closure dir : ${CLOSURE_DIR}"
    echo "times       : ${TIMES}"
    echo "smoothing   : ${SMOOTH} h_N (Gaussian width on ROM grid)"
    echo "mask        : 20 < x < 0.95 L (L = 800), ROM N = 4000"
} > "${OUT_DIR}/run_info.txt"
cat "${OUT_DIR}/run_info.txt"

for f in "${CLOSURE_DIR}"/thinfilm_closure_data_*.h5; do
    case=$(basename "${f}" .h5); case=${case#thinfilm_closure_data_}
    echo "=== ${case}"
    "${THINFILM_ENV}/bin/python" -u "${SCRIPT}" "${f}" --times "${TIMES}" --smooth "${SMOOTH}" \
        --csv "${OUT_DIR}/per_case/${case}.csv" --summary-csv "${OUT_DIR}/summary.csv" \
        || echo "FAILED: ${case}"
done

echo "Finished: $(date)"
