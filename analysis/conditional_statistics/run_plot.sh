#!/bin/bash
#SBATCH --job-name=closure_plots
#SBATCH --output=logfile_%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-dev

# Submit from the conditional_statistics directory: sbatch run_plot.sh
# Examples:
#   sbatch run_plot.sh
#   PLOT_FORMAT=pdf sbatch run_plot.sh
#   PLOT_KIND=both PLOT_FORMAT=both sbatch run_plot.sh
#   sbatch run_plot.sh --qois delta_Tw_ML --regions All
# Environment overrides: PYTHON_BIN, PLOT_WORK_DIR, CONFIG_FILE, RESULTS_DIR,
# FIGURES_DIR, PLOT_KIND, PLOT_FORMAT, PDF_TYPE.
# Load the same modules as your working Python environment if required.
set -euo pipefail

# Slurm copies batch scripts into its spool: use the submission directory there.
if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    DEFAULT_WORK_DIR="$SLURM_SUBMIT_DIR"
else
    DEFAULT_WORK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
WORK_DIR="${PLOT_WORK_DIR:-$DEFAULT_WORK_DIR}"
cd -- "$WORK_DIR"
WORK_DIR="$PWD"

PYTHON_BIN="${PYTHON_BIN:-/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/software/envs/thinfilm-ml-py311/bin/python}"
CONFIG_FILE="${CONFIG_FILE:-$WORK_DIR/config.json}"
FIGURES_DIR="${FIGURES_DIR:-$WORK_DIR/figures}"
PLOT_KIND="${PLOT_KIND:-correlations}"
PLOT_FORMAT="${PLOT_FORMAT:-png}"
PDF_TYPE="${PDF_TYPE:-standardized}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg

case "$PLOT_KIND" in correlations|pdfs|both) ;; *) printf 'Invalid PLOT_KIND: %s\n' "$PLOT_KIND" >&2; exit 1 ;; esac
case "$PLOT_FORMAT" in png|pdf|both) ;; *) printf 'Invalid PLOT_FORMAT: %s\n' "$PLOT_FORMAT" >&2; exit 1 ;; esac
case "$PDF_TYPE" in raw|standardized) ;; *) printf 'Invalid PDF_TYPE: %s\n' "$PDF_TYPE" >&2; exit 1 ;; esac
[[ -f "$WORK_DIR/plot_statistics.py" ]] || { printf 'Missing plot_statistics.py in %s\n' "$WORK_DIR" >&2; exit 1; }

# Default to the analysis output_dir specified in config.json.
if [[ -z "${RESULTS_DIR:-}" ]]; then
    RESULTS_DIR="$("$PYTHON_BIN" - "$CONFIG_FILE" <<'PY'
import json
from pathlib import Path
import sys
config = Path(sys.argv[1]).expanduser().resolve()
settings = json.loads(config.read_text())
path = Path(settings['output_dir']).expanduser()
print((path if path.is_absolute() else config.parent / path).resolve())
PY
)"
fi
[[ -d "$RESULTS_DIR" ]] || { printf 'Results directory does not exist: %s\n' "$RESULTS_DIR" >&2; exit 1; }
RESULTS_DIR="$(cd -- "$RESULTS_DIR" && pwd)"
mkdir -p -- "$FIGURES_DIR"
FIGURES_DIR="$(cd -- "$FIGURES_DIR" && pwd)"

printf 'Results: %s\nFigures: %s\nKind: %s; format: %s; PDF type: %s\n' \
    "$RESULTS_DIR" "$FIGURES_DIR" "$PLOT_KIND" "$PLOT_FORMAT" "$PDF_TYPE"

processed=0
failed=0
while IFS= read -r -d '' file; do
    # Preserve the configuration subdirectories (and any deeper structure).
    relative="${file#"$RESULTS_DIR"/}"
    relative_dir="$(dirname -- "$relative")"
    destination="$FIGURES_DIR/$relative_dir"
    # A PDF-only invocation otherwise shares correlation filenames.
    if [[ "$PLOT_KIND" == pdfs ]]; then
        destination="$destination/pdfs_$PDF_TYPE"
    fi
    printf '\nPlotting: %s\n' "$file"
    if "$PYTHON_BIN" -u "$WORK_DIR/plot_statistics.py" "$file" \
        --kind "$PLOT_KIND" --pdf "$PDF_TYPE" --format "$PLOT_FORMAT" \
        --output-dir "$destination" "$@"; then
        processed=$((processed + 1))
    else
        failed=$((failed + 1))
        printf 'FAILED: %s; continuing with the next file\n' "$file" >&2
    fi
done < <(find "$RESULTS_DIR" -type f -name '*_statistics.h5' -print0)

printf '\nFinished: %d successful files, %d failed files.\n' "$processed" "$failed"
if [[ "$processed" -eq 0 && "$failed" -eq 0 ]]; then
    printf 'No *_statistics.h5 files were found.\n'
fi
[[ "$failed" -eq 0 ]]
