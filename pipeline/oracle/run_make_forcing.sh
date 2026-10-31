#!/bin/bash
#SBATCH --job-name=rom_forcing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-dev

# Build <oracle_dir>/rom/rom_forcing.h5 from <oracle_dir>/closures/closures_part_*.h5.
#   sbatch --output=<oracle_dir>/rom/logs/forcing_%j.out run_make_forcing.sh <oracle_dir> [N]
set -euo pipefail
ROOT="${THINFILM_ROOT:-/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/code/thinfilm-closure-ml}"
source "${ROOT}/env.sh"
D="${1:?oracle directory}"; N="${2:-4000}"
mkdir -p "${D}/rom"
"${THINFILM_ENV}/bin/python" -u "${ROOT}/pipeline/oracle/make_rom_forcing.py" "${D}"/closures/closures_part_*.h5 \
    -o "${D}/rom/rom_forcing.h5" --N "${N}"
