#!/bin/bash
#SBATCH --job-name=rom_oracle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-dev

# Oracle a-posteriori runs of rom/lavalle-cpp for Re120 Ca0.01 beta45 (10 Hz restart, t = 1800-1900).
#   sbatch --output=<oracle_dir>/rom/logs/rom_%j.out run_rom_oracle.sh <oracle_dir>
# Runs <oracle_dir>/rom/runs/{lavalle,Tw,Tw_Mx,all}/ with the same input.txt, then the summary.
set -euo pipefail
ROOT="${THINFILM_ROOT:-/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/code/thinfilm-closure-ml}"
source "${ROOT}/env.sh"
module purge; module load gcc/14.2.0 hdf5
D="${1:?oracle directory}"
F="${D}/rom/rom_forcing.h5"
B="${D}/rom/thinfilm"
cp "${ROOT}/rom/lavalle-cpp/thinfilm" "${B}"
echo "git commit: $(git -C "${ROOT}" rev-parse HEAD)" > "${D}/rom/SOURCE_INFO.txt"

declare -A TERMS=([lavalle]=none [Tw]=Tw [Tw_Mx]=Tw,Mx [all]=Tw,Mx,Px,Dx)
for name in lavalle Tw Tw_Mx all; do
    R="${D}/rom/runs/${name}"; mkdir -p "${R}"; cd "${R}"
    cat > input.txt <<IN
fcut_dim  = 20.0
MNOISE    = 1000
F0_noise  = 0.0

t_end  = 99.99
dt     = 0.005
t_save = 0.1

L = 800.0
N = 4000

Re_input  = 120.0
theta_deg = 45.0
rho       = 1072.0
mu        = 0.00673
Ca_input  = 0.01
IN
    printf 'forcing_file = %s\nterms = %s\n' "${F}" "${TERMS[$name]}" > oracle.txt
    echo "=== ${name} (terms ${TERMS[$name]})"
    "${B}" input.txt oracle.txt > run.log 2>&1 && tail -n 2 run.log || { echo "FAILED ${name}"; tail -n 5 run.log; }
done

cd "${D}/rom"
"${THINFILM_ENV}/bin/python" "${ROOT}/scripts/oracle_summary.py" --forcing "${F}" \
    --run lavalle=runs/lavalle/thinfilm_training_data.h5 --run Tw=runs/Tw/thinfilm_training_data.h5 \
    --run Tw_Mx=runs/Tw_Mx/thinfilm_training_data.h5 --run all=runs/all/thinfilm_training_data.h5 \
    --csv summary.csv | tee summary.txt
