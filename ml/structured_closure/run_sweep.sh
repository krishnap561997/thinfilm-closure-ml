#!/bin/bash
# C.2 sweep of the structured closure V1: lam_lin {0, 10} x lam_F {0, 0.1} x sigma {0.5, 0.25}, one seed.
# Submit (8 runs, GPU, 4 at a time):
#   sbatch --array=0-7%4 ml/structured_closure/run_sweep.sh            # seed 0
#   SEED=1 sbatch --export=ALL --array=0-7%4 ml/structured_closure/run_sweep.sh   # backup seed
#   SWEEP=power sbatch --export=ALL --array=0-3 ml/structured_closure/run_sweep.sh # power penalty runs
#SBATCH --job-name=sc_sweep
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=01:00:00
#SBATCH --output=/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/ml-models/structured_closure/logs/sweep_%A_%a.out
source /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/code/thinfilm-closure-ml/env.sh
SEED=${SEED:-0}
i=${SLURM_ARRAY_TASK_ID}
LL=(0 10); LF=(0 0.1); SG=(0.5 0.25); LP=(10 100)
if [ "${SWEEP:-base}" = power ]; then          # SWEEP=power: sigma {0.5, 0.25} x lam_P {10, 100}, lam_lin 10, lam_F 0 (tasks 0-3)
  lam_lin=10; lam_F=0; lam_P=${LP[$((i % 2))]}; sigma=${SG[$((i / 2))]}
  name="v1_s${sigma}_L${lam_lin}_F${lam_F}_P${lam_P}_seed${SEED}"
else
  lam_lin=${LL[$((i % 2))]}; lam_F=${LF[$(((i / 2) % 2))]}; sigma=${SG[$((i / 4))]}; lam_P=0
  name="v1_s${sigma}_L${lam_lin}_F${lam_F}_seed${SEED}"
fi
echo "task $i: $name"
$THINFILM_ENV/bin/python -I $THINFILM_ROOT/ml/structured_closure/train.py --name "$name" --device cuda \
  --sigma "$sigma" --lam_lin "$lam_lin" --lam_F "$lam_F" --lam_P "$lam_P" --seed "$SEED" --epochs 100 --val_every 10 --lin_cases os
