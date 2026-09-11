#!/bin/bash
#SBATCH --job-name=thinfilm
#SBATCH -o run_%j.out
#SBATCH -e run_%j.err

#SBATCH --mail-type=ALL
# Uncomment and add your email if you want notifications
# #SBATCH --mail-user=your_email@example.com

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --time=4-00:00:00

#SBATCH --account=bala1s
#SBATCH --qos=bala1s

# -------------------------
# Move to submit directory
# -------------------------
cd "$SLURM_SUBMIT_DIR" || exit 1

echo "Job started:"
date
hostname
echo "Working directory: $PWD"

# -------------------------
# Modules
# -------------------------
module purge
module load gcc/14.2.0
module load hdf5

# -------------------------
# Input
# -------------------------
INPUT_FILE=${1:-input.txt}

echo "Input file: $INPUT_FILE"

# -------------------------
# Run
# -------------------------
./thinfilm "$INPUT_FILE"

status=$?

echo "Job finished:"
date
echo "Exit status: $status"

exit $status
