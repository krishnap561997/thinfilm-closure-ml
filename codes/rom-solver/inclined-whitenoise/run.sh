#!/bin/bash
#SBATCH --job-name=rom       #A name for your job
#SBATCH -o run_%j.out        #Name output file
#SBATCH --mail-type=ALL      #What emails you want
#!SBATCH --mail-user=<krishnap561997@gmail.com>   #Where
#SBATCH --nodes 1             #Request 1 node
#SBATCH --ntasks 1            #Request number of processors
#SBATCH --mem-per-cpu=30000mb   #Per processor memory request
#SBATCH -t 0-12:10:00   #Walltime in hh:mm:ss or d-hh:mm:ss
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#!SBATCH --partition=hpg2-compute

#
# Change to this job's submit directory
cd $SLURM_SUBMIT_DIR

date
hostname

module load matlab/2023a
export KMP_STACKSIZE=1g

#!export OMP_NUM_THREADS=`cat $PBS_NODEFILE | wc -l`

./main_whitenoise

