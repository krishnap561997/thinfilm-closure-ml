#!/bin/sh
#SBATCH --job-name=test
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=krishnap.kalivel@ufl.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l4:1
#SBATCH --partition=hpg-turin
#SBATCH --ntasks=4
#SBATCH --mem=24gb
#SBATCH --time=00-02:00:00
#SBATCH --output=testing.txt

export PATH=/blue/bala1s/krishnap.kalivel/ThinFilmFlow/thinfilm-closure-ml/software/Pytorch_ENV_THF/bin:$PATH

#module load intel/2025.1.0
#module load pytorch/1.13

for i in $(seq 1 1 48)
do
	python3 testing_bb.py --fl_no $i --num_lyrs 6 --lyr_wdt 250 --num_neig 26
done
