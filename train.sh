#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 2
#SBATCH -n 1
#SBATCH -o train.out

python train.py config_IDCycleGAN.json
