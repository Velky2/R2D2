#!/bin/bash
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -p batch-AMD

source ~/.bashrc
conda activate base

python -u Dataset_QM9-xTB_lite.py