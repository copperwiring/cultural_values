#!/bin/bash
#SBATCH --job-name=sy_cvqa
#SBATCH --cpus-per-task=4 --mem=28G
#SBATCH --partition=gpu
#SBATCH --time=48:00:00

echo $HOSTNAME
echo $CUDA_VISIBLE_DEVICES

python3 /home/vsl333/cultural_values/download_data/download_cvqa.py