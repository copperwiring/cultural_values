#!/bin/bash
#SBATCH --job-name=sy_ds
#SBATCH --cpus-per-task=4 --mem=28G
#SBATCH --partition=gpu
#SBATCH --time=48:00:00

# Specify the output directory
OUTPUT_DIR="/projects/belongielab/people/vsl333/ds/"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Download the dataset to the specified output directory
kaggle datasets download mlcommons/the-dollar-street-dataset -p "$OUTPUT_DIR"

