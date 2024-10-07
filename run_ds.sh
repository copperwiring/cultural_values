#!/bin/bash
#SBATCH --job-name=sy_dscvqa
#SBATCH --array=0
#SBATCH --cpus-per-task=4 --mem=30G
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=gpu
#SBATCH --time=0-30:00:00

echo $HOSTNAME
echo $CUDA_VISIBLE_DEVICES

cd /home/vsl333/cultural_values

# Set PYTHONPATH to include your desired directory
export PYTHONPATH=$PYTHONPATH:/home/vsl333/cultural_values

# Set up the virtual environment
source /home/vsl333/cultural_values/culture-values/bin/activate

echo "Running on GPU: $CUDA_VISIBLE_DEVICES"
python main/run_main_ds.py --model_name 'liuhaotian/llava-v1.5-7b' --output_dir 'output_results' --csv_file_path '/projects/belongielab/people/vsl333/ds/dollarstreet_accurate_images/ds_wvs_metadata.csv' --batch_size 8 --num_workers 1 --country_persona True
ß