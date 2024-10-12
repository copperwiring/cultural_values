#!/bin/bash
#SBATCH --job-name=sy_dscvqa
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4 --mem=50G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --time=0-30:00:00

echo $HOSTNAME
echo $CUDA_VISIBLE_DEVICES

cd /home/vsl333/cultural_values

# Set PYTHONPATH to include your desired directory
export PYTHONPATH=$PYTHONPATH:/home/vsl333/cultural_values

# Set up the virtual environment
source /home/vsl333/cultural_values/culture-values/bin/activate

csv_file_list=("/home/vsl333/cultural_values/datasets/dollarstreet_accurate_images/ds_wvs_metadata.csv" "/home/vsl333/cultural_values/datasets/cvqa_images/cvqa_wvs_metadata.csv")
country_persona_list=("true" "false")
# model_name=("liuhaotian/llava-v1.6-34b" "liuhaotian/llava-v1.5-13b")
model_name=("liuhaotian/llava-v1.6-34b")


 Calculate total number of combinations
total_csv=${#csv_file_list[@]}
total_persona=${#country_persona_list[@]}
total_combinations=$(( total_csv * total_persona ))

# Check if SLURM_ARRAY_TASK_ID is within the valid range
if [ "$SLURM_ARRAY_TASK_ID" -ge "$total_combinations" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is out of range (0-$((total_combinations-1)))"
    exit 1
fi

# Determine indices for csv_file and country_persona based on SLURM_ARRAY_TASK_ID
csv_index=$(( SLURM_ARRAY_TASK_ID / total_persona ))
persona_index=$(( SLURM_ARRAY_TASK_ID % total_persona ))

# Retrieve the corresponding csv_file and country_persona
csv_file=${csv_file_list[$csv_index]}
country_persona=${country_persona_list[$persona_index]}

echo "Running on GPU: $CUDA_VISIBLE_DEVICES"
echo "Slurm Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on model: $model_name"
echo "Running on csv file: $csv_file with country_persona: $country_persona"


# Execute the appropriate Python script based on the csv_file type
if [[ "${csv_file}" == *"ds_wvs_metadata.csv" ]]; then
    python main/run_main_ds.py \
        --model_name "${model_name}" \
        --output_dir 'output_results_llava' \
        --csv_file_path "${csv_file}" \
        --batch_size 18 \
        --num_workers 2 \
        --country_persona "${country_persona}"

elif [[ "${csv_file}" == *"cvqa_wvs_metadata.csv" ]]; then
    python main/run_main_cvqa.py \
        --model_name "${model_name}" \
        --output_dir 'output_results_llava' \
        --csv_file_path "${csv_file}" \
        --batch_size 18 \
        --num_workers 2 \
        --country_persona "${country_persona}"
else
    echo "Invalid csv file path"
fi


