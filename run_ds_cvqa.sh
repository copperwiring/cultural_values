#!/bin/bash

#Set job requirements
#SBATCH --array=0-1
#SBATCH -n 1
#SBATCH -t 0:50:00
#SBATCH -p gpu_h100
#SBATCH --gpus-per-node=2
#SBATCH --job-name=llava72b
#SBATCH --output=slurm_output_%A_%a_llava72b.out





export PYTHONPATH=$PYTHONPATH:/gpfs/work5/0/prjs0370/zhizhang/projects/cultural_values

# Set up the virtual environment
source activate llavanext

csv_file_list=("datasets/dollarstreet_accurate_images/ds_wvs_metadata.csv" "datasets/cvqa_images/cvqa_wvs_metadata.csv")
country_persona_list=("true")
#model_name=("liuhaotian/llava-v1.5-13b")
#model_name=("liuhaotian/llava-v1.6-34b")
model_name=("llava-hf/llava-next-72b-hf")


#Calculate total number of combinations
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
        --batch_size 8 \
        --num_workers 2 \
        --country_persona "${country_persona}"

elif [[ "${csv_file}" == *"cvqa_wvs_metadata.csv" ]]; then
    python main/run_main_cvqa.py \
        --model_name "${model_name}" \
        --output_dir 'output_results_llava' \
        --csv_file_path "${csv_file}" \
        --batch_size 8 \
        --num_workers 2 \
        --country_persona "${country_persona}"
else
    echo "Invalid csv file path"
fi


