#!/bin/bash

model_ids=(
    # "phi-3.5"
    # "cambrian-8b"
    # "cambrian-13b"
    # "molmo-7b-d-0924"
    # "llava-v1.6-mistral-7b-hf"
    # "llava-v1.6-vicuna-7b-hf"
    # "llava-v1.6-vicuna-13b-hf"
    # "llama3-llava-next-8b-hf"
    # "llava-1.5-7b-hf"
    # "llava-1.5-13b-hf"
    # "llama-3-2-chat"
    # "internvl2-8b"
    # "mplug-owl3-7b-240728"
    # "qwen2-vl-chat"
    # "glm-4v-chat"
    # "cogvlm2-llama3-chat-19b"
    # "deepseek-janus-7b"
    "deepseek-vl2"
)
# Activate the virtual environment
export APIKEY_FILE="env/apikey_local.yml" 
eval "$(conda shell.bash hook)"
conda activate multitrust_tf45

# Set the main directory
main_dir="scripts/echo_run"
# Get the passed model_id
for model_id in "${model_ids[@]}"; do
    # echo $model_id
    # Loop through all subdirectories in the main directory
    for sub_dir in "$main_dir"/*; do
        if [ -d "$sub_dir" ]; then
            # echo "Entering directory: $sub_dir"
            # Loop through all .sh files in the subdirectory
            for sh_file in "$sub_dir"/*.sh; do
                if [ -f "$sh_file" ]; then
                    # echo "Running script: $sh_file"
                    # Run the script file and pass model_id
                    bash "$sh_file" "$model_id"
                fi
            done
        fi
    done
done
