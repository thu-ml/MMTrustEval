model_ids=(
    'InternVL-Chat-ViT-6B-Vicuna-13B'
    'LVIS-Instruct4V'
    'ShareGPT4V-13B'
    'ShareGPT4V-7B'
    'claude-3-sonnet-20240229'
    'cogvlm-chat-hf'
    'gemini-pro'
    'gemini-pro-vision'
    'gpt-3.5-turbo'
    'gpt-4-0613'
    'gpt-4-1106-preview'
    'gpt-4-1106-vision-preview'
    'instructblip-flan-t5-xxl'
    'instructblip-vicuna-13b'
    'instructblip-vicuna-7b'
    'internlm-xcomposer-7b'
    'internlm-xcomposer2-vl-7b'
    'llava-rlhf-13b'
    'llava-v1.5-13b'
    'llava-v1.5-7b'
    'llava-v1.6-13b'
    'lrv-instruction'
    'mplug-owl-llama-7b'
    'mplug-owl2-llama2-7b'
    'otter-mpt-7b-chat'
    'qwen-vl-chat'
    'qwen-vl-max'
    'qwen-vl-plus'
)

# Number of GPUs available
num_gpus=8

# Function to run a model on a specified GPU
run_model_on_gpu() {
    local model_id=$1
    local gpu_id=$2
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python -m docker.test_env.run_models --model-id "$model_id" > "${model_id}.log" 2>&1 &
    echo "Started model $model_id on GPU $gpu_id"
}

# Process models in batches of 8
for ((i=0; i<${#model_ids[@]}; i+=num_gpus)); do
    echo "Starting batch $((i/num_gpus + 1))"
    
    # Run each model in the current batch
    for ((j=0; j<num_gpus; j++)); do
        idx=$((i + j))
        if [[ $idx -lt ${#model_ids[@]} ]]; then
            run_model_on_gpu "${model_ids[idx]}" "$j"
        fi
    done
    
    # Wait for the current batch to complete
    wait
    echo "Batch $((i/num_gpus + 1)) completed"
done

echo "All models have been started."
