#!/bin/bash 

module load anaconda/2024.02
#SBATCH -N 1
#SBATCH -p gpu # partition
#SBATCH -o /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/sbatch_logs/job-run_privacy-query-images_models.o # STDOUT
#SBATCH -e /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/sbatch_logs/job-run_privacy-query-images_models.e # STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jankinfmail@gmail.com
#SBATCH --gres=gpu:6

## SBATCH -n 1 # core

# source activate mllm-dev
# salloc -N 1 -p gpu --gres=gpu:6 --mem-per-gpu=80G /bin/bash

export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
cd /data/zhangyichi/Trustworthy-MLLM/MMTrustEval
    
model_ids=( 
    "ShareGPT4V-7B"
    "cogvlm-chat-hf"
    "gpt-4-1106-vision-preview"
    "internlm-xcomposer-7b"
    "llava-v1.5-7b"
    "minigpt-4-vicuna-13b-v0"
    "otter-mpt-7b-chat"
    "qwen-vl-chat"
    "mplug-owl2-llama2-7b"
    # "minigpt-4-llama2-7b"
)

for model_id in "${model_ids[@]}";
do
    # CUDA_VISIBLE_DEVICES=0 nohup python -u test_run_new.py --task confaide-info-sensitivity --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide-info-sensitivity-on-${model_id}.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=1 nohup python -u test_run_new.py --task confaide-infoflow-expectation-text --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide-infoflow-expectation-text-on-${model_id}.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=2 nohup python -u test_run_new.py --task confaide-infoflow-expectation-images --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide-infoflow-expectation-images-on-${model_id}.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=3 nohup python -u test_run_new.py --task confaide-infoflow-expectation-unrelated-images-nature --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide-infoflow-expectation-unrelated-images-nature-on-${model_id}.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=4 nohup python -u test_run_new.py --task confaide-infoflow-expectation-unrelated-images-noise --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide-infoflow-expectation-unrelated-images-noise-on-${model_id}.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=5 nohup python -u test_run_new.py --task confaide-infoflow-expectation-unrelated-images-color --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide-infoflow-expectation-unrelated-images-color-on-${model_id}.log 2>&1 &
    # wait
done
