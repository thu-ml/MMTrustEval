#!/bin/bash
# SBATCH -N 1
# SBATCH --gres=gpu:1
# SBATCH -w FM

# init virtual environment
module load anaconda/2024.02
source activate /data/home/zhangyichi/miniconda3/envs/mllm-dev

TASK_ID=static_jailbreak_prompt
MODEL_ID=ShareGPT4V-13B
OUTPUT_DIR=/data/zhangyichi/Trustworthy-MLLM/output/safety/$TASK_ID
# log file path contain the task id and model id
LOG_FILE_PATH=/data/zhangyichi/Trustworthy-MLLM/output/safety/log/slurm-$TASK_ID-$MODEL_ID.out
# SBATCH -o=/data/zhangyichi/Trustworthy-MLLM/output/safety/log/slurm-$TASK_ID-$MODEL_ID.out
python test_run_george.py --task $TASK_ID --model_id $MODEL_ID --output_dir $OUTPUT_DIR