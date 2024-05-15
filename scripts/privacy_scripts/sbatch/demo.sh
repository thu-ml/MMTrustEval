#!/bin/bash
# SBATCH -N 1
# SBATCH --gres=gpu:1
# SBATCH -w FM

# init virtual environment
module load anaconda/2024.02
source activate /data/home/zhangyichi/miniconda3/envs/mllm-dev


# command to run
python test_run_fairness.py --task subjective-choice-text --output_dir /data/zhangyichi/Trustworthy-MLLM/output/fairness/subjective_choice_text