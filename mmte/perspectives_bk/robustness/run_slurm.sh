#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1

# init virtual environment
module load anaconda/2024.02
source activate mllm-dev


# command to run
python /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/test_chat_lc.py