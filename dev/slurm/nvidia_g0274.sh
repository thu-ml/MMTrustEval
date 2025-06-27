#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --partition=FM
#SBATCH --nodelist=g0274

nvidia-smi