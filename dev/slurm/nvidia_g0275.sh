#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --nodelist=g0275

nvidia-smi