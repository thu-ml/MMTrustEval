#!/bin/bash
#SBATCH --job-name=check_job        # Job name
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=10M                  # Memory limit for the job
#SBATCH --cpus-per-task=1           # Number of CPU cores per task
#SBATCH --partition=FM   # Ensure it runs on a CPU-only partition
#SBATCH --nodelist=g0274

# Command to execute (sleep for 1 hour)
sleep 3600

# Keep the job alive
while true; do sleep 60; done

