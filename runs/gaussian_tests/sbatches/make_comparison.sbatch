#!/bin/bash
#SBATCH --job-name=tables_scRatio
#SBATCH --output=./runs/gaussian_tests/sbatches/outputs/tables_output_%A.txt
#SBATCH --error=./runs/gaussian_tests/sbatches/outputs/tables_error_%A.txt
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=30Gb
#SBATCH --cpus-per-task=10
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=0

export HYDRA_FULL_ERROR=1

python ./runs/gaussian_tests/scripts/make_comparison.py \
    --multirun \
    +num_samples=110_000 \
    +loc=1,2 \
    +num_dims=2,5,10,20,30 \
    +start_from_scratch=false \
    +launcher=super_gpu_normal