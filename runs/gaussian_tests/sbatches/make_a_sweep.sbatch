#!/bin/bash
#SBATCH --job-name=tables_scRatio
#SBATCH --output=./runs/gaussian_tests/sbatches/outputs/sweep_output_%A.txt
#SBATCH --error=./runs/gaussian_tests/sbatches/outputs/sweep_error_%A.txt
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=30Gb
#SBATCH --cpus-per-task=10
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=0

export HYDRA_FULL_ERROR=1

python ./runs/gaussian_tests/scripts/make_a_sweep.py \
    --multirun \
    +num_dims=2,5,10,20,30 \
    +lr=0.001,0.0001 \
    +cond_latent_dim=32,64,128 \
    +start_from_scratch=false \
    +launcher=gpu_normal