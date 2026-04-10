#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

module load python/3.10
module load cuda

source /scratch/trujim/myenv/bin/activate

python /scratch/trujim/test_gpu.py
