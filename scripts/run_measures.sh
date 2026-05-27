#!/bin/bash
#SBATCH --job-name=measures
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/trujim/measures_b/results/logs/output_%j.txt

module load python/3.10

source ~/body2vec_env/bin/activate

cd /scratch/trujim/measures_b/scripts

python process_measures.py
