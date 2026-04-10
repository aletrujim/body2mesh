#!/bin/bash
#SBATCH --job-name=body2mesh
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

#SBATCH --output=/scratch/trujim/logs/%j.out
#SBATCH --error=/scratch/trujim/logs/%j.err

echo "=== INICIO JOB ==="
hostname
nvidia-smi

# Cargar módulos
module load python/3.11
module load cuda
module load gcc opencv

export PYTHONPATH=/scratch/trujim/body2mesh_run/body2mesh:$PYTHONPATH
export PYTHONPATH=/scratch/trujim/body2mesh_run/body2mesh/human-pose:$PYTHONPATH

echo "=== TEST OPENCV ==="
python -c "import cv2; print('CV2 OK', cv2.__version__)"

echo "=== EJECUTANDO PIPELINE ==="

python /scratch/trujim/body2mesh_run/run_body2mesh.py \
    --input_dir /scratch/trujim/data/input \
    --output_dir /scratch/trujim/data/output \
    --repo_path /scratch/trujim/body2mesh_run/body2mesh

echo "=== FIN JOB ==="
