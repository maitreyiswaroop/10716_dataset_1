#!/bin/bash
#SBATCH --job-name=
#SBATCH --output=/data/user_data/mswaroop/10716_dataset_1/logs/baselines_%j.out
#SBATCH --error=/data/user_data/mswaroop/10716_dataset_1/logs/baselines_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --partition=debug

# venv
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate venv 

# Set base directories
BASE_DIR="/data/user_data/mswaroop/10716_dataset_1"
mkdir -p ${BASE_DIR}/{results/run_val,logs}

# Set CUDA device
# Check if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "GPU available, using GPU"
    export CUDA_VISIBLE_DEVICES=0
else
    echo "No GPU available, using CPU"
    export CUDA_VISIBLE_DEVICES=""
fi

# Run training script
python3 baselines.py

# EOF