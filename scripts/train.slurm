#!/bin/bash
#SBATCH --job-name=pavenet_track
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --error=ucf_output/slurm-%j.err
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=6G 
#SBATCH --cpus-per-gpu=12


PORT=$((RANDOM % 55 + 12345))
while ss -tuln | grep -q ":$PORT"; do
  PORT=$((RANDOM % 55 + 12345))
done
echo "Free port found: $PORT"

# Load necessary modules
module load anaconda3
module load cuda

eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate pavenet
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

echo "Checking PyTorch CUDA version and availability:"
python -c "import torch; print('torch.version.cuda:', torch.version.cuda); print('torch.cuda.is_available:', torch.cuda.is_available())"

echo "========================================"
echo "Training script configuration:"
echo "========================================"
cat $0
echo "========================================"

cd /home/av354855/Video-PoseTracker

CONFIG=configs/PAVE/res50_num_frames_3_posetrack17_tracking.py
GPUS=2
PORT=${PORT:-29500}

python -m torch.distributed.launch --nproc_per_node=$GPUS tools/train.py $CONFIG