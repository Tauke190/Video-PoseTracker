#!/bin/bash
cd /home/av354855/projects/PAVENet
conda activate pavenet

# Add project to PYTHONPATH
export PYTHONPATH=/home/av354855/projects/PAVENet:$PYTHONPATH

# Usage: python demo/image_demo.py <image> <config.py> <checkpoint.pth> [options]

# Example: Run inference on an image
python demo/image_demo.py \
    demo/players.jpeg \
    configs/PAVE/res50_num_frames_3_posetrack17.py \
    Weights/resnet50_posetrack.pth \
    --device cuda:1 \
    --score-thr 0.3 \
    --out-file demo/result.jpg

