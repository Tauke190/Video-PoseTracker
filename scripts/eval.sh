#!/bin/bash
cd /home/av354855/projects/PAVENet

# Add project to PYTHONPATH
export PYTHONPATH=/home/av354855/projects/PAVENet:$PYTHONPATH

# Usage: python tools/test.py <config> <checkpoint> [options]
#
# Key arguments:
#   --eval keypoints     : Evaluate keypoint metrics (AP, AR)
#   --show-dir DIR       : Save visualizations to directory
#   --gpu-id N           : GPU to use (default: 0)
#   --work-dir DIR       : Directory to save evaluation results
#   --cfg-options        : Override config options

# ============ Configurable Parameters ============
SAMPLES_PER_GPU=1          # Batch size per GPU
GPU_ID=0                   # Which GPU to use
CUDA_DEVICE=1              # CUDA_VISIBLE_DEVICES value
# =================================================

COCO=configs/_base_/datasets/coco_keypoint.py
POSETRACK=configs/PAVE/res50_num_frames_3_posetrack17.py
CROWDPOSE=opera/datasets/crowd_pose.py

# Run evaluation on PoseTrack17 validation set
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python tools/test.py \
    configs/PAVE/res50_num_frames_3_posetrack17.py \
    Weights/resnet50_posetrack.pth \
    --eval keypoints \
    --gpu-id $GPU_ID \
    --cfg-options data.samples_per_gpu=$SAMPLES_PER_GPU \
    --work-dir work_dirs/eval_results_pavenet \
    --eval-options score_thr=0.5 joint_score_thr=0.1

#CrowdPose evaluation
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python tools/test.py \
#     configs/PAVE/res50_num_frames_3_posetrack17.py \
#     Weights/petr_swin-l-p4-w7-_16x1_100e_crowdpose.pth \
#     --eval keypoints \
#     --gpu-id $GPU_ID \
#     --cfg-options data.samples_per_gpu=$SAMPLES_PER_GPU \
#     --work-dir work_dirs/eval_results_petr_posetrack

# configs/PETR/petr_swin-l-p4-w7-224-22kto1k_16x1_100e_crowdpose_flip_test.py

# Example 3: Visualize results
# python tools/test.py \
#     configs/PAVE/res50_num_frames_3_posetrack17.py \
#     work_dirs/res50_num_frames_3_posetrack17/latest.pth \
#     --show-dir work_dirs/vis_results \
#     --show-score-thr 0.3

