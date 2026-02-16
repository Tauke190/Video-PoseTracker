#!/bin/bash
cd /home/av354855/projects/PAVENet
conda activate pavenet

# Add project to PYTHONPATH
export PYTHONPATH=/home/av354855/projects/PAVENet:$PYTHONPATH

# Usage: python demo/video_demo.py <video> <config.py> <checkpoint.pth> [options]

# Example: Run inference on a video and save output with pose tracking
python demo/video_demo.py \
    demo/walking.mp4 \
    configs/PAVE/res50_num_frames_3_posetrack17.py \
    Weights/resnet50_posetrack.pth \
    --device cuda:1 \
    --score-thr 0.3 \
    --out-file demo/output_walking_tracking.mp4 \
    --tracking --joint-ids


# Example: Run inference with pose tracking AND joint-level tracking (showing keypoint IDs)
# python demo/video_demo.py \
#     demo/sidewalk.mp4 \
#     configs/PAVE/res50_num_frames_3_posetrack17.py \
#     Weights/resnet50_posetrack.pth \
#     --device cuda:1 \
#     --score-thr 0.3 \
#     --out-file demo/output_sidewalk_tracking_with_joints.mp4 \
#     --tracking \
#     --joint-ids

# Example: Show video without saving
# python demo/video_demo.py \
#     demo/sample_video.mp4 \
#     configs/PAVE/res50_num_frames_3_posetrack17.py \
#     Weights/resnet50_posetrack.pth \
#     --show \
#     --score-thr 0.3

# Example: Save output to file
# python demo/image_demo.py \
#     demo/players.jpeg \
#     configs/PAVE/res50_num_frames_3_posetrack17.py \
#     Weights/resnet50_posetrack.pth \
#     --out-file output.jpg \
#     --score-thr 0.3