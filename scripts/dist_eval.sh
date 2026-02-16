#!/usr/bin/env bash
cd /home/av354855/projects/PAVENet
CONFIG=configs/PAVE/res50_num_frames_3_posetrack17.py
CHECKPOINT=Weights/resnet50_posetrack.pth
GPUS=1
PORT=${PORT:-29500}

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $CONFIG $CHECKPOINT --eval keypoints --launcher pytorch ${@:4} \