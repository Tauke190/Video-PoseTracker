#!/usr/bin/env bash
CONFIG=configs/PAVE/res50_num_frames_3_posetrack17_tracking.py
GPUS=1
PORT=${PORT:-29502}

# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#                          tools/train.py $CONFIG --launcher pytorch ${@:3}

CUDA_VISIBLE_DEVICES=2 python tools/train.py $CONFIG
                          
# --cfg-options evaluation.start=0