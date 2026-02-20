#!/usr/bin/env bash
cd /home/av354855/projects/PAVENet
CONFIG=configs/PAVE/res50_num_frames_3_posetrack17_tracking.py
CHECKPOINT=Weights/resnet50_posetrack.pth
GPUS=1
PORT=${PORT:-29500}
SAMPLES_PER_GPU=1
EVAL_WORK_DIR=work_dirs/eval_results_pavenet_tracking_epoch1

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $POSETRACK_TRACKING \
    $WEIGHTs \
    --eval keypoints \
    --gpu-id $GPU_ID \
    --cfg-options data.samples_per_gpu=$SAMPLES_PER_GPU \
    --work-dir $EVAL_WORK_DIR \
    --eval-options score_thr=0.5 joint_score_thr=0.1 save_dir=$EVAL_WORK_DIR