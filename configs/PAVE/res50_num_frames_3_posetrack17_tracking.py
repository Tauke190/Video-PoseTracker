_base_ = './res50_num_frames_3_posetrack17.py'

model = dict(
    bbox_head=dict(
        track_score_thr=0.5,
        max_track_queries=100,
    ),
    train_cfg=dict(
        assigner=dict(
            type='opera.TrackAwarePoseHungarianAssigner',
            track_bonus=10.0,
            cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
            kpt_cost=dict(type='opera.KptL1Cost', weight=70.0),
            oks_cost=dict(type='opera.OksCost', num_keypoints=15, weight=7.0),
        ),
    ),
)

# Sequential dataset for track query training
data = dict(
    samples_per_gpu=1,
    train=dict(
        type='opera.PosetrackSequentialDataset',
        seq_length=2,
    ),
)

# Fine-tune from existing checkpoint
load_from = 'Weights/resnet50_posetrack.pth'

# Lower LR for fine-tuning
optimizer = dict(lr=1e-5)
lr_config = dict(policy='step', step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)