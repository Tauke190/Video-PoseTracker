_base_ = './res50_num_frames_3_posetrack17.py'

model = dict(
    backbone=dict(
        frozen_stages=4,  # Freeze entire ResNet backbone
        with_cp=True),  # Enable gradient checkpointing to reduce memory
    bbox_head=dict(
        track_score_thr=0.5,
        track_nms_thr=0.9,
        track_train_score_thr=0.3,
        track_p_fn=0.4,
        max_track_queries=100,
        transformer=dict(
            reanchor_alpha=0.8,  # Pose-Aware Re-Anchoring: blend ratio
        ),
        # --- Freeze pose detection: zero out all keypoint regression losses ---
        loss_kpt=dict(type='opera.RLELoss', loss_weight=0.0),
        loss_kpt_rpn=dict(type='opera.RLELoss', loss_weight=0.0),
        loss_kpt_refine=dict(type='opera.RLELoss', loss_weight=0.0),
        # --- Keep classification loss (the only signal for tracking) ---
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),  # Increased from 0.5 — cls is now the primary loss
    ),
    train_cfg=dict(
        assigner=dict(
            type='opera.TrackAwarePoseHungarianAssigner',
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

# load_from ='Weights/petr_resnet50_coco.pth'

# Freeze everything except cls_branches (and decoder self-attention for query mixing)
# - backbone: lr_mult=0 (fully frozen via frozen_stages=4 too)
# - neck, encoder, hm_encoder, kpt_branches, sigma_branches: lr_mult=0
# - decoder + cls_branches: trainable (learn track-vs-detect scoring & query interaction)
optimizer = dict(
    lr=1e-4,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.0),
            'neck': dict(lr_mult=0.0),
            'bbox_head.transformer.encoder': dict(lr_mult=0.0),
            'bbox_head.transformer.hm_encoder': dict(lr_mult=0.0),
            'kpt_branches': dict(lr_mult=0.0),
            'refine_kpt_branches': dict(lr_mult=0.0),
            'pre_kpt_branches': dict(lr_mult=0.0),
            'next_kpt_branches': dict(lr_mult=0.0),
            'pre_refine_kpt_branches': dict(lr_mult=0.0),
            'next_refine_kpt_branches': dict(lr_mult=0.0),
            'dec_fc_sigma_branches': dict(lr_mult=0.0),
            'refine_fc_sigma_branches': dict(lr_mult=0.0),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1),
        }))

lr_config = dict(policy='step', step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)

# Validate on a small subset for fast hyperparameter feedback.
# - val_samples: number of val images to use (None = full val set)
# - val_iter_interval: run validation every N batches within an epoch
#   (None = only at epoch end)
evaluation = dict(val_samples=100, val_iter_interval=400)