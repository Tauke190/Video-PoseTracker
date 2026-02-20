_base_ = [
    '../_base_/datasets/posetrack17_video_keypoint.py', '../_base_/default_runtime.py'
]

model = dict(
    type='opera.PAVE',
    backbone=dict(
        type='mmdet.ResNet',
        init_cfg=dict(type='Pretrained', checkpoint='Weights/resnet50_posetrack.pth', prefix='backbone.'),
        input_type='mul_frames',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=4,  # Freeze entire ResNet backbone (tracking override)
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        with_cp=True),  # Enable gradient checkpointing to reduce memory (tracking override)
    neck=dict(
        type='mmdet.ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='opera.PAVEHeadMulFrames',
        num_frames=3,
        num_keypoints=15,
        num_query=300,
        num_classes=1,  # only person
        in_channels=2048,
        sync_cls_avg_factor=True,
        with_kpt_refine=True,
        as_two_stage=True,
        # Tracking-specific parameters
        track_nms_thr=0.9,
        track_score_thr=0.3,  # Inference track propagation score threshold
        track_train_score_thr=0.3,
        track_p_fn=0.4,
        max_track_queries=100,
        transformer=dict(
            type='opera.TransformerMulFrames',
            num_keypoints=15,
            num_frames=3,
            reanchor_alpha=0.8,  # Pose-Aware Re-Anchoring: blend ratio (tracking override)
            encoder=dict(
                type='mmcv.DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='mmcv.BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='mmcv.MultiScaleDeformableAttention',
                        embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='opera.TransformerDecoderV2',
                num_keypoints=15,
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='mmcv.DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='opera.MulFramesMultiScaleDeformablePoseAttentionNumFrames3',
                            num_points=15,
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 
                                     'ffn', 'norm'))),
            hm_encoder=dict(
                type='mmcv.DetrTransformerEncoder',
                num_layers=1,
                transformerlayers=dict(
                    type='mmcv.BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='mmcv.MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=1),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            refine_decoder=dict(
                type='mmcv.DeformableDetrTransformerDecoderV1',
                num_layers=2,
                return_intermediate=True,
                transformerlayers=dict(
                    type='mmcv.DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='mmcv.MulFramesMultiScaleDeformableAttentionNumFrames3',
                            embed_dims=256,
                            im2col_step=128)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='mmcv.SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # Increased from 0.5 — cls is now the primary loss (tracking override)
        loss_kpt=dict(type='opera.RLELoss', loss_weight=0.05),  # Regularization to prevent collapse when freezing kpt branches (tracking override)
        loss_kpt_rpn=dict(type='opera.RLELoss', loss_weight=0.0),  # (tracking override)
        loss_oks=dict(type='opera.OKSLoss', num_keypoints=15, loss_weight=0.0),
        loss_hm=dict(type='opera.CenterFocalLoss', loss_weight=0.0),
        loss_kpt_refine=dict(type='opera.RLELoss', loss_weight=0.05),  # (tracking override)
        loss_oks_refine=dict(type='opera.OKSLoss', num_keypoints=15, loss_weight=0.0)),
    train_cfg=dict(
        assigner=dict(
            type='opera.TrackAwarePoseHungarianAssigner',  # (tracking override)
            cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
            kpt_cost=dict(type='opera.KptL1Cost', weight=70.0),
            oks_cost=dict(type='opera.OksCost', num_keypoints=15, weight=7.0))),
    test_cfg=dict(max_per_img=20))

# Sequential dataset for track query training (tracking override)
data = dict(
    samples_per_gpu=1,
    train=dict(
        type='opera.PosetrackSequentialDataset',
        seq_length=2,
    ),
)

# Optimizer with tracking-specific learning rates (tracking override)
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0001,
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
            'cls_branches': dict(lr_mult=3.0),
        }))

optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=8,
    grad_clip=dict(max_norm=1, norm_type=2))

# Learning policy
lr_config = dict(policy='step', step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=300) 
checkpoint_config = dict(interval=1, max_keep_ckpts=20)  

# Fine-tune from existing checkpoint
load_from = 'Weights/resnet50_posetrack.pth'

# Validate on a small subset for fast hyperparameter feedback (tracking override)
# - val_samples: number of val images to use (None = full val set)
# - val_iter_interval: run validation every N batches within an epoch
#   (None = only at epoch end)
evaluation = dict(val_samples=100, val_iter_interval=400)
