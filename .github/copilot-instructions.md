# Copilot Instructions for PAVENet

## Project Overview
PAVENet is an end-to-end multi-person 2D video pose estimation framework (AAAI 2026). It eliminates heuristic steps (detection, NMS) by using a spatial encoder + spatiotemporal pose decoder with pose-aware attention. Based on [PETR/opera](https://github.com/hikvision-research/opera), integrates MMCV 1.5.3 + MMDetection 2.25.0. Components borrowed from DCPose, DSTA, and RLE.

## Architecture (TrackFormer-style)
The model follows a **TrackFormer-inspired** track query / detection query paradigm:
- **Track queries**: Propagated from the previous frame's high-confidence outputs. Prepended before detection queries in the transformer. Hard-assigned to GT by track ID (no cost matching).
- **Detection queries**: 300 learnable queries that detect new/untracked persons via Hungarian matching on remaining GTs.
- **Sequential training**: `T=2` windows per step. Window 0 has no track queries; window 1 receives track state from window 0 with **pFN dropout** (`track_p_fn=0.4`) to force detection queries to cover dropped tracks.
- **Key files**: `opera/models/detectors/pave.py` (model + tracking state), `opera/models/dense_heads/pave_head_mul_frames.py` (head + loss), `opera/core/bbox/assigners/hungarian_assigner.py` (TrackAwarePoseHungarianAssigner).

### Data Flow
```
Backbone (ResNet/Swin, frozen) → Neck (ChannelMapper) → Spatial Encoder (6-layer MSDA)
  → Heatmap Encoder (1-layer, generates proposals) → Pose Decoder (3-layer, pose-aware cross-attn)
  → Refine Decoder (2-layer) → cls_branches (Linear→sigmoid) + kpt_branches (keypoints + sigma)
```

### Loss Landscape
- **RLELoss** (`loss_kpt`, `loss_kpt_rpn`, `loss_kpt_refine`): Negative log-likelihood loss; produces **negative total loss** (~-29 to -30). This is expected—more negative = better. Dominates gradient magnitude.
- **FocalLoss** (`loss_cls`, `enc_loss_cls`): Classification loss with weight=0.5. Produces small gradients by design (focal mechanism suppresses easy negatives).
- **OKSLoss** / **CenterFocalLoss**: Currently disabled (`loss_weight=0.0`).
- **Known issue**: RLE gradients dominate the shared decoder, causing cls_branches calibration drift. Monitor `det_cls_val` and `trk_cls_val` in logs for score suppression.

### Training Log Fields
- `n_trk` / `n_trk_pos` / `n_det_pos`: Track query count / matched tracks / matched detections
- `trk_id_con`: Track ID consistency (fraction of track queries matched to same GT identity)
- `trk_cls_val` / `det_cls_val`: Diagnostic FocalLoss split by query type (not backpropped)
- `loss`: Total loss (negative due to RLE); should decrease (more negative) during training

### Evaluation
- `score_thr=0.5` in eval config filters predictions in `_kpt2json` → Precision=100% but Recall sensitive to score suppression
- `val_samples=100`: Only 100 validation samples used for periodic eval during training
- Metrics: Per-joint mAP (Head, Shoulder, etc.), Mean mAP, Person_MOTA, Person_Precision, Person_Recall

## Key Components
- **opera/models/**: Detectors (`pave.py`), heads (`pave_head_mul_frames.py`), transformers, losses (`oks_loss.py` for RLE/OKS), attention modules
- **opera/core/**: Assigners (`hungarian_assigner.py`), evaluation, post-processing, runner
- **opera/datasets/**: `posetrack_video_pose.py`, pipelines for multi-frame loading
- **configs/PAVE/**: Python config files controlling all experiment parameters
- **Weights/**: Pretrained full-model checkpoints (backbone + neck + transformer + heads)
- **work_dirs/**: Logs, checkpoints, eval results as JSON

## Developer Workflows
- **Training**: `python tools/train.py --cfg configs/PAVE/res50_num_frames_3_posetrack17_tracking.py`
- **Evaluation**: `python tools/test.py --cfg configs/PAVE/res50_num_frames_3_posetrack17_tracking.py`
- **Video Demo**: `python demo/video_demo.py --config <config> --checkpoint <weights>`
- **Logs**: JSON lines in `work_dirs/<exp_name>/*.log.json`; first line is env/config metadata

## Conventions
- **Config-driven**: All experiments via Python config files (not YAML). Never hardcode parameters.
- **Registry pattern**: Models, losses, datasets use `@REGISTRY.register_module()` decorators (MMCV/MMDet convention). Type strings prefixed with `opera.`, `mmdet.`, or `mmcv.`.
- **Multi-frame pipelines**: `MulResize`, `MulRandomFlip`, `MulRandomCrop`, `MulKeypointRandomAffineForFrames3` — always use `Mul` variants for video data.
- **Gradient accumulation**: `cumulative_iters=8` via `GradientCumulativeOptimizerHook` — effective batch size = 8 × `samples_per_gpu`.

## Tips for AI Agents
- The total training loss is **negative** — this is correct (RLE log-likelihood). Focus on whether it *decreases* over time.
- When debugging recall issues: check `det_cls_val` trend (should not collapse to 0), and try lowering `score_thr` in evaluation config.
- When modifying losses: the RLE loss gradient magnitude dwarfs FocalLoss. Adjust `loss_weight` ratios or add `lr_mult` for `cls_branches` in `paramwise_cfg`.
- Pretrained weights in `Weights/` are **full model** (not just backbone) — `load_from` loads all matching keys.
- Follow modular structure in `opera/models/` and `opera/datasets/` when adding new components.
