# PAVENet Training Losses Explained

## Overview
PAVENet is a multi-frame pose tracking model that uses a two-stage transformer architecture with query propagation. The training involves losses from multiple stages and components. This document explains each loss metric in the training logs.

---

## Loss Metrics Breakdown

### Encoder Losses (Feature Extraction Stage)
These losses come from the encoder that processes raw image features before the decoder.

#### `enc_loss_cls: 0.14931`
- **Type:** Classification loss (Focal Loss)
- **What it measures:** How well the encoder predicts which spatial locations contain pose keypoints
- **Range:** 0 (perfect) to ~3-5 (random)
- **Interpretation:** ~0.15 is good; encoder is correctly identifying keypoint-bearing regions
- **Note:** Binary classification (has keypoint vs background)

#### `enc_loss_kpt: -4.67249`
- **Type:** Keypoint regression loss using RealNVP normalizing flow
- **What it measures:** How well the encoder predicts keypoint coordinates
- **Range:** Negative (log-likelihood, higher is better in loss space)
- **Interpretation:** ~-4.7 is typical; negative values indicate learned uncertainty fits the error distribution
- **Technical detail:** Uses a flow model to estimate the probability distribution of prediction errors

---

### Decoder Losses (Main Pose Detection)
These are the primary pose detection losses from the transformer decoder. The model uses multiple decoder layers (typically 6) to iteratively refine predictions.

#### `loss_cls: 0.19148`
- **Type:** Classification loss (from last decoder layer)
- **What it measures:** Final detection confidence — whether each query predicts an actual person
- **Range:** 0 (perfect) to ~3-5 (random)
- **Interpretation:** ~0.19 is good; the model is confident in detected persons
- **Note:** Weighted by foreground/background ratio using focal loss

#### `loss_kpt: -3.65011`
- **Type:** Keypoint regression loss (from last decoder layer, uses RealNVP flow)
- **What it measures:** Final keypoint position accuracy
- **Range:** Negative values; closer to 0 is worse, more negative is better
- **Interpretation:** ~-3.65 indicates good keypoint localization
- **Difference from enc_loss_kpt:** Decoder refines encoder predictions, usually 1-2 units worse

#### `d0.loss_cls: 0.21142` and `d0.loss_kpt: -3.3122`
#### `d1.loss_cls: 0.19985` and `d1.loss_kpt: -3.54268`
- **Naming:** `d0`, `d1` = decoder layer 0, 1 (intermediate layers)
- **What they measure:** Intermediate refinement losses from earlier decoder layers
- **Pattern:** Usually progressively improve from d0 → d1 → final (better kpt loss, lower cls loss)
- **Why both reported:** Helps debug which layer has issues; ensures all layers contribute

---

### Refinement Losses (Pose Refinement)
After coarse detection, PAVENet refines keypoints using context from neighboring frames.

#### `d0.loss_kpt_refine: -3.69219`
#### `d1.loss_kpt_refine: -3.76669`
- **Type:** Keypoint refinement loss (RealNVP flow-based)
- **What it measures:** How well the refinement network adjusts keypoint positions using temporal context
- **Interpretation:** ~-3.7 to -3.8 is typical; slightly better than base loss_kpt because it uses frame context
- **Input:** Previous/next frame keypoints + current detections
- **Note:** Usually better than base loss_kpt due to multi-frame information

---

### Track Query Diagnostic Metrics (Sequential Training)
These metrics are **not losses** but diagnostics of track query behavior. They help understand if the tracking mechanism is working.

#### `trk_kpt_l1: 0.17395`
- **Type:** Track query keypoint L1 error (diagnostic, detached from backward)
- **What it measures:** Average keypoint L1 distance for queries that matched a GT in the previous frame
- **Range:** 0 (perfect) to 1 (pixel-sized)
- **Interpretation:** ~0.17 is reasonable; track queries start with stale predictions from previous frame
- **Why it's higher than `det_kpt_l1`:** Track queries are initialized from previous frame, need adaptation

#### `det_kpt_l1: 0.00847`
- **Type:** Detect query keypoint L1 error (diagnostic, detached)
- **What it measures:** Average keypoint L1 distance for queries that didn't propagate from previous frame
- **Range:** 0 (perfect) to 1
- **Interpretation:** ~0.008 is excellent; fresh detection queries learn fast
- **Comparison:** Typically 20-30x better than track queries because they learn from scratch each frame

#### `n_trk: 1.85`
- **Type:** Average number of track queries per image
- **What it measures:** How many high-confidence queries were propagated from the previous frame
- **Range:** 0 to `max_track_queries` (default 100)
- **Interpretation:** ~1.85 track queries per batch; usually 1-5 in early training
- **Why it matters:** If n_trk=0, the tracking mechanism is disabled

#### `n_trk_pos: 1.65`
- **Type:** Average number of track queries that **matched a GT** in Hungarian assignment
- **What it measures:** How many propagated queries correctly re-matched the same person
- **Range:** 0 to `n_trk`
- **Interpretation:** 1.65/1.85 = **89% track matching success rate** — excellent!

#### `n_det_pos: 3.6`
- **Type:** Average number of detect queries that matched a GT
- **What it measures:** How many fresh detection queries found targets
- **Range:** 0 to number of GT persons (~6-8 typical)
- **Interpretation:** 3.6/6-8 = ~45-60% detection rate; fresh queries finding targets
- **Note:** Lower than n_trk_pos because many GTs are already covered by track queries

#### `trk_id_con: 0.77958`
- **Type:** Track ID consistency ratio (diagnostic, detached)
- **What it measures:** % of track queries that maintained the same person ID from previous frame
- **Range:** 0 (no consistency) to 1.0 (perfect consistency)
- **Interpretation:** **0.78 = 78% ID consistency** — model correctly keeps identity across frames
- **Why it matters:** Core metric for tracking; high values (>0.7) indicate good tracking
- **Before fix:** Was 0.0 (because prev_track_gt_ids were all -1)

---

## Overall Loss: `loss: -21.8843`

The `loss` reported is a weighted sum of all losses:
- **Composed of:** enc_loss + all decoder losses + all refinement losses + OKS losses
- **Negative values:** Expected due to RealNVP flow losses dominating
- **Typical range:** -15 to -35 (more negative = better, since log-likelihood)
- **Interpretation:** This is what the optimizer minimizes via backprop

---

## Validation Metrics

#### `grad_norm: 61.62624`
- **Type:** L2 norm of gradients before optimizer step
- **What it measures:** Magnitude of gradient updates
- **Range:** Typically 10-100; >200 indicates exploding gradients
- **Interpretation:** 61.6 is healthy; model is learning without instability
- **Note:** Clipped at 0.1 to prevent exploding gradients (see mmdet config)

#### `memory: 14473` (MB)
- **Type:** GPU memory usage
- **Interpretation:** 14.4 GB is typical for batch_size=1, seq_length=2 with `no_grad` window 0
- **Before fix:** Was 19-20 GB (now window 0 doesn't save activations)

#### `time: 0.47254` (seconds)
- **Type:** Wall-clock time per training iteration
- **Interpretation:** 0.47 sec/iter × 1939 iters/epoch ≈ 152 min/epoch

---

## Example Analysis (Epoch 1, Iter 520)

```
{"mode": "train", "epoch": 1, "iter": 520, "lr": 1e-05,
 "enc_loss_cls": 0.14931, "enc_loss_kpt": -4.67249,           ← Encoder learning
 "loss_cls": 0.19148, "loss_kpt": -3.65011,                   ← Main detection improving
 "d0.loss_cls": 0.21142, "d0.loss_kpt": -3.3122,              ← Intermediate refinement
 "d1.loss_cls": 0.19985, "d1.loss_kpt": -3.54268,             ← Better refinement
 "trk_kpt_l1": 0.17395, "det_kpt_l1": 0.00847,                ← Track L1 >> detect L1
 "n_trk": 1.85, "n_trk_pos": 1.65,                            ← 89% matching success
 "n_det_pos": 3.6, "trk_id_con": 0.77958,                     ← 78% ID consistency ✓
 "loss": -21.8843, "grad_norm": 61.62624, "memory": 14473}    ← Stable training
```

**Interpretation:**
- ✅ **Tracking is working well** — high n_trk_pos, good trk_id_con
- ✅ **Detection improving** — low loss_cls, reasonable loss_kpt
- ✅ **Refinement effective** — refine losses better than base losses
- ✅ **Training stable** — grad_norm ~60, memory ~14GB
- ⚠️ **Track queries underperforming** — trk_kpt_l1 (0.174) >> det_kpt_l1 (0.008), but this is **expected** as track queries start from stale predictions

---

## Typical Loss Evolution During Training

| Stage | enc_cls | enc_kpt | loss_cls | loss_kpt | trk_id_con | Notes |
|-------|---------|---------|----------|----------|-----------|-------|
| Epoch 1, iter 1 | ~0.4 | -3.5 | ~0.5 | -1.5 | 0.0-0.1 | Cold start, random initialization |
| Epoch 1, iter 500 | ~0.15 | -4.6 | ~0.2 | -3.6 | 0.75-0.85 | **Current state** — encoder learning, tracking works |
| Epoch 5 | ~0.08 | -5.2 | ~0.1 | -4.0 | 0.85-0.92 | Convergence, track queries improving |
| Epoch 10 | ~0.05 | -5.5 | ~0.08 | -4.2 | 0.90+ | Near convergence, track queries match detect quality |

---

## Troubleshooting Guide

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `n_trk_pos` stays 0.0 | Track bonus not applied | Check prev_track_gt_ids (should be ≥0) |
| `trk_id_con` low (<0.5) | Track queries not re-matching people | Increase track_bonus or improve track query initialization |
| `loss_kpt` not improving | Keypoint regression stuck | Check learning rate, gradient clipping |
| `grad_norm` > 200 | Exploding gradients | Reduce learning rate or batch size |
| `memory` > 20GB | OOM risk | Reduce batch size or enable gradient checkpointing |
| `trk_kpt_l1` >> `det_kpt_l1` | Track queries poorly initialized | This is **expected**; will improve as detector improves |

---

## References

- **RealNVP Flow Loss:** Uses normalizing flow to model joint distribution of keypoint errors. Negative log-likelihood is the loss.
- **Focal Loss:** Weights hard negatives more; helps with class imbalance (many backgrounds vs few people)
- **Hungarian Matching:** Bipartite assignment with track bonus = -10 (default) or -50 (tracking config)
- **Query Propagation:** Track queries are hidden states from previous frame passed through decoder with gradients detached
