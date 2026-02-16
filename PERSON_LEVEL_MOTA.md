# Person-Level MOTA Metrics

## What Changed

The evaluation pipeline now computes **person-level MOTA** in addition to the standard joint-level MOTA. This provides much clearer metrics for tracking quality.

## Joint-Level vs Person-Level

### Joint-Level MOTA (Original)
- **1 person = 15 objects** (one per joint)
- Each joint is tracked independently
- Metrics are per-joint, then averaged
- **Problem**: A single unmatched person generates 15 false positives, amplifying errors by 15×

### Person-Level MOTA (New)
- **1 person = 1 object**
- Entire person is matched or unmatched
- Much more intuitive and interpretable
- **Benefit**: Reflects actual tracking quality

## Example

Suppose a frame has:
- 5 GT persons
- 8 predicted persons (3 are noise)
- 4 predictions match GT persons
- 1 person changed ID

### Joint-Level Breakdown:
- GT Objects: 5 × 15 = 75 joints
- Matched: 4 × 15 = 60 joints
- FP: 3 × 15 = 45 joints (the 3 spurious people)
- Misses: 1 × 15 = 15 joints
- ID Switches: 1 × 15 = 15 joints (attributed to all joints)

**MOTA = 1 - (45 + 15 + 15) / 75 = -13.3%** ← Looks terrible!

### Person-Level Breakdown:
- GT Persons: 5
- Detections: 4
- FP: 3
- Misses: 1
- ID Switches: 1

**MOTA = 1 - (3 + 1 + 1) / 5 = 0%** ← More honest!

## Output Format

After evaluation, you'll see:

```
================================================================================
PERSON-LEVEL TRACKING METRICS (Poses, not Joints)
================================================================================
Total GT Persons:         N
Correctly Detected:       M
Misses (False Negatives): M_miss
False Positives:          N_fp
ID Switches:              N_id

Person-level MOTA:        X.XX%
Precision:                Y.YY%
Recall:                   Z.ZZ%
================================================================================
```

## Metric Definitions

- **Total GT Persons**: Sum of all unique persons across all frames in ground truth
- **Correctly Detected**: Persons that matched a GT person at least once
- **Misses**: GT persons with no detection
- **False Positives**: Predicted persons with no GT match
- **ID Switches**: Times a matched GT person's ID changes between frames
- **MOTA**: `1 - (FP + Misses + ID_Switches) / GT_Persons`
- **Precision**: `Detections / (Detections + FP)`
- **Recall**: `Detections / GT_Persons`

## When to Use

- **Joint-level MOTA**: Fine-grained analysis of which body parts track well
- **Person-level MOTA**: Overall tracking quality assessment

The person-level metrics are usually much higher and more intuitive than joint-level metrics, especially when filtering by score threshold.

## Related Parameters

The `score_thr` parameter in evaluation filters predictions at the **person level**:
- Only persons with bbox confidence ≥ `score_thr` are included
- Default: 0.3
- Usage: `python tools/test.py config.py ckpt.pth --eval keypoints --eval-options score_thr=0.4`

The person-level MOTA is computed on **filtered predictions**, so it reflects real-world performance with the confidence threshold applied.