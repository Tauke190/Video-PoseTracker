"""
Convert COCO 17-keypoint annotations to PoseTrack 15-keypoint format.

COCO (17): nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
           left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip,
           left_knee, right_knee, left_ankle, right_ankle

PoseTrack (15): nose, head_bottom, head_top, left_shoulder, right_shoulder,
                left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip,
                left_knee, right_knee, left_ankle, right_ankle

Mapping strategy:
- nose (0) -> nose (0)
- head_bottom (1) <- average of left_ear (3) and right_ear (4), or nose if not visible
- head_top (2) <- average of left_eye (1) and right_eye (2), or head_bottom offset
- left_shoulder (3) <- left_shoulder (5)
- right_shoulder (4) <- right_shoulder (6)
- left_elbow (5) <- left_elbow (7)
- right_elbow (6) <- right_elbow (8)
- left_wrist (7) <- left_wrist (9)
- right_wrist (8) <- right_wrist (10)
- left_hip (9) <- left_hip (11)
- right_hip (10) <- right_hip (12)
- left_knee (11) <- left_knee (13)
- right_knee (12) <- right_knee (14)
- left_ankle (13) <- left_ankle (15)
- right_ankle (14) <- right_ankle (16)
"""

import json
import argparse
import numpy as np
from tqdm import tqdm


def convert_keypoints(coco_kpts):
    """Convert 17 COCO keypoints to 15 PoseTrack keypoints."""
    # coco_kpts is a flat list: [x0, y0, v0, x1, y1, v1, ...]
    coco_kpts = np.array(coco_kpts).reshape(-1, 3)  # (17, 3)

    posetrack_kpts = np.zeros((15, 3))

    # Direct mappings (COCO idx -> PoseTrack idx)
    # nose (0) -> nose (0)
    posetrack_kpts[0] = coco_kpts[0]

    # Shoulders, elbows, wrists (5-10 -> 3-8)
    posetrack_kpts[3] = coco_kpts[5]   # left_shoulder
    posetrack_kpts[4] = coco_kpts[6]   # right_shoulder
    posetrack_kpts[5] = coco_kpts[7]   # left_elbow
    posetrack_kpts[6] = coco_kpts[8]   # right_elbow
    posetrack_kpts[7] = coco_kpts[9]   # left_wrist
    posetrack_kpts[8] = coco_kpts[10]  # right_wrist

    # Hips, knees, ankles (11-16 -> 9-14)
    posetrack_kpts[9] = coco_kpts[11]   # left_hip
    posetrack_kpts[10] = coco_kpts[12]  # right_hip
    posetrack_kpts[11] = coco_kpts[13]  # left_knee
    posetrack_kpts[12] = coco_kpts[14]  # right_knee
    posetrack_kpts[13] = coco_kpts[15]  # left_ankle
    posetrack_kpts[14] = coco_kpts[16]  # right_ankle

    # head_bottom (1) <- average of ears, or use nose
    left_ear = coco_kpts[3]
    right_ear = coco_kpts[4]
    if left_ear[2] > 0 and right_ear[2] > 0:
        posetrack_kpts[1] = [(left_ear[0] + right_ear[0]) / 2,
                             (left_ear[1] + right_ear[1]) / 2,
                             min(left_ear[2], right_ear[2])]
    elif left_ear[2] > 0:
        posetrack_kpts[1] = left_ear
    elif right_ear[2] > 0:
        posetrack_kpts[1] = right_ear
    else:
        posetrack_kpts[1] = coco_kpts[0]  # fallback to nose

    # head_top (2) <- average of eyes, or offset from head_bottom
    left_eye = coco_kpts[1]
    right_eye = coco_kpts[2]
    if left_eye[2] > 0 and right_eye[2] > 0:
        posetrack_kpts[2] = [(left_eye[0] + right_eye[0]) / 2,
                             (left_eye[1] + right_eye[1]) / 2,
                             min(left_eye[2], right_eye[2])]
    elif left_eye[2] > 0:
        posetrack_kpts[2] = left_eye
    elif right_eye[2] > 0:
        posetrack_kpts[2] = right_eye
    elif posetrack_kpts[1][2] > 0:
        # Estimate head_top above head_bottom
        posetrack_kpts[2] = [posetrack_kpts[1][0],
                             posetrack_kpts[1][1] - 20,  # 20 pixels above
                             posetrack_kpts[1][2]]
    else:
        posetrack_kpts[2] = [0, 0, 0]

    return posetrack_kpts.flatten().tolist()


def main():
    parser = argparse.ArgumentParser(description='Convert COCO to PoseTrack keypoints')
    parser.add_argument('input', help='Input COCO annotation JSON file')
    parser.add_argument('output', help='Output PoseTrack annotation JSON file')
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    with open(args.input) as f:
        data = json.load(f)

    # Update category info
    for cat in data.get('categories', []):
        if cat.get('name') == 'person':
            cat['num_keypoints'] = 15
            cat['keypoints'] = [
                'nose', 'head_bottom', 'head_top',
                'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist',
                'left_hip', 'right_hip',
                'left_knee', 'right_knee',
                'left_ankle', 'right_ankle'
            ]
            cat['skeleton'] = [
                [0, 1], [1, 2],  # nose -> head_bottom -> head_top
                [0, 3], [0, 4],  # nose -> shoulders
                [3, 5], [5, 7],  # left arm
                [4, 6], [6, 8],  # right arm
                [3, 9], [4, 10],  # shoulders -> hips
                [9, 11], [11, 13],  # left leg
                [10, 12], [12, 14]  # right leg
            ]

    # Convert annotations
    print("Converting keypoints...")
    for ann in tqdm(data.get('annotations', [])):
        if 'keypoints' in ann and len(ann['keypoints']) == 51:  # 17 * 3
            ann['keypoints'] = convert_keypoints(ann['keypoints'])
            # Update num_keypoints count (visible keypoints)
            kpts = np.array(ann['keypoints']).reshape(-1, 3)
            ann['num_keypoints'] = int(np.sum(kpts[:, 2] > 0))

    print(f"Saving to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(data, f)

    print("Done!")


if __name__ == '__main__':
    main()