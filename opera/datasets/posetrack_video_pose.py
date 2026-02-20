# Copyright (c) OpenMMLab. All rights reserved.  
import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from mmdet.datasets.api_wrappers import COCO, COCOeval
from .builder import DATASETS
from mmdet.datasets import CustomDataset
import os
import scipy.io as sio
import json
import torch

# 更新时间   ---- 2024-8-02
@DATASETS.register_module()
class PosetrackVideoPoseDataset(CustomDataset):
    """
        PoseTrack Dataset
    """
    """
    PoseTrack keypoint indexes::
        0: 'nose',
        1: 'head_bottom',
        2: 'head_top',
        3: 'left_shoulder',
        4: 'right_shoulder',
        5: 'left_elbow',
        6: 'right_elbow',
        7: 'left_wrist',
        8: 'right_wrist',
        9: 'left_hip',
        10: 'right_hip',
        11: 'left_knee',
        12: 'right_knee',
        13: 'left_ankle',
        14: 'right_ankle'
    """

    CLASSES = ('person', )

    FLIP_PAIRS = [[3, 4],
                [5, 6],
                [7, 8],
                [9, 10],
                [11, 12],
                [13, 14]]

    
    def __init__(self,
                 *args,
                 skip_invaild_pose=True,
                 **kwargs):
        super(PosetrackVideoPoseDataset, self).__init__(*args, **kwargs)
        self.skip_invaild_pose = skip_invaild_pose
    
    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        logger = logging.getLogger(__file__)
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler() # 添加处理程序（这里使用 StreamHandler 输出到控制台）
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)  # 获取标签对应的标签id
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids() # 获取到posetrack_train or val json中的所有图片 7700左右
        # NOTE: Debug line below was limiting test to 1 image - now commented out
        # if self.test_mode:
        #     self.img_ids = self.img_ids[1193:1194]
        # self.img_ids = [] # 存放过滤后的所有图片 --- 只包含标签中有标注信息的图片 2607张
        data_infos = self._get_data() # 获取到所有标注图片的基本信息通过img_ids
        # for id, data in enumerate(data_infos):
        #     if 'bonn_mpii_test_5sec/16236_mpii' in data['file_name']:
        #         print(id)
        #         print(data['file_name'])

        logger.info(f"本次加载{'测试' if self.test_mode else '训练'}数据样本个数为：{len(data_infos)}")
        return data_infos
    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['keypoint_fields'] = []
        results['area_fields'] = []

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
            else:
                a = img_info
        self.img_ids = valid_img_ids
        print(f"本次{'训练' if not self.test_mode else '测试'}合格的样本数共{len(self.img_ids)}个")
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_keypoints = []
        gt_areas = []
        gt_track_ids = []
        w = img_info['width']
        h = img_info['height']
        
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            # skip invalid pose annotation
            if ann['num_keypoints'] == 0 and self.skip_invaild_pose:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                gt_keypoints.append(ann.get('keypoints', None))
                gt_areas.append(ann.get('area', None))
                gt_track_ids.append(ann.get('track_id', -1))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_bboxes[:, 0::2].clip(min=0, max=w)
            gt_bboxes[:, 1::2].clip(min=0, max=h)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_keypoints = np.array(gt_keypoints, dtype=np.float32)
            gt_areas = np.array(gt_areas, dtype=np.float32)
            gt_track_ids = np.array(gt_track_ids, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_keypoints = np.zeros((0, 45), dtype=np.float32)
            gt_areas = np.array([], dtype=np.float32)
            gt_track_ids = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        # no use
        # seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            keypoints=gt_keypoints,
            areas=gt_areas,
            track_ids=gt_track_ids,
            flip_pairs=self.FLIP_PAIRS)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _prepare_video_results(self, coco_dt, results):
        """Extract detections from COCO results and group by video/frame.

        Args:
            coco_dt: COCO detection results object
            results: Raw detection results list

        Returns:
            dict: {video_name: [(frame_num, [(bbox, kpts), ...]), ...]}
        """
        from collections import defaultdict

        results_by_video = defaultdict(list)

        # Build a mapping from image_id to detection results
        img_id_to_results_idx = {}
        for idx in range(len(self.data_infos)):
            img_id_to_results_idx[self.data_infos[idx]['id']] = idx

        # Extract detections for each image
        for img_id in coco_dt.getImgIds():
            img_info = self.coco.loadImgs(img_id)[0]
            file_name = img_info['file_name']

            # Extract video name and frame number
            video_name = os.path.dirname(file_name)
            if video_name.startswith('images/'):
                video_name = video_name[7:]

            frame_name = os.path.basename(file_name)
            try:
                frame_num = int(os.path.splitext(frame_name)[0])
            except ValueError:
                frame_num = 0

            # Get detection annotations
            ann_ids = coco_dt.getAnnIds(imgIds=img_id)
            anns = coco_dt.loadAnns(ann_ids)

            frame_detections = []
            for ann in anns:
                # Extract bbox (convert COCO format [x, y, w, h] to [x1, y1, x2, y2])
                bbox = ann.get('bbox', [0, 0, 0, 0])
                x, y, w, h = bbox
                x1, y1, x2, y2 = x, y, x + w, y + h
                score = ann.get('score', 0.0)
                bbox_full = np.array([x1, y1, x2, y2, score], dtype=np.float32)

                # Extract keypoints
                keypoints = ann.get('keypoints', [])
                kpts = np.array(keypoints).reshape(-1, 3).astype(np.float32)

                frame_detections.append((bbox_full, kpts))

            # Group by video
            results_by_video[video_name].append((frame_num, frame_detections))

        return dict(results_by_video)

    def _assign_tracking_ids(self, results_by_video):
        """Assign persistent tracking IDs across frames using Hungarian matching.

        Args:
            results_by_video (dict): Results grouped by video name.
                Format: {video_name: [(frame_num, [(bbox, kpts), ...]), ...]}

        Returns:
            dict: Same structure but with track_ids assigned.
                  Format: {video_name: [(frame_num, [track_id, ...]), ...]}
        """
        from opera.core.keypoint.transforms import match_detections_hungarian
        from mmcv.utils import print_log

        tracking_results = {}

        for video_name, frames_data in results_by_video.items():
            # Sort by frame number
            frames_data = sorted(frames_data, key=lambda x: x[0])

            # Initialize tracking
            frame_track_ids = []  # [(frame_num, [track_ids for each detection])]
            next_track_id = 0

            for frame_idx, (frame_num, detections) in enumerate(frames_data):
                if frame_idx == 0:
                    # First frame: assign sequential IDs
                    track_ids = list(range(len(detections)))
                    next_track_id = len(detections)
                else:
                    # Match with previous frame
                    prev_frame_num, prev_detections = frames_data[frame_idx - 1]
                    prev_track_ids = frame_track_ids[-1][1]

                    # Compute matches
                    matches = match_detections_hungarian(
                        detections, prev_detections,
                        max_distance=0.5, alpha=0.3
                    )

                    track_ids = [-1] * len(detections)

                    if matches is not None:
                        for curr_idx, prev_idx in matches:
                            # Reuse ID from previous frame
                            track_ids[curr_idx] = prev_track_ids[prev_idx]

                    # Assign new IDs to unmatched detections
                    for i in range(len(detections)):
                        if track_ids[i] == -1:
                            track_ids[i] = next_track_id
                            next_track_id += 1

                frame_track_ids.append((frame_num, track_ids))

            tracking_results[video_name] = frame_track_ids

        return tracking_results

    def _build_model_tracking_ids(self, results, score_thr=0.0):
        """Build tracking IDs dict from model-provided track IDs.

        Instead of running post-processing Hungarian matching, this uses
        the track IDs assigned by the model's query propagation mechanism.

        Args:
            results: Detection results with model track IDs.
                Each result is a 3-tuple (det, kpt, track_ids).
            score_thr (float): Score threshold to filter detections.

        Returns:
            dict: {video_name: [(frame_num, [track_id, ...]), ...]}
        """
        from collections import defaultdict

        tracking_results = defaultdict(list)

        for idx in range(len(self.data_infos)):
            img_info = self.data_infos[idx]
            file_name = img_info['file_name']

            video_name = os.path.dirname(file_name)
            if video_name.startswith('images/'):
                video_name = video_name[7:]

            frame_name = os.path.basename(file_name)
            try:
                frame_num = int(os.path.splitext(frame_name)[0])
            except ValueError:
                frame_num = 0

            result_i = results[idx]
            if len(result_i) < 3 or result_i[2] is None:
                # No model track IDs, skip
                continue

            det, kpt, tid = result_i[0], result_i[1], result_i[2]
            # Collect track IDs for all detections (across classes)
            frame_track_ids = []
            for label in range(len(det)):
                bboxes = det[label]
                tids = tid[label]
                for i in range(bboxes.shape[0]):
                    if bboxes[i][4] < score_thr:
                        continue
                    frame_track_ids.append(int(tids[i]))

            tracking_results[video_name].append(
                (frame_num, frame_track_ids))

        # Sort frames within each video
        for video_name in tracking_results:
            tracking_results[video_name].sort(key=lambda x: x[0])

        # Remap raw track IDs to compact 0-based IDs per video
        # (PoseTrack evaluator requires track_id < 10000)
        for video_name in tracking_results:
            raw_ids = set()
            for frame_num, tids in tracking_results[video_name]:
                raw_ids.update(tids)
            id_map = {raw_id: new_id
                      for new_id, raw_id in enumerate(sorted(raw_ids))}
            tracking_results[video_name] = [
                (frame_num, [id_map[tid] for tid in tids])
                for frame_num, tids in tracking_results[video_name]
            ]

        return dict(tracking_results)

    def _build_posetrack_output(self, results, coco_dt, tracking_ids=None,
                                joint_score_thr=0.0):
        """Build PoseTrack format output from predictions.

        Args:
            results: Detection results from the model
            coco_dt: COCO object with detection results loaded
            tracking_ids (dict, optional): Pre-computed persistent tracking IDs.
                Format: {video_name: [(frame_num, [track_id, ...]), ...]}
            joint_score_thr (float): Per-joint confidence threshold. Joints
                with score below this are omitted from the output, reducing
                false positives for unannotated/occluded joints. Default: 0.0.

        Returns:
            dict: Output data organized by video name
        """
        from collections import defaultdict

        out_data = defaultdict(list)

        # Build a lookup for tracking IDs if provided
        tracking_id_lookup = {}
        if tracking_ids is not None:
            for video_name, frame_data in tracking_ids.items():
                if video_name not in tracking_id_lookup:
                    tracking_id_lookup[video_name] = {}
                for frame_num, track_ids in frame_data:
                    tracking_id_lookup[video_name][frame_num] = track_ids

        # Get all predictions from coco_dt
        for img_id in coco_dt.getImgIds():
            # Get image info
            img_info = self.coco.loadImgs(img_id)[0]
            file_name = img_info['file_name']

            # Extract video name (directory path)
            video_name = os.path.dirname(file_name)
            if video_name.startswith('images/'):
                video_name = video_name[7:]  # Remove 'images/' prefix

            # Extract frame number from filename
            frame_name = os.path.basename(file_name)
            try:
                frame_num = int(os.path.splitext(frame_name)[0])
            except ValueError:
                frame_num = 0

            # Get annotations for this image
            ann_ids = coco_dt.getAnnIds(imgIds=img_id)
            anns = coco_dt.loadAnns(ann_ids)

            # Get pre-computed tracking IDs if available
            video_track_ids = None
            if video_name in tracking_id_lookup and frame_num in tracking_id_lookup[video_name]:
                video_track_ids = tracking_id_lookup[video_name][frame_num]

            annorect = []
            for det_idx, ann in enumerate(anns):
                keypoints = ann.get('keypoints', [])
                score = ann.get('score', 0.0)

                # Use pre-computed tracking ID if available, otherwise use sequential index
                if video_track_ids is not None and det_idx < len(video_track_ids):
                    track_id = video_track_ids[det_idx]
                else:
                    track_id = det_idx

                # Convert keypoints to PoseTrack evaluation format.
                # Model output order (15 kpts):
                #   0:nose, 1:head_bottom, 2:head_top, 3:l_shoulder,
                #   4:r_shoulder, 5:l_elbow, 6:r_elbow, 7:l_wrist,
                #   8:r_wrist, 9:l_hip, 10:r_hip, 11:l_knee,
                #   12:r_knee, 13:l_ankle, 14:r_ankle
                # Evaluation expects MPII/LSP IDs (see Joint class):
                #   0:r_ankle, 1:r_knee, 2:r_hip, 3:l_hip, 4:l_knee,
                #   5:l_ankle, 6:r_wrist, 7:r_elbow, 8:r_shoulder,
                #   9:l_shoulder, 10:l_elbow, 11:l_wrist, 12:neck,
                #   13:nose, 14:head_top
                MODEL_TO_EVAL_ID = [13, 12, 14, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]
                points = []
                kpts = np.array(keypoints).reshape(-1, 3)
                for kpt_id, kpt in enumerate(kpts):
                    if kpt[2] < joint_score_thr:
                        continue
                    points.append({
                        'id': [MODEL_TO_EVAL_ID[kpt_id]],
                        'x': [float(kpt[0])],
                        'y': [float(kpt[1])],
                        'score': [float(kpt[2])]
                    })

                annorect.append({
                    'annopoints': [{'point': points}],
                    'score': [float(score)],
                    'track_id': [track_id]
                })

            frame_data = {
                'image': {'name': file_name},
                'imgnum': [frame_num],
                'img_num': [frame_num],
                'annorect': annorect
            }
            out_data[video_name].append(frame_data)

        # Sort frames within each video by frame number
        for video_name in out_data:
            out_data[video_name].sort(key=lambda x: x['imgnum'][0])

        return dict(out_data)

    def _kpt2json(self, results, score_thr=0.0):
        """Convert keypoint detection results to COCO json style.

        Args:
            results: Detection results from the model.
            score_thr (float): Score threshold to filter low-confidence
                detections. Detections with score < score_thr are discarded.
                Default: 0.0 (no filtering).
        """
        bbox_json_results = []
        kpt_json_results = []
        for idx in range(len(self)):
            img_id = self.data_infos[idx]['id']
            result_i = results[idx]
            det, kpt = result_i[0], result_i[1]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                kpts = kpt[label]
                for i in range(bboxes.shape[0]):
                    if bboxes[i][4] < score_thr:
                        continue
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                    # kpt results
                    data = dict()
                    data['image_id'] = img_id
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    i_kpt = kpts[i].reshape(-1)
                    data['keypoints'] = i_kpt.tolist()
                    kpt_json_results.append(data)
        return bbox_json_results, kpt_json_results

    def results2json(self, results, outfile_prefix, score_thr=0.0):
        """Dump the detection results to a COCO style json file.

        There are 4 types of results: proposals, bbox predictions, mask
        predictions, keypoint_predictions, and they have different data types.
        This method will automatically recognize the type, and dump them to
        json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".
            score_thr (float): Score threshold to filter low-confidence
                detections. Default: 0.0 (no filtering).

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            # Check for keypoint results: kpt arrays are 3D (n, K, 3).
            # results[0][1] is always the kpt result list (2nd element).
            if isinstance(results[0][1][0],
                          np.ndarray) and results[0][1][0].ndim == 3:
                json_results = self._kpt2json(results, score_thr=score_thr)
                result_files['bbox'] = f'{outfile_prefix}.bbox.json'
                result_files['proposal'] = f'{outfile_prefix}.bbox.json'
                result_files['keypoints'] = f'{outfile_prefix}.keypoints.json'
                # mmcv.dump(json_results[0], result_files['bbox'])
                mmcv.dump(json_results[1], result_files['keypoints'])
            else:
                json_results = self._segm2json(results)
                result_files['bbox'] = f'{outfile_prefix}.bbox.json'
                result_files['proposal'] = f'{outfile_prefix}.bbox.json'
                result_files['segm'] = f'{outfile_prefix}.segm.json'
                # mmcv.dump(json_results[0], result_files['bbox'])
                mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='keypoints',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 save_dir=None,
                 score_thr=0.3,
                 joint_score_thr=0.1,
                 subset_eval=False):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'keypoints', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.
            score_thr (float): Score threshold to filter low-confidence
                detections before tracking (MOTA) evaluation. AP evaluation
                uses all predictions unfiltered. Default: 0.3.
            joint_score_thr (float): Per-joint confidence threshold for a
                second MOTA evaluation. Joints with score below this are
                omitted, reducing FPs from unannotated/occluded joints.
                Default: 0.1.
            subset_eval (bool): If True, use only COCO evaluation for fast
                subset validation. Skips full PoseTrack evaluation which
                requires external GT files. Default: False.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        # Use default save_dir if not specified
        if save_dir is None:
            save_dir = 'work_dirs/test'

        from ..core.posetrack_utils.poseval.py.evaluateAP import evaluateAP
        from ..core.posetrack_utils.poseval.py.evaluateTracking import \
            evaluateTracking
        from ..core.posetrack_utils.poseval.py.eval_helpers import (
            Joint, printTable, load_data_dir)

        # --- Step 1: Unfiltered results for COCO AP ---
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        # Suppress pycocotools verbose output
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = self.coco.loadRes(result_files['keypoints'])
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.params.maxDets = [20]
        coco_eval.params.imgIds = list(coco_dt.imgToAnns.keys())
        coco_eval.params.kpt_oks_sigmas = np.array([
            0.026, 0.025, 0.025, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
        ])
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.evaluate()

        # --- Subset evaluation: per-joint AP + MOTA on filtered GT ---
        if subset_eval:
            # Build set of subset image names (with images/ prefix for GT)
            subset_images = set()
            for info in self.data_infos:
                subset_images.add('images/' + info['file_name'])

            annot_dir = 'DcPose_supp_files/posetrack17_annotation_dirs/jsons/val/'

            # Build tracking IDs and filtered predictions
            has_model_track_ids = (len(results) > 0 and len(results[0]) >= 3
                                   and results[0][2] is not None)
            if has_model_track_ids:
                tracking_ids_filt = self._build_model_tracking_ids(
                    results, score_thr=score_thr)
            else:
                filtered_prefix = osp.join(
                    tempfile.mkdtemp(), 'results_filtered_subset')
                filtered_files = self.results2json(
                    results, filtered_prefix, score_thr=score_thr)
                with contextlib.redirect_stdout(io.StringIO()):
                    coco_dt_filt = self.coco.loadRes(filtered_files['keypoints'])
                results_by_video_filt = self._prepare_video_results(
                    coco_dt_filt, results)
                tracking_ids_filt = self._assign_tracking_ids(
                    results_by_video_filt)

            # Build filtered predictions (score-thresholded)
            if has_model_track_ids:
                filtered_prefix = osp.join(
                    tempfile.mkdtemp(), 'results_filtered_subset')
                filtered_files = self.results2json(
                    results, filtered_prefix, score_thr=score_thr)
                with contextlib.redirect_stdout(io.StringIO()):
                    coco_dt_filt = self.coco.loadRes(filtered_files['keypoints'])

            # Also build UNfiltered predictions for AP evaluation
            tracking_ids_all = (
                self._build_model_tracking_ids(results, score_thr=0.0)
                if has_model_track_ids
                else self._assign_tracking_ids(
                    self._prepare_video_results(coco_dt, results))
            )
            out_data_ap = self._build_posetrack_output(
                results, coco_dt, tracking_ids_all)
            out_data_track = self._build_posetrack_output(
                results, coco_dt_filt, tracking_ids_filt)

            # Create temp dirs for subset-filtered GT and predictions
            subset_gt_dir = os.path.join(tempfile.mkdtemp(), 'subset_gt')
            subset_pred_dir = os.path.join(tempfile.mkdtemp(), 'subset_pred')
            subset_ap_dir = os.path.join(tempfile.mkdtemp(), 'subset_ap')
            os.makedirs(subset_gt_dir)
            os.makedirs(subset_pred_dir)
            os.makedirs(subset_ap_dir)

            # For each GT video, filter to only subset frames
            for gt_fname in os.listdir(annot_dir):
                if not gt_fname.endswith('.json'):
                    continue
                with open(os.path.join(annot_dir, gt_fname)) as f:
                    gt_data = json.load(f)
                if 'annolist' not in gt_data:
                    continue
                gt_frames = gt_data['annolist']

                # Filter GT to subset frames only
                filtered_gt = [
                    fr for fr in gt_frames
                    if fr['image'][0]['name'] in subset_images
                ]
                if len(filtered_gt) < 2:
                    continue  # Need >= 2 frames for MOTA (last frame is dropped)

                # Get video name key (without images/ prefix)
                first_img = filtered_gt[0]['image'][0]['name']
                vname_key = os.path.dirname(first_img)
                if vname_key.startswith('images/'):
                    vname_key = vname_key[len('images/'):]

                # Build matching prediction frames for MOTA (filtered)
                pred_frames = out_data_track.get(vname_key, [])
                pred_by_name = {}
                for pf in pred_frames:
                    pred_by_name[pf['image']['name']] = pf

                matched_preds = []
                for gt_fr in filtered_gt:
                    gt_img = gt_fr['image'][0]['name']
                    pred_img = gt_img
                    if pred_img.startswith('images/'):
                        pred_img = pred_img[len('images/'):]
                    if pred_img in pred_by_name:
                        matched_preds.append(pred_by_name[pred_img])
                    else:
                        matched_preds.append({
                            'image': {'name': pred_img},
                            'imgnum': gt_fr.get('imgnum', [0]),
                            'annorect': []
                        })

                # Build matching prediction frames for AP (unfiltered)
                ap_frames = out_data_ap.get(vname_key, [])
                ap_by_name = {}
                for af in ap_frames:
                    ap_by_name[af['image']['name']] = af

                matched_ap = []
                for gt_fr in filtered_gt:
                    gt_img = gt_fr['image'][0]['name']
                    pred_img = gt_img
                    if pred_img.startswith('images/'):
                        pred_img = pred_img[len('images/'):]
                    if pred_img in ap_by_name:
                        matched_ap.append(ap_by_name[pred_img])
                    else:
                        matched_ap.append({
                            'image': {'name': pred_img},
                            'imgnum': gt_fr.get('imgnum', [0]),
                            'annorect': []
                        })

                assert len(matched_preds) == len(filtered_gt)
                assert len(matched_ap) == len(filtered_gt)

                write_json_to_file(
                    {'annolist': filtered_gt},
                    os.path.join(subset_gt_dir, gt_fname))
                write_json_to_file(
                    {'annolist': matched_preds},
                    os.path.join(subset_pred_dir, gt_fname))
                write_json_to_file(
                    {'annolist': matched_ap},
                    os.path.join(subset_ap_dir, gt_fname))

            # Run per-joint AP evaluation on subset
            print_log("=> Running AP evaluation (subset)")
            gtFramesAP, prFramesAP = load_data_dir(
                ['', subset_gt_dir, subset_ap_dir])
            apAll, _, _ = evaluateAP(gtFramesAP, prFramesAP)
            ap_cum = printTable(apAll)

            # Run MOTA evaluation on subset-filtered data
            print_log("=> Running tracking evaluation (subset)")
            gtFramesAll, prFramesAll = load_data_dir(
                ['', subset_gt_dir, subset_pred_dir])
            metricsAll = evaluateTracking(gtFramesAll, prFramesAll, False)

            nJoints = Joint().count
            person_metrics = self.compute_person_level_mota(
                gtFramesAll, prFramesAll, distThresh=0.5)

            summary_str = (
                f"=> Validation Summary (Subset): "
                f"mAP={ap_cum[7]:.2f} | "
                f"Person_MOTA={person_metrics['mota']:.2f}% | "
                f"Precision={person_metrics['precision']:.2f}% | "
                f"Recall={person_metrics['recall']:.2f}%"
            )
            print_log(summary_str)

            name_value = OrderedDict([
                ('Head', ap_cum[0]),
                ('Shoulder', ap_cum[1]),
                ('Elbow', ap_cum[2]),
                ('Wrist', ap_cum[3]),
                ('Hip', ap_cum[4]),
                ('Knee', ap_cum[5]),
                ('Ankle', ap_cum[6]),
                ('Mean', ap_cum[7]),
                ('Person_MOTA', person_metrics['mota']),
                ('Person_Precision', person_metrics['precision']),
                ('Person_Recall', person_metrics['recall']),
            ])
            return name_value, ap_cum[7]

        annot_dir = 'DcPose_supp_files/posetrack17_annotation_dirs/jsons/val/'
        # annot_dir = '/root/autodl-tmp/datasets/posetrack18/posetrack18_annotation_dirs/val/'
        out_filenames, L = self.video2filenames(annot_dir)

        # --- Step 2: Unfiltered PoseTrack output for AP evaluation ---
        # Check if results contain model-assigned track IDs (3-tuple)
        has_model_track_ids = (len(results) > 0 and len(results[0]) >= 3
                               and results[0][2] is not None)
        if has_model_track_ids:
            print_log("=> Using model-assigned track IDs (query propagation)")
            tracking_ids_all = self._build_model_tracking_ids(
                results, score_thr=0.0)
        else:
            print_log("=> Using post-processing Hungarian matching for IDs")
            results_by_video_all = self._prepare_video_results(
                coco_dt, results)
            tracking_ids_all = self._assign_tracking_ids(results_by_video_all)
        print_log("=> Building unfiltered predictions for AP evaluation")
        out_data_ap = self._build_posetrack_output(
            results, coco_dt, tracking_ids_all)
        ap_output_dir = os.path.join(save_dir, 'val_set_json_results_ap')
        self.create_folder(ap_output_dir)
        out_data_ap = self._fill_missing_frames(
            out_data_ap, L, out_filenames)
        for vname in out_data_ap.keys():
            vdata = out_data_ap[vname]
            outfpath = os.path.join(
                ap_output_dir,
                out_filenames[os.path.join('images', vname)])  # posetrack17
            # outfpath = os.path.join(ap_output_dir, out_filenames[vname])  # posetrack18
            write_json_to_file({'annolist': vdata}, outfpath)

        # Run AP evaluation on unfiltered predictions
        print_log("=> Running AP evaluation (unfiltered predictions)")
        gtFramesAll_ap, prFramesAll_ap = load_data_dir(
            ['', annot_dir, ap_output_dir])
        apAll, _, _ = evaluateAP(gtFramesAll_ap, prFramesAll_ap)
        ap_cum = printTable(apAll)

        # --- Step 3: Filtered PoseTrack output for MOTA tracking ---
        print_log(f'=> Building filtered predictions for tracking '
                  f'(score_thr={score_thr})')
        # Write filtered results to a separate JSON
        filtered_prefix = osp.join(
            tempfile.mkdtemp(), 'results_filtered')
        filtered_files = self.results2json(
            results, filtered_prefix, score_thr=score_thr)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt_filtered = self.coco.loadRes(filtered_files['keypoints'])

        if has_model_track_ids:
            tracking_ids_filt = self._build_model_tracking_ids(
                results, score_thr=score_thr)
        else:
            results_by_video_filt = self._prepare_video_results(
                coco_dt_filtered, results)
            tracking_ids_filt = self._assign_tracking_ids(
                results_by_video_filt)
        print_log("=> Assigned tracking IDs across frames")
        out_data_track = self._build_posetrack_output(
            results, coco_dt_filtered, tracking_ids_filt)

        track_output_dir = os.path.join(save_dir, 'val_set_json_results')
        self.create_folder(track_output_dir)
        out_data_track = self._fill_missing_frames(
            out_data_track, L, out_filenames)
        print_log("=> Saving files for tracking evaluation")
        for vname in out_data_track.keys():
            vdata = out_data_track[vname]
            outfpath = os.path.join(
                track_output_dir,
                out_filenames[os.path.join('images', vname)])  # posetrack17
            # outfpath = os.path.join(track_output_dir, out_filenames[vname])  # posetrack18
            write_json_to_file({'annolist': vdata}, outfpath)

        # Run MOTA tracking evaluation on filtered predictions
        print_log("=> Running tracking evaluation (filtered predictions)")
        gtFramesAll_trk, prFramesAll_trk = load_data_dir(
            ['', annot_dir, track_output_dir])
        metricsAll = evaluateTracking(
            gtFramesAll_trk, prFramesAll_trk, False)

        nJoints = Joint().count
        metrics = np.full((nJoints + 4, 1), np.nan)
        for i in range(nJoints + 1):
            metrics[i, 0] = metricsAll['mota'][0, i]
        metrics[nJoints + 1, 0] = metricsAll['motp'][0, nJoints]
        metrics[nJoints + 2, 0] = metricsAll['pre'][0, nJoints]
        metrics[nJoints + 3, 0] = metricsAll['rec'][0, nJoints]

        # Store unfiltered metrics for comparison
        joint_mota_unfiltered = metricsAll['mota'][0, nJoints]
        motp_unfiltered = metricsAll['motp'][0, nJoints]
        pre_unfiltered = metricsAll['pre'][0, nJoints]
        rec_unfiltered = metricsAll['rec'][0, nJoints]

        # --- Compute person-level MOTA for better interpretability ---
        person_metrics = self.compute_person_level_mota(
            gtFramesAll_trk, prFramesAll_trk, distThresh=0.5)

        # Log person-level MOTA to wandb
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'person_mota': person_metrics['mota'],
                    'person_precision': person_metrics['precision'],
                    'person_recall': person_metrics['recall'],
                    'person_id_switches': person_metrics['id_switches'],
                    'person_fp': person_metrics['fp'],
                    'person_misses': person_metrics['misses'],
                })
        except ImportError:
            pass

        # --- Step 4: Joint-filtered MOTA tracking (second evaluation) ---
        if joint_score_thr > 0.0:
            print_log(f'=> Building joint-filtered predictions for tracking '
                      f'(person score_thr={score_thr}, '
                      f'joint_score_thr={joint_score_thr})')
            out_data_jfilt = self._build_posetrack_output(
                results, coco_dt_filtered, tracking_ids_filt,
                joint_score_thr=joint_score_thr)
            jfilt_output_dir = os.path.join(
                save_dir, 'val_set_json_results_joint_filtered')
            self.create_folder(jfilt_output_dir)
            out_data_jfilt = self._fill_missing_frames(
                out_data_jfilt, L, out_filenames)
            for vname in out_data_jfilt.keys():
                vdata = out_data_jfilt[vname]
                outfpath = os.path.join(
                    jfilt_output_dir,
                    out_filenames[os.path.join('images', vname)])
                write_json_to_file({'annolist': vdata}, outfpath)

            print_log(f'=> Running joint-filtered tracking evaluation '
                      f'(joint_score_thr={joint_score_thr})')
            gtFramesAll_jf, prFramesAll_jf = load_data_dir(
                ['', annot_dir, jfilt_output_dir])
            metricsAll_jf = evaluateTracking(
                gtFramesAll_jf, prFramesAll_jf, False)

            # Store filtered metrics for comparison
            joint_mota_filtered = metricsAll_jf['mota'][0, nJoints]
            motp_filtered = metricsAll_jf['motp'][0, nJoints]
            pre_filtered = metricsAll_jf['pre'][0, nJoints]
            rec_filtered = metricsAll_jf['rec'][0, nJoints]

            person_metrics_jf = self.compute_person_level_mota(
                gtFramesAll_jf, prFramesAll_jf, distThresh=0.5)
        else:
            # No joint filtering
            joint_mota_filtered = None
            motp_filtered = None
            pre_filtered = None
            rec_filtered = None
            person_metrics_jf = None

        # --- Save detailed validation metrics to file ---
        metrics_dict = {
            'joint_mota_unfiltered': float(joint_mota_unfiltered),
            'joint_mota_filtered': float(joint_mota_filtered) if joint_mota_filtered is not None else None,
            'motp_unfiltered': float(motp_unfiltered),
            'motp_filtered': float(motp_filtered) if motp_filtered is not None else None,
            'pre_unfiltered': float(pre_unfiltered),
            'pre_filtered': float(pre_filtered) if pre_filtered is not None else None,
            'rec_unfiltered': float(rec_unfiltered),
            'rec_filtered': float(rec_filtered) if rec_filtered is not None else None,
            'person_metrics': person_metrics,
            'person_metrics_jf': person_metrics_jf,
            'ap_results': {
                'Head': float(ap_cum[0]),
                'Shoulder': float(ap_cum[1]),
                'Elbow': float(ap_cum[2]),
                'Wrist': float(ap_cum[3]),
                'Hip': float(ap_cum[4]),
                'Knee': float(ap_cum[5]),
                'Ankle': float(ap_cum[6]),
                'Mean': float(ap_cum[7])
            }
        }

        # Save to metrics file
        metrics_file = os.path.join(save_dir, 'validation_metrics.json')
        os.makedirs(save_dir, exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print_log(f"=> Saved detailed validation metrics to {metrics_file}")

        # Print concise summary to training log and terminal
        summary_str = (
            f"=> Validation Summary: "
            f"Person_MOTA={person_metrics['mota']:.2f}% | "
            f"mAP={ap_cum[7]:.2f} | "
            f"Joint_MOTA={joint_mota_unfiltered:.2f}% | "
            f"Precision={pre_unfiltered:.2f}% | "
            f"Recall={rec_unfiltered:.2f}%"
        )
        print_log(summary_str)

        name_value = OrderedDict([
            ('Head', ap_cum[0]),
            ('Shoulder', ap_cum[1]),
            ('Elbow', ap_cum[2]),
            ('Wrist', ap_cum[3]),
            ('Hip', ap_cum[4]),
            ('Knee', ap_cum[5]),
            ('Ankle', ap_cum[6]),
            ('Mean', ap_cum[7]),
            ('Person_MOTA', person_metrics['mota']),
            ('Person_Precision', person_metrics['precision']),
            ('Person_Recall', person_metrics['recall']),
            ('Person_ID_Switches', person_metrics['id_switches']),
            ('Joint_MOTA', float(joint_mota_unfiltered)),
        ])

        return name_value, name_value['Mean']

    def _fill_missing_frames(self, out_data, L, out_filenames):
        """Fill in empty placeholder frames for videos missing predictions.

        Args:
            out_data (dict): PoseTrack output data keyed by video name.
            L (dict): Video name to total frame count mapping.
            out_filenames (dict): Video name to output filename mapping.

        Returns:
            dict: out_data with missing frames filled in.
        """
        for key in out_data.keys():
            new_data = []
            data = out_data[key]
            arrs = data[0]['image']['name'].split('/')
            num_frames = L['images/' + key]  # posetrack17
            # num_frames = L[key]  # posetrack18

            frame_ids = [i for i in range(1, num_frames + 1)]  # posetrack17
            # frame_ids = [i for i in range(0, num_frames)]  # posetrack18
            count = 0
            used_frame_ids = [d['img_num'][0] for d in data]
            for frame_id in frame_ids:
                if frame_id not in used_frame_ids:
                    annorect = []
                    img_sfx = (arrs[0] + '/' + arrs[1] + '/'
                               + str(frame_id).zfill(8) + '.jpg')  # posetrack17
                    # img_sfx = (arrs[0] + '/' + arrs[1] + '/' + arrs[2]
                    #            + '/' + str(frame_id).zfill(6) + '.jpg')  # posetrack18
                    annorect.append({
                        'annopoints': [{'point': [{
                            'id': [0],
                            'x': [0],
                            'y': [0],
                            'score': [-100.0],
                        }]}],
                        'score': [0],
                        'track_id': [0]})
                    new_data.append({
                        'image': {'name': img_sfx},
                        'imgnum': [frame_id],
                        'annorect': annorect
                    })
                    count += 1
                else:
                    new_data.append(data[frame_id - count - 1])  # posetrack17
                    # new_data.append(data[frame_id - count])  # posetrack18
            out_data[key] = new_data
        return out_data

    def evaluate_from_json(self, save_dir, logger=None):
        """Recompute metrics from previously saved JSON result directories.

        Expects the following subdirectories under save_dir:
          - val_set_json_results_ap/          (unfiltered, for AP)
          - val_set_json_results/             (person-filtered, for MOTA)
          - val_set_json_results_joint_filtered/  (joint-filtered, optional)

        Args:
            save_dir (str): Directory containing the saved JSON results.
            logger: Optional logger.

        Returns:
            OrderedDict: AP name-value pairs.
        """
        from ..core.posetrack_utils.poseval.py.evaluateAP import evaluateAP
        from ..core.posetrack_utils.poseval.py.evaluateTracking import \
            evaluateTracking
        from ..core.posetrack_utils.poseval.py.eval_helpers import (
            Joint, printTable, load_data_dir)

        annot_dir = 'DcPose_supp_files/posetrack17_annotation_dirs/jsons/val/'
        nJoints = Joint().count

        ap_output_dir = os.path.join(save_dir, 'val_set_json_results_ap')
        track_output_dir = os.path.join(save_dir, 'val_set_json_results')
        jfilt_output_dir = os.path.join(
            save_dir, 'val_set_json_results_joint_filtered')

        # --- AP evaluation ---
        if os.path.isdir(ap_output_dir):
            print_log(f"=> Running AP evaluation from {ap_output_dir}")
            gtFramesAll_ap, prFramesAll_ap = load_data_dir(
                ['', annot_dir, ap_output_dir])
            apAll, _, _ = evaluateAP(gtFramesAll_ap, prFramesAll_ap)
            ap_cum = printTable(apAll)
        else:
            print_log(f"WARNING: AP dir not found: {ap_output_dir}")
            ap_cum = [0.0] * 8

        # --- Unfiltered MOTA ---
        if os.path.isdir(track_output_dir):
            print_log(f"=> Running MOTA evaluation from {track_output_dir}")
            gtFramesAll_trk, prFramesAll_trk = load_data_dir(
                ['', annot_dir, track_output_dir])
            metricsAll = evaluateTracking(
                gtFramesAll_trk, prFramesAll_trk, False)
            joint_mota_unfiltered = metricsAll['mota'][0, nJoints]
            motp_unfiltered = metricsAll['motp'][0, nJoints]
            pre_unfiltered = metricsAll['pre'][0, nJoints]
            rec_unfiltered = metricsAll['rec'][0, nJoints]
            person_metrics = self.compute_person_level_mota(
                gtFramesAll_trk, prFramesAll_trk, distThresh=0.5)
        else:
            print_log(f"WARNING: Track dir not found: {track_output_dir}")
            joint_mota_unfiltered = 0.0
            motp_unfiltered = 0.0
            pre_unfiltered = 0.0
            rec_unfiltered = 0.0
            person_metrics = {
                'gt_persons': 0, 'matched': 0, 'misses': 0,
                'fp': 0, 'id_switches': 0,
                'mota': 0.0, 'precision': 0.0, 'recall': 0.0}

        # --- Joint-filtered MOTA ---
        if os.path.isdir(jfilt_output_dir):
            print_log(f"=> Running joint-filtered MOTA from {jfilt_output_dir}")
            gtFramesAll_jf, prFramesAll_jf = load_data_dir(
                ['', annot_dir, jfilt_output_dir])
            metricsAll_jf = evaluateTracking(
                gtFramesAll_jf, prFramesAll_jf, False)
            joint_mota_filtered = metricsAll_jf['mota'][0, nJoints]
            motp_filtered = metricsAll_jf['motp'][0, nJoints]
            pre_filtered = metricsAll_jf['pre'][0, nJoints]
            rec_filtered = metricsAll_jf['rec'][0, nJoints]
            person_metrics_jf = self.compute_person_level_mota(
                gtFramesAll_jf, prFramesAll_jf, distThresh=0.5)
        else:
            print_log(f"NOTE: Joint-filtered dir not found: {jfilt_output_dir}")
            joint_mota_filtered = None
            motp_filtered = None
            pre_filtered = None
            rec_filtered = None
            person_metrics_jf = None

        # --- Print consolidated results ---
        filt_col = f"{'Joint-Filtered':<20}" if joint_mota_filtered is not None else ""
        filt_val = lambda v: f"{v:<20.2f}" if v is not None else ""

        lines = []
        lines.append("\n" + "="*60)
        lines.append("JOINT-LEVEL MEAN MOTA")
        lines.append("="*60)
        lines.append(f"{'Metric':<20} {'Unfiltered':<20} {filt_col}")
        lines.append("-"*60)
        lines.append(f"{'MOTA (%)':<20} {joint_mota_unfiltered:<20.2f} {filt_val(joint_mota_filtered)}")
        lines.append(f"{'MOTP (%)':<20} {motp_unfiltered:<20.2f} {filt_val(motp_filtered)}")
        lines.append(f"{'Precision (%)':<20} {pre_unfiltered:<20.2f} {filt_val(pre_filtered)}")
        lines.append(f"{'Recall (%)':<20} {rec_unfiltered:<20.2f} {filt_val(rec_filtered)}")
        lines.append("="*60)

        pf_col = f"{'Joint-Filtered':<20}" if person_metrics_jf else ""
        pf_val = lambda k: f"{person_metrics_jf[k]:<20}" if person_metrics_jf else ""
        pf_fval = lambda k: f"{person_metrics_jf[k]:<20.2f}" if person_metrics_jf else ""

        lines.append("")
        lines.append("="*60)
        lines.append("PERSON-LEVEL MOTA")
        lines.append("="*60)
        lines.append(f"{'Metric':<20} {'Unfiltered':<20} {pf_col}")
        lines.append("-"*60)
        lines.append(f"{'GT Persons':<20} {person_metrics['gt_persons']:<20} {pf_val('gt_persons')}")
        lines.append(f"{'Matched':<20} {person_metrics['matched']:<20} {pf_val('matched')}")
        lines.append(f"{'Misses':<20} {person_metrics['misses']:<20} {pf_val('misses')}")
        lines.append(f"{'False Positives':<20} {person_metrics['fp']:<20} {pf_val('fp')}")
        lines.append(f"{'ID Switches':<20} {person_metrics['id_switches']:<20} {pf_val('id_switches')}")
        lines.append(f"{'MOTA (%)':<20} {person_metrics['mota']:<20.2f} {pf_fval('mota')}")
        lines.append(f"{'Precision (%)':<20} {person_metrics['precision']:<20.2f} {pf_fval('precision')}")
        lines.append(f"{'Recall (%)':<20} {person_metrics['recall']:<20.2f} {pf_fval('recall')}")
        lines.append("="*60)

        joint_ap_names = ['Head', 'Shoulder', 'Elbow', 'Wrist',
                          'Hip', 'Knee', 'Ankle']
        lines.append("")
        lines.append("="*40)
        lines.append("AVERAGE PRECISION (mAP)")
        lines.append("="*40)
        lines.append(f"{'Joint':<20} {'AP (%)':<20}")
        lines.append("-"*40)
        for i, name in enumerate(joint_ap_names):
            lines.append(f"{name:<20} {ap_cum[i]:<20.2f}")
        lines.append("-"*40)
        lines.append(f"{'Mean':<20} {ap_cum[7]:<20.2f}")
        lines.append("="*40 + "\n")

        print_log("\n".join(lines))

        name_value = OrderedDict([
            ('Head', ap_cum[0]), ('Shoulder', ap_cum[1]),
            ('Elbow', ap_cum[2]), ('Wrist', ap_cum[3]),
            ('Hip', ap_cum[4]), ('Knee', ap_cum[5]),
            ('Ankle', ap_cum[6]), ('Mean', ap_cum[7])
        ])
        return name_value, name_value['Mean']

    def compute_person_level_mota(self, gtFramesAll, prFramesAll, distThresh=0.5):
        """Compute MOTA metrics at the person/pose level (not joint level).

        This provides a clearer picture of tracking performance:
        - 1 person = 1 object (not 15 joints)
        - Counts misses, detections, FPs, ID switches for persons

        Args:
            gtFramesAll: Ground truth frames (from load_data_dir)
            prFramesAll: Predicted frames (from load_data_dir)
            distThresh: PCK threshold for matching (default 0.5)

        Returns:
            dict: Person-level MOTA metrics
        """
        from ..core.posetrack_utils.poseval.py.eval_helpers import (
            assignGTmulti)

        # Get the assignment info
        _, _, _, motAll = assignGTmulti(gtFramesAll, prFramesAll, distThresh)

        # Track person-level metrics
        total_gt_persons = 0
        total_matched_persons = 0
        total_fp_persons = 0
        total_id_switches = 0
        prev_frame_gt_ids = {}  # track_id -> matched_pred_id mapping

        for imgidx in range(len(gtFramesAll)):
            gtFrame = gtFramesAll[imgidx]
            prFrame = prFramesAll[imgidx]

            # Extract person counts from current frame
            num_gt_persons = len(gtFrame['annorect'])
            num_pr_persons = len(prFrame['annorect'])

            total_gt_persons += num_gt_persons

            # Build GT track_id -> index mapping
            gt_track_ids = {}
            for ridxGT in range(num_gt_persons):
                if 'track_id' in gtFrame['annorect'][ridxGT]:
                    tid = gtFrame['annorect'][ridxGT]['track_id'][0]
                    gt_track_ids[tid] = ridxGT

            # Build prediction track_id -> index mapping
            pr_track_ids = {}
            for ridxPr in range(num_pr_persons):
                if 'track_id' in prFrame['annorect'][ridxPr]:
                    tid = prFrame['annorect'][ridxPr]['track_id'][0]
                    pr_track_ids[tid] = ridxPr

            # For person-level matching, use the joint-level assignment
            # A person is matched if at least one joint matched
            pr_to_gt_person = {}  # pr_idx -> gt_idx
            for ridxPr in range(num_pr_persons):
                for ridxGT in range(num_gt_persons):
                    # Check if any joint matched between this pr and gt
                    any_match = False
                    for i in range(15):  # 15 joints
                        if i in motAll[imgidx]:
                            # Check if this pred-gt pair has a joint match
                            if (ridxPr in motAll[imgidx][i]['ridxsPr'] and
                                    ridxGT in motAll[imgidx][i]['ridxsGT']):
                                # Find indices in the per-joint arrays
                                pr_idx_in_joint = list(
                                    motAll[imgidx][i]['ridxsPr']).index(ridxPr)
                                gt_idx_in_joint = list(
                                    motAll[imgidx][i]['ridxsGT']).index(ridxGT)
                                if not np.isnan(
                                        motAll[imgidx][i]['dist'][
                                            gt_idx_in_joint, pr_idx_in_joint]):
                                    any_match = True
                                    break
                    if any_match:
                        pr_to_gt_person[ridxPr] = ridxGT
                        break

            # Count matched and FP persons this frame
            num_matched = len(pr_to_gt_person)
            num_fp = num_pr_persons - num_matched
            num_misses = num_gt_persons - num_matched

            total_matched_persons += num_matched
            total_fp_persons += num_fp

            # Count ID switches - when a GT person matched by different PR
            # person than in previous frame
            for gt_tid, gt_idx in gt_track_ids.items():
                if gt_tid in prev_frame_gt_ids:
                    prev_pr_idx = prev_frame_gt_ids[gt_tid]
                    # Check if same prediction matched this GT person
                    curr_pr_idx = None
                    for ridxPr, ridxGT in pr_to_gt_person.items():
                        if ridxGT == gt_idx:
                            curr_pr_idx = ridxPr
                            break
                    # If different PR matched or no match, that's an ID switch
                    if curr_pr_idx is None or curr_pr_idx != prev_pr_idx:
                        if curr_pr_idx is not None:  # Only count if matched now
                            total_id_switches += 1

            # Update tracking for next frame
            prev_frame_gt_ids = {}
            for ridxPr, ridxGT in pr_to_gt_person.items():
                gt_tid = None
                if 'track_id' in gtFrame['annorect'][ridxGT]:
                    gt_tid = gtFrame['annorect'][ridxGT]['track_id'][0]
                if gt_tid is not None:
                    pr_tid = None
                    if 'track_id' in prFrame['annorect'][ridxPr]:
                        pr_tid = prFrame['annorect'][ridxPr]['track_id'][0]
                    if pr_tid is not None:
                        prev_frame_gt_ids[gt_tid] = ridxPr

        # Compute person-level MOTA
        if total_gt_persons > 0:
            person_mota = 1.0 - (
                total_fp_persons + (total_gt_persons - total_matched_persons)
                + total_id_switches) / total_gt_persons
        else:
            person_mota = 0.0

        precision = (total_matched_persons / (total_matched_persons
                     + total_fp_persons)
                     if (total_matched_persons + total_fp_persons) > 0 else 0.0)
        recall = (total_matched_persons / total_gt_persons
                  if total_gt_persons > 0 else 0.0)

        return {
            'gt_persons': total_gt_persons,
            'matched': total_matched_persons,
            'misses': total_gt_persons - total_matched_persons,
            'fp': total_fp_persons,
            'id_switches': total_id_switches,
            'mota': person_mota * 100,
            'precision': precision * 100,
            'recall': recall * 100,
        }

    def create_folder(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
    def video2filenames(self, annot_dir):
        pathtodir = annot_dir

        output, L = {}, {}
        mat_files = [f for f in os.listdir(pathtodir) if
                    os.path.isfile(os.path.join(pathtodir, f)) and '.mat' in f]
        json_files = [f for f in os.listdir(pathtodir) if
                    os.path.isfile(os.path.join(pathtodir, f)) and '.json' in f]

        if len(json_files) > 1:
            files = json_files
            ext_types = '.json'
        else:
            files = mat_files
            ext_types = '.mat'

        for fname in files:
            if ext_types == '.mat':
                out_fname = fname.replace('.mat', '.json')
                data = sio.loadmat(
                    os.path.join(pathtodir, fname), squeeze_me=True,
                    struct_as_record=False)
                temp = data['annolist'][0].image.name

                data2 = sio.loadmat(os.path.join(pathtodir, fname))
                num_frames = len(data2['annolist'][0])
            elif ext_types == '.json':
                out_fname = fname
                with open(os.path.join(pathtodir, fname), 'r') as fin:
                    data = json.load(fin)

                if 'annolist' in data:
                    temp = data['annolist'][0]['image'][0]['name']
                    num_frames = len(data['annolist'])
                else:
                    temp = data['images'][0]['file_name']
                    num_frames = data['images'][0]['nframes']


            else:
                raise NotImplementedError()
            video = os.path.dirname(temp)
            output[video] = out_fname
            L[video] = num_frames
        return output, L

    # 根据当前图片获取前后帧
    def _get_auxiliary_frames(self,img_index):
        info = self.coco.load_imgs([img_index])[0]
        image_file_path = info['file_name']
        zero_fill = len(osp.basename(image_file_path).replace('.jpg', ''))
        current_idx = int(osp.basename(image_file_path).replace('.jpg', ''))
        prev_delta = 1
        next_delta = 1
        # posetrack17
        # 当为第一帧的时，则取当前帧作为前一帧
        if current_idx == 1:
            prev_delta = 0
            next_delta = 1
        # 当为最后一帧时，则取当前帧作为后一帧
        if current_idx == info['nframes']:
            prev_delta = 1
            next_delta = 0
        # posetrack18
        # # 当为第一帧的时，则取当前帧作为前一帧
        # if current_idx == 0:
        #     prev_delta = 0
        #     next_delta = 1
        # # 当为最后一帧时，则取当前帧作为后一帧
        # if current_idx == info['nframes'] - 1:
        #     prev_delta = 1
        #     next_delta = 0
        prev_idx = current_idx - prev_delta
        next_idx = current_idx + next_delta
        now_image_file = image_file_path
        prev_image_file = osp.join(osp.dirname(image_file_path), str(prev_idx).zfill(zero_fill) + '.jpg')
        next_image_file = osp.join(osp.dirname(image_file_path), str(next_idx).zfill(zero_fill) + '.jpg')
        # print(now_image_file)
        
        return [prev_image_file,now_image_file,next_image_file]
 
    def _get_data(self):
        data_infos  = []
        for img_index in self.img_ids:
            info = self.coco.load_imgs([img_index])[0]
            # 需要判断当前帧是否已经被标注了，没有被标注则被排除
            if info['is_labeled']:
                imgs = self._get_auxiliary_frames(img_index=img_index)
                info['filename_prev'] = imgs[0]
                info['filename_now'] = imgs[1]
                info['filename_next'] = imgs[2]
                data_infos.append(info)
        return data_infos
    
@DATASETS.register_module()
class PosetrackSequentialDataset(PosetrackVideoPoseDataset):
    """PoseTrack dataset that yields sequences of overlapping 3-frame windows.

    Each __getitem__ returns a list of T samples (dicts) from consecutive
    labeled frames in the same video, enabling track query propagation
    during training.

    The parent class handles data loading and filtering. After that completes,
    this class groups the final filtered frames by video and builds sliding
    window sequences.

    Args:
        seq_length (int): Number of consecutive windows per sample. Default: 2.
        *args, **kwargs: Same as PosetrackVideoPoseDataset.
    """

    def __init__(self, *args, seq_length=2, **kwargs):
        self.seq_length = seq_length
        self._seq_indices = None  # Built after parent init completes
        super().__init__(*args, **kwargs)
        # At this point self.data_infos is fully loaded and filtered.
        # Build sequential windows from the final data_infos.
        self._build_sequences()
        # Rebuild group flag now that __len__ returns sequence count
        self._set_group_flag()

    def _build_sequences(self):
        """Build sequential windows from the final (filtered) data_infos."""
        from collections import defaultdict
        # Group data_infos indices by video
        video_frames = defaultdict(list)
        for i, info in enumerate(self.data_infos):
            video_name = os.path.dirname(info['file_name'])
            video_frames[video_name].append(i)

        # Sort frames within each video by frame number
        for vname in video_frames:
            video_frames[vname].sort(
                key=lambda idx: int(os.path.splitext(
                    os.path.basename(self.data_infos[idx]['file_name']))[0]))

        # Create sliding window sequences
        self._seq_indices = []
        for vname, frame_indices in video_frames.items():
            n = len(frame_indices)
            if n < self.seq_length:
                # Video too short — pad by repeating last frame
                seq = list(frame_indices)
                while len(seq) < self.seq_length:
                    seq.append(seq[-1])
                self._seq_indices.append(seq)
            else:
                for i in range(n - self.seq_length + 1):
                    self._seq_indices.append(
                        frame_indices[i:i + self.seq_length])

    def _set_group_flag(self):
        """Set flag according to image aspect ratio."""
        if self._seq_indices is None:
            # During super().__init__(), before sequences are built —
            # use parent behavior (iterates over data_infos directly)
            super()._set_group_flag()
            return
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            frame_idx = self._seq_indices[i][0]
            img_info = self.data_infos[frame_idx]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def __len__(self):
        if self._seq_indices is None:
            # During super().__init__(), before sequences are built
            return len(self.data_infos)
        return len(self._seq_indices)

    def __getitem__(self, idx):
        """Return a list of T processed samples from consecutive frames.

        All frames in the sequence share the same random augmentation seed
        so that flip/scale are consistent across windows. Without this,
        a flipped window 0 produces track queries at mirrored positions that
        can never match GT in an unflipped window 1, causing the model to
        learn to suppress all track query scores and degrading MOTA.
        """
        seq = self._seq_indices[idx]
        samples = []
        # Sample one seed for the whole sequence so all frames in the window
        # get the same random flip / scale / crop decisions.
        aug_seed = np.random.randint(0, 2**31)
        for frame_idx in seq:
            # Reset numpy, Python, and PyTorch random state to the shared seed
            # before each frame so the pipeline sees identical random draws.
            np.random.seed(aug_seed)
            import random as _random
            _random.seed(aug_seed)
            torch.manual_seed(aug_seed)
            # Use parent's prepare_train_img which applies the full pipeline
            sample = self.prepare_train_img(frame_idx)
            if sample is None:
                # If pipeline returns None (e.g., empty annotations),
                # retry with a random different index
                return self.__getitem__(
                    np.random.randint(0, len(self)))
            samples.append(sample)
        return samples


def sequential_collate(batch, samples_per_gpu=1):
    """Custom collate for sequential dataset.

    Each item in batch is a list of T dicts (one per window).
    We collate across the batch for each time step separately.

    Returns:
        list[dict]: T dicts, each collated across the batch.
    """
    from mmcv.parallel import collate
    seq_length = len(batch[0])
    collated = []
    for t in range(seq_length):
        # Gather time step t from all batch items
        step_batch = [item[t] for item in batch]
        collated.append(collate(step_batch, samples_per_gpu=samples_per_gpu))
    return collated


def write_json_to_file(data, output_path, flag_verbose=False):
    with open(output_path, "w") as write_file:
        json.dump(data, write_file)
    if flag_verbose is True:
        print("Json string dumped to: %s", output_path)
