# Copyright (c) Hikvision Research Institute. All rights reserved.
import random
import mmcv
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle
from mmdet.core.visualization import color_val_matplotlib
from mmdet.core import bbox_mapping_back, multiclass_nms
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.detectors.detr import DETR

from opera.core.keypoint import bbox_kpt2result, kpt_mapping_back
from ..builder import DETECTORS
import cv2


@DETECTORS.register_module()
class PAVE(DETR):
    """Implementation of `End-to-End Multi-Person Pose Estimation with
    Transformers`"""

    # Keys to show in the training log; all others are suppressed.
    _LOG_KEYS = {
        'enc_loss_cls',  # encoder
        'trk_cls_val', 'det_cls_val',  # per-query-type classification metrics
        'loss_cls',
        'loss',
        'n_trk', 'n_trk_pos', 'n_det_pos', 'trk_id_con',
    }

    def __init__(self, *args, **kwargs):
        super(DETR, self).__init__(*args, **kwargs)
        self.num_iters = 0
        self.max_iters = 1000
        self.avg_losses = []
        # Tracking state for inference
        self.track_queries = None
        self.track_reference_points = None
        self.track_query_pos = None
        self.track_ids = None
        self.next_track_id = 0

    def reset_tracking_state(self):
        """Reset tracking state between videos during inference."""
        self.track_queries = None
        self.track_reference_points = None
        self.track_query_pos = None
        self.track_ids = None
        self.next_track_id = 0

    def train_step(self, data, optimizer, **kwargs):
        """Override train_step to handle sequential data.

        When using PosetrackSequentialDataset, the dataloader returns a list
        of T collated dicts (one per time step). The default train_step does
        self(**data) which fails because data is a list, not a dict.
        """
        if isinstance(data, list):
            # Sequential training: data is [dict_t0, dict_t1, ...]
            # Each dict has keys: img, img_metas, gt_bboxes, gt_labels, etc.
            T = len(data)
            imgs = [data[t]['img'] for t in range(T)]
            img_metas_list = [data[t]['img_metas'] for t in range(T)]
            gt_bboxes_list = [data[t]['gt_bboxes'] for t in range(T)]
            gt_labels_list = [data[t]['gt_labels'] for t in range(T)]
            gt_keypoints_list = [data[t]['gt_keypoints'] for t in range(T)]
            gt_areas_list = [data[t]['gt_areas'] for t in range(T)]
            gt_track_ids_list = [data[t].get('gt_track_ids') for t in range(T)]

            losses = self.forward_train(
                imgs, img_metas_list, gt_bboxes_list, gt_labels_list,
                gt_keypoints_list, gt_areas_list,
                gt_track_ids=gt_track_ids_list)

            loss, log_vars = self._parse_losses(losses)
            log_vars = {k: v for k, v in log_vars.items() if k in self._LOG_KEYS}
            outputs = dict(
                loss=loss, log_vars=log_vars,
                num_samples=len(data[0]['img_metas']))
            return outputs
        else:
            # Standard single-frame training
            outputs = super().train_step(data, optimizer, **kwargs)
            outputs['log_vars'] = {
                k: v for k, v in outputs['log_vars'].items()
                if k in self._LOG_KEYS}
            return outputs

    def forward_train(self,
                      img,
                      img_metas=None,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_keypoints=None,
                      gt_areas=None,
                      gt_bboxes_ignore=None,
                      gt_track_ids=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
            gt_keypoints (list[Tensor]): Each item are the truth keypoints for
                each image in [p^{1}_x, p^{1}_y, p^{1}_v, ..., p^{K}_x,
                p^{K}_y, p^{K}_v] format.
            gt_areas (list[Tensor]): mask areas corresponding to each box.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Check if this is sequential training (img is list of T tensors)
        if isinstance(img, (list, tuple)):
            # Set batch_input_shape per window (normally done by super().forward_train)
            for t in range(len(img)):
                batch_input_shape = tuple(img[t].size()[-2:])
                for meta in img_metas[t]:
                    meta['batch_input_shape'] = batch_input_shape
            return self._forward_train_sequential(
                img, img_metas, gt_bboxes, gt_labels,
                gt_keypoints, gt_areas, gt_track_ids, gt_bboxes_ignore)

        super(SingleStageDetector, self).forward_train(img, img_metas)

        # Standard single-window training (backward compatible)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_keypoints,
                                              gt_areas, gt_bboxes_ignore)
            
        return losses

    def _forward_train_sequential(self, imgs, img_metas_list, gt_bboxes_list,
                                   gt_labels_list, gt_keypoints_list,
                                   gt_areas_list, gt_track_ids_list,
                                   gt_bboxes_ignore):
        """Process T sequential windows with track query propagation.

        Args:
            imgs (list[Tensor]): T image tensors, each shape [N, C, H, W].
            img_metas_list (list[list[dict]]): T sets of img_metas.
            gt_bboxes_list (list[list[Tensor]]): T sets of GT bboxes.
            gt_labels_list (list[list[Tensor]]): T sets of GT labels.
            gt_keypoints_list (list[list[Tensor]]): T sets of GT keypoints.
            gt_areas_list (list[list[Tensor]]): T sets of GT areas.
            gt_track_ids_list (list[list[Tensor]]): T sets of GT track IDs.
            gt_bboxes_ignore: Ignored bboxes.

        Returns:
            dict[str, Tensor]: Accumulated loss components averaged over T.
        """
        total_losses = {}
        track_state = None
        # Track diagnostic keys — use last window's values (not averaged),
        # since only windows after the first have meaningful track queries.
        _track_diag_keys = {'trk_kpt_l1', 'det_kpt_l1', 'n_trk', 'n_trk_pos',
                            'n_det_pos', 'trk_id_con', 'n_fn_drop', 'n_fp_inj',
                            'n_hard_match', 'n_hard_bg', 'trk_cls_val', 'det_cls_val'}

        # ---- TrackFormer-style per-query augmentation probabilities ----
        # pFN: each track query is independently dropped with this probability.
        #      Dropped queries' GT objects fall back to detect queries, forcing
        #      detect queries to cover ~pFN of all tracked persons too.
        # Configurable via bbox_head.track_p_fn (default 0.4 per TrackFormer).
        _P_FN = getattr(self.bbox_head, 'track_p_fn', 0.4)

        T = len(imgs)
        for t in range(T):
            is_last = (t == T - 1)

            # ---- Per-query false-negative dropout (TrackFormer §3.3) ----
            # Applied before the last window (which has track queries).
            effective_track_state = track_state
            n_fn_dropped = 0
            if is_last and track_state is not None:
                tq, tr, tp, tids = track_state
                n_queries = tq.shape[1]  # [bs, N_trk, dim]
                keep_mask = torch.tensor(
                    [random.random() >= _P_FN for _ in range(n_queries)],
                    dtype=torch.bool)
                n_fn_dropped = int((~keep_mask).sum().item())
                if keep_mask.any():
                    effective_track_state = (
                        tq[:, keep_mask, :],
                        tr[:, keep_mask, :],
                        tp[:, keep_mask, :],
                        tids[keep_mask],
                    )
                else:
                    # All queries dropped — equivalent to first-frame
                    effective_track_state = None

            # Compute loss on ALL windows with gradients (TrackFormer-style).
            # Window 0 has no track queries → detect queries must cover all GT.
            # Window 1+ has track queries → detect queries cover remaining GT.
            # This prevents detect query specialization.
            x = self.extract_feat(imgs[t])
            losses, track_state = self.bbox_head.forward_train_sequential(
                x, img_metas_list[t], gt_bboxes_list[t],
                gt_labels_list[t], gt_keypoints_list[t],
                gt_areas_list[t],
                gt_track_ids_list[t] if gt_track_ids_list is not None else None,
                gt_bboxes_ignore, track_state=effective_track_state)
            if is_last:
                # Inject FN dropout count into diagnostics
                losses['n_fn_drop'] = torch.tensor(
                    float(n_fn_dropped), device=imgs[t].device)

            for k, v in losses.items():
                if k in _track_diag_keys:
                    # Diagnostic keys: keep last window's values only
                    if is_last:
                        total_losses[k] = v
                elif isinstance(v, torch.Tensor):
                    total_losses[k] = total_losses.get(k, 0) + v / T
                elif isinstance(v, list):
                    if k not in total_losses:
                        total_losses[k] = [vi / T for vi in v]
                    else:
                        total_losses[k] = [a + vi / T for a, vi in zip(total_losses[k], v)]

            # Detach track state (prevents gradient flow across windows;
            # also needed even for no_grad windows to drop any stale grad_fn)
            if track_state is not None:
                track_state = tuple(
                    s.detach() if isinstance(s, torch.Tensor) else s
                    for s in track_state)

        return total_losses

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`.
        """
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3),
                scale_factor=(1., 1., 1., 1.)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.bbox_head(x, img_metas=dummy_img_metas)
        # Strip n_track from outs before passing to get_bboxes
        bbox_list = self.bbox_head.get_bboxes(
            *outs[:-1], dummy_img_metas, rescale=True)
        return bbox_list

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Supports track query propagation when self.track_queries is set.

        Args:
            img (list[torch.Tensor]): List of multiple images.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox and keypoint results of each image
                and classes. The outer list corresponds to each image.
                The inner list corresponds to each class.
        """
        import time
        start = time.time()
        batch_size = len(img_metas)
        assert batch_size == 1, 'Currently only batch_size 1 for inference ' \
            f'mode is supported. Found batch_size {batch_size}.'
        
        feat = self.extract_feat(img)

        # Pass track queries if available
        outs = self.bbox_head(
            feat, img_metas,
            track_queries=self.track_queries,
            track_reference_points=self.track_reference_points,
            track_query_pos=self.track_query_pos)

        n_track = outs[-1]

        # Build track ID array for ALL queries before filtering.
        # Track queries (0..n_track-1) keep their propagated IDs.
        # Detect queries (n_track..num_queries-1) get new sequential IDs.
        num_queries = outs[0].shape[2]  # [nb_dec, bs, num_query, ...]
        num_detect = num_queries - n_track
        all_query_track_ids = []
        for i in range(num_queries):
            if i < n_track and self.track_ids is not None:
                all_query_track_ids.append(self.track_ids[i])
            else:
                all_query_track_ids.append(
                    self.next_track_id + (i - n_track))

        results_list = self.bbox_head.get_bboxes(
            *outs[:-1], img_metas, rescale=rescale)

        # --- TrackFormer §3.3: Two-stage OKS-based NMS ---
        # Stage 1: Track-vs-detect NMS — track queries take priority;
        #          overlapping detect queries are suppressed.
        # Stage 2: All-vs-all OKS NMS — removes strongly overlapping
        #          duplicates that the decoder self-attention could not
        #          resolve ("not resolvable by the decoder self-attention").
        # Both stages use a HIGH threshold (default 0.9) to only suppress
        # near-duplicates and preserve recall.
        track_nms_thr = getattr(self.bbox_head, 'track_nms_thr', 0.9)
        if n_track > 0 and track_nms_thr < 1.0:
            results_list = [
                self._nms_track_detect(item, n_track, track_nms_thr)
                for item in results_list]
        # All-vs-all OKS NMS with high threshold to remove remaining duplicates
        results_list = [
            self._oks_nms_all(item, oks_thr=track_nms_thr)
            for item in results_list]

        # Update tracking state using the consistent ID array
        self._update_tracking_state(
            outs, results_list, n_track,
            all_query_track_ids=all_query_track_ids)

        end = time.time()
        # print(f'Inference time: {end - start}')

        bbox_kpt_results = []
        for item in results_list:
            det_bboxes, det_labels, det_kpts, bbox_index = item
            # Map query indices to track IDs
            det_track_ids = np.array(
                [all_query_track_ids[idx.item()] for idx in bbox_index],
                dtype=np.int64)
            bbox_kpt_results.append(
                bbox_kpt2result(det_bboxes, det_labels, det_kpts,
                                self.bbox_head.num_classes,
                                track_ids=det_track_ids))
        return bbox_kpt_results

    def _update_tracking_state(self, outs, results_list, n_track,
                                score_thr=None, max_track_queries=None,
                                all_query_track_ids=None):
        """Update tracking state after inference on a frame.

        Extracts high-confidence queries and their reference points to
        propagate as track queries for the next frame.

        Args:
            outs: Raw outputs from bbox_head forward.
            results_list: Detection results.
            n_track (int): Number of track queries used in this frame.
            score_thr (float): Score threshold for query selection.
            max_track_queries (int): Maximum track queries to propagate.
            all_query_track_ids (list[int] | None): Pre-built track ID array
                for all queries. When provided, IDs are looked up from here
                instead of being independently assigned.
        """
        # Read thresholds from head config, with fallback defaults
        if score_thr is None:
            score_thr = getattr(self.bbox_head, 'track_score_thr', 0.5)
        if max_track_queries is None:
            max_track_queries = getattr(self.bbox_head, 'max_track_queries', 100)

        # Use stored hidden states from the head
        if not hasattr(self.bbox_head, '_last_hs') or self.bbox_head._last_hs is None:
            self.track_queries = None
            self.track_reference_points = None
            self.track_query_pos = None
            return

        last_hs = self.bbox_head._last_hs  # [bs//3, num_q, 256]
        # Get classification scores from the last decoder layer output
        all_cls_scores = outs[0]  # [nb_dec, bs, num_q, 1]
        cls_scores = all_cls_scores[-1]  # [bs, num_q, 1]
        scores = cls_scores.sigmoid().squeeze(-1)  # [bs, num_q]

        # Get kpt predictions from last decoder layer
        all_kpt_preds = outs[1]  # [nb_dec, bs, num_q, K*2]
        kpt_preds = all_kpt_preds[-1]  # [bs, num_q, K*2]

        # Select high-confidence queries
        scores_0 = scores[0]
        mask = scores_0 > score_thr
        if mask.sum() == 0:
            self.track_queries = None
            self.track_reference_points = None
            self.track_query_pos = None
            return

        if mask.sum() > max_track_queries:
            topk_vals, topk_inds = scores_0.topk(max_track_queries)
            mask = torch.zeros_like(scores_0, dtype=torch.bool)
            mask[topk_inds] = True

        selected_inds = mask.nonzero(as_tuple=True)[0]

        self.track_queries = last_hs[:, selected_inds, :].detach()
        self.track_reference_points = kpt_preds[:, selected_inds, :].detach()
        self.track_query_pos = torch.zeros_like(self.track_queries)

        # Assign persistent track IDs from the pre-built array
        if all_query_track_ids is not None:
            new_track_ids = [all_query_track_ids[idx.item()]
                             for idx in selected_inds]
            # Update next_track_id to be beyond all assigned IDs
            self.next_track_id = max(all_query_track_ids) + 1
        else:
            new_track_ids = []
            for i, idx in enumerate(selected_inds):
                idx_val = idx.item()
                if n_track > 0 and idx_val < n_track and self.track_ids is not None:
                    new_track_ids.append(self.track_ids[idx_val])
                else:
                    new_track_ids.append(self.next_track_id)
                    self.next_track_id += 1
        self.track_ids = new_track_ids

    @staticmethod
    def _get_oks_sigmas(det_kpts):
        """Get per-keypoint sigmas based on the number of keypoints."""
        num_kpts = det_kpts.shape[1]
        if num_kpts == 15:
            sigmas = det_kpts.new_tensor([
                .26, .79, .79, .79, .79, .72, .72,
                .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
        elif num_kpts == 17:
            sigmas = det_kpts.new_tensor([
                .26, .25, .25, .35, .35, .79, .79, .72, .72,
                .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
        elif num_kpts == 14:
            sigmas = det_kpts.new_tensor([
                .79, .79, .72, .72, .62, .62, 1.07, 1.07,
                .87, .87, .89, .89, .79, .79]) / 10.0
        else:
            sigmas = det_kpts.new_full((num_kpts,), 0.079)
        return sigmas

    @staticmethod
    def _compute_pairwise_oks(kpts_a, bboxes_a, kpts_b, bboxes_b, sigmas):
        """Compute pairwise OKS between two sets of keypoint detections.

        Uses the average area of both poses as the OKS denominator (as in
        COCO OKS evaluation), which is more stable than using only one side's
        area — especially when keypoint-derived bounding boxes are small or
        noisy.

        Args:
            kpts_a (Tensor): [A, K, 3] keypoints (x, y, score).
            bboxes_a (Tensor): [A, 4+] bounding boxes.
            kpts_b (Tensor): [B, K, 3] keypoints (x, y, score).
            bboxes_b (Tensor): [B, 4+] bounding boxes.
            sigmas (Tensor): [K] per-keypoint sigma values.

        Returns:
            Tensor: [A, B] pairwise OKS values.
        """
        variances = (sigmas * 2) ** 2  # [K]

        areas_a = ((bboxes_a[:, 2] - bboxes_a[:, 0]) *
                   (bboxes_a[:, 3] - bboxes_a[:, 1]))  # [A]
        areas_b = ((bboxes_b[:, 2] - bboxes_b[:, 0]) *
                   (bboxes_b[:, 3] - bboxes_b[:, 1]))  # [B]

        # Use average area of both poses (COCO-style OKS denominator)
        avg_areas = (areas_a[:, None] + areas_b[None, :]) / 2.0  # [A, B]

        a_xy = kpts_a[:, :, :2]   # [A, K, 2]
        b_xy = kpts_b[:, :, :2]   # [B, K, 2]
        a_vis = kpts_a[:, :, 2]   # [A, K]
        b_vis = kpts_b[:, :, 2]   # [B, K]

        # Squared distances: [A, B, K]
        diff = a_xy[:, None, :, :] - b_xy[None, :, :, :]  # [A, B, K, 2]
        sq_dist = (diff ** 2).sum(-1)  # [A, B, K]

        # Denominator: 2 * avg_area * var_k  → [A, B, K]
        denom = 2.0 * avg_areas[:, :, None] * variances[None, None, :]
        denom = denom.clamp(min=1e-6)

        # Per-keypoint similarity
        ks = torch.exp(-sq_dist / denom)  # [A, B, K]

        # Visibility mask: both keypoints must be visible
        vis_mask = (a_vis[:, None, :] > 0) & (b_vis[None, :, :] > 0)  # [A, B, K]

        # OKS = mean over visible keypoints
        n_vis = vis_mask.float().sum(-1).clamp(min=1.0)  # [A, B]
        oks = (ks * vis_mask.float()).sum(-1) / n_vis     # [A, B]

        return oks

    @staticmethod
    def _nms_track_detect(result_item, n_track, oks_thr=0.9):
        """Remove detect query outputs that overlap with track query outputs.

        TrackFormer (§3.3): after decoding, NMS is applied between track and
        detect outputs.  Track queries take priority — any detect query whose
        OKS with a track query exceeds ``oks_thr`` is suppressed.

        Uses a HIGH threshold (default 0.9, as recommended by TrackFormer) to
        only remove near-duplicate detections while preserving recall.

        Uses OKS with the *average area* of both poses as denominator (COCO
        convention), which is more stable than using only one side's area.

        Args:
            result_item (tuple): (det_bboxes, det_labels, det_kpts, bbox_index)
                as returned by ``get_bboxes``.  det_kpts has shape [N, K, 3]
                with (x, y, score) per keypoint.
            n_track (int): Number of track queries (first n_track in the
                original query ordering).
            oks_thr (float): OKS threshold for suppression. Use high values
                (e.g. 0.9) to only suppress true duplicates.

        Returns:
            tuple: Filtered (det_bboxes, det_labels, det_kpts, bbox_index).
        """
        det_bboxes, det_labels, det_kpts, bbox_index = result_item
        if len(det_bboxes) == 0 or n_track == 0:
            return result_item

        # Identify which results came from track vs detect queries
        is_track = bbox_index < n_track  # bool [N]
        if not is_track.any() or is_track.all():
            return result_item

        sigmas = PAVE._get_oks_sigmas(det_kpts)

        track_kpts = det_kpts[is_track]       # [M, K, 3]
        track_bboxes = det_bboxes[is_track]   # [M, 5]
        detect_inds = (~is_track).nonzero(as_tuple=True)[0]
        detect_kpts = det_kpts[detect_inds]   # [D, K, 3]
        detect_bboxes = det_bboxes[detect_inds]  # [D, 5]

        keep = torch.ones(len(det_bboxes), dtype=torch.bool,
                          device=det_bboxes.device)

        if len(detect_kpts) > 0 and len(track_kpts) > 0:
            # [D, M] pairwise OKS using average area
            oks = PAVE._compute_pairwise_oks(
                detect_kpts, detect_bboxes, track_kpts, track_bboxes, sigmas)

            # Suppress detect queries whose max OKS with any track query > threshold
            max_oks, _ = oks.max(dim=1)  # [D]
            suppress = max_oks > oks_thr
            keep[detect_inds[suppress]] = False

        return (det_bboxes[keep], det_labels[keep],
                det_kpts[keep], bbox_index[keep])

    @staticmethod
    def _oks_nms_all(result_item, oks_thr=0.9):
        """Apply greedy OKS-NMS across ALL detections (track + detect).

        TrackFormer paper (§3.3) applies NMS with a high IoU threshold
        (sigma_NMS = 0.9) to remove strongly overlapping duplicate predictions
        that the decoder self-attention could not resolve.

        This is applied AFTER _nms_track_detect as an additional cleanup step.

        Args:
            result_item (tuple): (det_bboxes, det_labels, det_kpts, bbox_index).
            oks_thr (float): OKS threshold for suppression.

        Returns:
            tuple: Filtered (det_bboxes, det_labels, det_kpts, bbox_index).
        """
        det_bboxes, det_labels, det_kpts, bbox_index = result_item
        N = len(det_bboxes)
        if N <= 1:
            return result_item

        sigmas = PAVE._get_oks_sigmas(det_kpts)

        # Compute full NxN pairwise OKS
        oks_matrix = PAVE._compute_pairwise_oks(
            det_kpts, det_bboxes, det_kpts, det_bboxes, sigmas)  # [N, N]

        # Greedy NMS: iterate in descending score order
        scores = det_bboxes[:, 4]  # confidence scores
        order = scores.argsort(descending=True)
        keep_mask = torch.ones(N, dtype=torch.bool, device=det_bboxes.device)

        for i in range(N):
            idx = order[i].item()
            if not keep_mask[idx]:
                continue
            # Suppress all lower-scoring detections with OKS > threshold
            for j in range(i + 1, N):
                jdx = order[j].item()
                if keep_mask[jdx] and oks_matrix[idx, jdx] > oks_thr:
                    keep_mask[jdx] = False

        return (det_bboxes[keep_mask], det_labels[keep_mask],
                det_kpts[keep_mask], bbox_index[keep_mask])

    def merge_aug_results(self, aug_bboxes, aug_kpts, aug_scores, img_metas):
        """Merge augmented detection bboxes and keypoints.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4).
            aug_kpts (list[Tensor] or None): shape (n, K, 2).
            img_metas (list): meta information.

        Returns:
            tuple: (bboxes, kpts, scores).
        """
        recovered_bboxes = []
        recovered_kpts = []
        for bboxes, kpts, img_info in zip(aug_bboxes, aug_kpts, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                       flip_direction)
            kpts = kpt_mapping_back(kpts, img_shape, scale_factor, flip,
                                    flip_direction)
            recovered_bboxes.append(bboxes)
            recovered_kpts.append(kpts)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        kpts = torch.cat(recovered_kpts, dim=0)
        if aug_scores is None:
            return bboxes, kpts
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, kpts, scores

    def aug_test(self, imgs, img_metas, rescale=False):
        feats = self.extract_feats(imgs)
        aug_bboxes = []
        aug_scores = []
        aug_kpts = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.bbox_head(x, img_meta)
            # Strip n_track from outs before passing to get_bboxes
            bbox_list = self.bbox_head.get_bboxes(
                *outs[:-1], img_meta, rescale=False)

            for item in bbox_list:
                det_bboxes, det_labels, det_kpts = item[0], item[1], item[2]
                aug_bboxes.append(det_bboxes[:, :4])
                aug_scores.append(det_bboxes[:, 4])
                aug_kpts.append(det_kpts[..., :2])

        merged_bboxes, merged_kpts, merged_scores = self.merge_aug_results(
            aug_bboxes, aug_kpts, aug_scores, img_metas)

        merged_scores = merged_scores.unsqueeze(1)
        padding = merged_scores.new_zeros(merged_scores.shape[0], 1)
        merged_scores = torch.cat([merged_scores, padding], dim=-1)
        det_bboxes, det_labels, keep_inds = multiclass_nms(
            merged_bboxes,
            merged_scores,
            self.test_cfg.score_thr,
            self.test_cfg.nms,
            self.test_cfg.max_per_img,
            return_inds=True)
        det_kpts = merged_kpts[keep_inds]
        det_kpts = torch.cat(
            (det_kpts, det_kpts.new_ones(det_kpts[..., :1].shape)), dim=2)

        bbox_kpt_results = [
            bbox_kpt2result(det_bboxes, det_labels, det_kpts,
                            self.bbox_head.num_classes)
        ]
        return bbox_kpt_results

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=4,
                    font_size=10,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'.
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'.
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None.
            thickness (int): Thickness of lines. Default: 2.
            font_size (int): Font size of texts. Default: 13.
            win_name (str): The window name. Default: ''.
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, keypoint_result = result
            segm_result = None
        else:
            bbox_result, segm_result, keypoint_result = result, None, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # draw keypoints
        keypoints = None
        if keypoint_result is not None:
            keypoints = np.vstack(keypoint_result)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = self.imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            keypoints,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

    def imshow_det_bboxes(self,
                          img,
                          bboxes,
                          labels,
                          segms=None,
                          keypoints=None,
                          class_names=None,
                          score_thr=0,
                          bbox_color='green',
                          text_color='green',
                          mask_color=None,
                          thickness=4,
                          font_size=8,
                          win_name='',
                          show=True,
                          wait_time=0,
                          out_file=None):
        """Draw bboxes and class labels (with scores) on an image.

        Args:
            img (str or ndarray): The image to be displayed.
            bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
                (n, 5).
            labels (ndarray): Labels of bboxes.
            segms (ndarray or None): Masks, shaped (n,h,w) or None.
            keypoints (ndarray): keypoints (with scores), shaped (n, K, 3).
            class_names (list[str]): Names of each classes.
            score_thr (float): Minimum score of bboxes to be shown. Default: 0.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
                The tuple of color should be in BGR order. Default: 'green'.
                text_color (str or tuple(int) or :obj:`Color`):Color of texts.
                The tuple of color should be in BGR order. Default: 'green'.
            mask_color (str or tuple(int) or :obj:`Color`, optional):
                Color of masks. The tuple of color should be in BGR order.
                Default: None.
            thickness (int): Thickness of lines. Default: 2.
            font_size (int): Font size of texts. Default: 13.
            show (bool): Whether to show the image. Default: True.
            win_name (str): The window name. Default: ''.
            wait_time (float): Value of waitKey param. Default: 0.
            out_file (str, optional): The filename to write the image.
                Default: None.

        Returns:
            ndarray: The image with bboxes drawn on it.
        """
        assert bboxes.ndim == 2, \
            f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
        assert labels.ndim == 1, \
            f' labels ndim should be 1, but its ndim is {labels.ndim}.'
        assert bboxes.shape[0] == labels.shape[0], \
            'bboxes.shape[0] and labels.shape[0] should have the same length.'
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
            f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
        img = mmcv.imread(img).astype(np.uint8)

        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            if segms is not None:
                segms = segms[inds, ...]
            if keypoints is not None:
                keypoints = keypoints[inds, ...]

        num_keypoint = keypoints.shape[1]
        if num_keypoint == 14:
            colors_hp = [(169, 209, 142), (255, 255, 0), (169, 209, 142),
                         (255, 255, 0), (169, 209, 142), (255, 255, 0),
                         (0, 176, 240), (252, 176, 243), (0, 176, 240),
                         (252, 176, 243), (0, 176, 240), (252, 176, 243),
                         (236, 6, 124), (236, 6, 124)]
        elif num_keypoint == 15:
            colors_hp = [(169, 209, 142), (255, 255, 0), (169, 209, 142),
                         (255, 255, 0), (169, 209, 142), (255, 255, 0),
                         (0, 176, 240), (252, 176, 243), (0, 176, 240),
                         (252, 176, 243), (0, 176, 240), (252, 176, 243),
                         (236, 6, 124), (236, 6, 124),(252, 176, 243)]
        elif num_keypoint == 17:
            colors_hp = [(236, 6, 124), (236, 6, 124), (236, 6, 124),
                         (236, 6, 124), (236, 6, 124), (169, 209, 142),
                         (255, 255, 0), (169, 209, 142), (255, 255, 0),
                         (169, 209, 142), (255, 255, 0), (0, 176, 240),
                         (252, 176, 243), (0, 176, 240), (252, 176, 243),
                         (0, 176, 240), (252, 176, 243)]
        else:
            raise ValueError(f'unsupported keypoint amount {num_keypoint}')
        colors_hp = [color[::-1] for color in colors_hp]
        colors_hp = [color_val_matplotlib(color) for color in colors_hp]

        if num_keypoint == 14:
            edges = [
                [0, 2],
                [2, 4],
                [1, 3],
                [3, 5],  # arms
                [0, 1],
                [0, 6],
                [1, 7],  # body
                [6, 8],
                [8, 10],
                [7, 9],
                [9, 11],  # legs
                [12, 13]
            ]  # neck
            ec = [(169, 209, 142),
                  (169, 209, 142), (255, 255, 0), (255, 255, 0), (255, 102, 0),
                  (0, 176, 240), (252, 176, 243), (0, 176, 240), (0, 176, 240),
                  (252, 176, 243), (252, 176, 243), (236, 6, 124)]
        elif num_keypoint == 15:
            edges = [
                [0, 2],
                [0, 1],
                [1, 3],
                [1, 4],
                [3, 5],  
                [4, 6],
                [3, 9],  
                [4, 10],
                [5, 7],
                [6, 8],
                [9, 11],  
                [10, 12],
                [11, 13],
                [12, 14]
            ] 
            ec = [(169, 209, 142),
                  (169, 209, 142), (255, 255, 0), (255, 255, 0), (255, 102, 0),
                  (0, 176, 240), (252, 176, 243), (0, 176, 240), (0, 176, 240),
                  (252, 176, 243), (252, 176, 243), (236, 6, 124), (236, 6, 124), (236, 6, 124)]
        elif num_keypoint == 17:
            edges = [
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 4],  # head
                [5, 7],
                [7, 9],
                [6, 8],
                [8, 10],  # arms
                [5, 6],
                [5, 11],
                [6, 12],  # body
                [11, 13],
                [13, 15],
                [12, 14],
                [14, 16]
            ]  # legs
            ec = [(236, 6, 124), (236, 6, 124), (236, 6, 124), (236, 6, 124),
                  (169, 209, 142),
                  (169, 209, 142), (255, 255, 0), (255, 255, 0), (255, 102, 0),
                  (0, 176, 240), (252, 176, 243), (0, 176, 240), (0, 176, 240),
                  (252, 176, 243), (252, 176, 243)]
        else:
            raise ValueError(f'unsupported keypoint amount {num_keypoint}')
        ec = [color[::-1] for color in ec]
        ec = [color_val_matplotlib(color) for color in ec]

        img = mmcv.bgr2rgb(img)
        width, height = img.shape[1], img.shape[0]
        img = np.ascontiguousarray(img)

        EPS = 1e-2
        fig = plt.figure(win_name, frameon=False)
        plt.title(win_name)
        canvas = fig.canvas
        dpi = fig.get_dpi()
        # add a small EPS to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

        # remove white edges by set subplot margin
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = plt.gca()
        ax.axis('off')

        polygons = []
        color = []
        for i, (bbox, label, kpt) in enumerate(zip(bboxes, labels, keypoints)):
            bbox_int = bbox.astype(np.int32)
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            # polygons.append(Polygon(np_poly))
            # color.append(bbox_color)
            # label_text = class_names[
            #     label] if class_names is not None else f'class {label}'
            # if len(bbox) > 4:
            #     label_text += f'|{bbox[-1]:.02f}'
            # get left-top corner of all keypoints
            bbox_int[0] = np.floor(kpt[:, 0].min()).astype(np.int32)
            bbox_int[1] = np.floor(kpt[:, 1].min() - 30).astype(np.int32)
            label_text = f'{bbox[-1]:.02f}'
            # ax.text(
            #     bbox_int[0],
            #     bbox_int[1],
            #     f'{label_text}',
            #     bbox={
            #         'facecolor': 'black',
            #         'alpha': 0.8,
            #         'pad': 0.7,
            #         'edgecolor': 'none'
            #     },
            #     color=text_color,
            #     fontsize=font_size,
            #     verticalalignment='top',
            #     horizontalalignment='left')
            for j in range(kpt.shape[0]):
                ax.add_patch(
                    Circle(
                        xy=(kpt[j, 0], kpt[j, 1]),
                        radius=2,
                        color=colors_hp[j]))
            for j, e in enumerate(edges):
                poly = [[kpt[e[0], 0], kpt[e[0], 1]],
                        [kpt[e[1], 0], kpt[e[1], 1]]]
                np_poly = np.array(poly).reshape((2, 2))
                polygons.append(Polygon(np_poly))
                color.append(ec[j])
            if segms is not None:
                color_mask = mask_colors[labels[i]]
                mask = segms[i].astype(bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5

        plt.imshow(img)

        p = PatchCollection(
            polygons, facecolor='none', edgecolors=color, linewidths=thickness)
        ax.add_collection(p)

        stream, _ = canvas.print_to_buffer()
        buffer = np.frombuffer(stream, dtype='uint8')
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        img = rgb.astype('uint8')
        img = mmcv.rgb2bgr(img)

        if show:
            # We do not use cvc2 for display because in some cases, opencv will
            # conflict with Qt, it will output a warning: Current thread
            # is not the object's thread. You an refer to
            # https://github.com/opencv/opencv-python/issues/46 for details
            if wait_time == 0:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        plt.close()
        return img
